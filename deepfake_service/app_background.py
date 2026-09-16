import os
import cv2
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import urllib.request
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from torchvision import models, transforms
import gradio as gr
from faster_whisper import WhisperModel
from google import genai

# 1. 기본 설정 및 디바이스
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
COMPUTE_TYPE = "float16" if torch.cuda.is_available() else "int8"
CHECKPOINT_PATH = "resnet50_frame_v3_best.pth"
MODEL_PATH = "face_landmarker.task"
THRESHOLD = 0.40

# 2. 구글 Gemini 클라이언트
GEMINI_KEY = os.environ.get("GEMINI_API_KEY")
if GEMINI_KEY:
    ai_client = genai.Client(api_key=GEMINI_KEY)
else:
    ai_client = None
    print("경고: GEMINI_API_KEY 환경변수가 설정되지 않았습니다.")

# 3. Faster-Whisper 로드
print(f"Faster-Whisper 모델 로딩 중... (Device: {DEVICE})")
whisper_model = WhisperModel("deepdml/faster-whisper-large-v3-turbo-ct2", device=str(DEVICE), compute_type=COMPUTE_TYPE)

# 4. MediaPipe 얼굴 검출 설정
if not os.path.exists(MODEL_PATH):
    url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
    urllib.request.urlretrieve(url, MODEL_PATH)

base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.FaceLandmarkerOptions(base_options=base_options, num_faces=1, min_face_detection_confidence=0.3)
detector = vision.FaceLandmarker.create_from_options(options)

MOUTH_LANDMARKS = [
    61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 78, 95, 88, 178, 87, 14, 
    317, 402, 318, 324, 308, 13, 312, 311, 310, 415, 0, 267, 269, 270, 409
]

# 5. 비전 전처리 함수
def apply_mouth_crop(frame, img_size=224, padding_ratio=0.3, detect_max_dim=480):
    h_orig, w_orig = frame.shape[:2]
    small = cv2.resize(
        frame, 
        (int(w_orig * (detect_max_dim / max(h_orig, w_orig))), int(h_orig * (detect_max_dim / max(h_orig, w_orig))))
    ) if max(h_orig, w_orig) > detect_max_dim else frame
    
    rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
    results = detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_small))
    if not results.face_landmarks:
        return None
        
    landmarks = results.face_landmarks[0]
    mouth_points = np.array([(int(landmarks[idx].x * w_orig), int(landmarks[idx].y * h_orig)) for idx in MOUTH_LANDMARKS])
    x_min, y_min = mouth_points.min(axis=0)
    x_max, y_max = mouth_points.max(axis=0)
    
    pad_x, pad_y = int((x_max - x_min) * padding_ratio), int((y_max - y_min) * padding_ratio)
    x1, y1 = max(0, x_min - pad_x), max(0, y_min - pad_y)
    x2, y2 = min(w_orig, x_max + pad_x), min(h_orig, y_max + pad_y)
    
    crop_w, crop_h = x2 - x1, y2 - y1
    if crop_w > crop_h:
        y1, y2 = max(0, y1 - (crop_w - crop_h) // 2), min(h_orig, y2 + (crop_w - crop_h + 1) // 2)
    elif crop_h > crop_w:
        x1, x2 = max(0, x1 - (crop_h - crop_w) // 2), min(w_orig, x2 + (crop_h - crop_w + 1) // 2)
        
    return cv2.resize(frame[y1:y2, x1:x2], (img_size, img_size)) if (x2 - x1) * (y2 - y1) > 0 else None

def try_extract(cap, fps, start_sec):
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(fps * start_sec))
    valid = []
    for _ in range(8):
        ret, frame = cap.read()
        if not ret:
            break
        proc = apply_mouth_crop(frame)
        if proc is not None:
            valid.append(proc)
    return valid

# 6. 비전 모델 및 Grad-CAM 로드
model = models.resnet50(weights=None)
model.fc = nn.Linear(model.fc.in_features, 2)
if os.path.exists(CHECKPOINT_PATH):
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=False)["model_state_dict"])
model = model.to(DEVICE).eval()

class GradCAM:
    def __init__(self, model, target):
        self.model = model
        self.activations = None
        self.gradients = None
        target.register_forward_hook(lambda m, i, o: setattr(self, 'activations', o.detach()))
        target.register_full_backward_hook(lambda m, gi, go: setattr(self, 'gradients', go[0].detach()))

    def __call__(self, x, class_idx=1):
        logits = self.model(x)
        self.model.zero_grad()
        logits[0, class_idx].backward()
        cam = (self.gradients.mean(dim=(2, 3), keepdim=True) * self.activations).sum(dim=1, keepdim=True)
        cam = torch.relu(cam).squeeze().cpu().numpy()
        return (cam - cam.min()) / (cam.max() - cam.min() + 1e-8) if cam.max() > 0 else cam

gradcam = GradCAM(model, model.layer4[-1])
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 7. 사람 작성 스타일의 간결한 오류 분석 함수
def fact_check_and_correct(transcribed_text):
    if not transcribed_text or transcribed_text.startswith("음성 추출"):
        return "분석할 음성 텍스트가 없습니다."

    if not ai_client:
        return "오류: GEMINI_API_KEY가 설정되지 않았습니다."

    prompt = f"""
아래 문장에서 사실과 다르거나 왜곡된 핵심 오류를 찾아서 간결하게 지적해줘.

문장:
"{transcribed_text}"

작성 규칙:
- 별표(*)나 마크다운 기호는 절대 쓰지 말 것.
- AI 티가 나지 않게 사람이 직접 요약해 주듯 자연스러운 문장으로 작성할 것.
- 길게 늘어놓지 말고 어떤 부분이 실제 사실과 어떻게 다른지 핵심만 2~3줄 내외로 짧게 설명할 것.
- 사실과 일치해서 오류가 없다면 정확히 '발견된 오류 없음 (사실과 일치함)'이라고만 적을 것.
"""

    try:
        response = ai_client.models.generate_content(
            model="gemini-3.6-flash",
            contents=prompt
        )
        return response.text.strip()
    except Exception as e:
        return f"분석 중 오류 발생: {str(e)}"

# 8. 메인 통합 추론 함수
def predict_video(video_path, decision_rule):
    if video_path is None:
        return "영상을 업로드해주세요", "", "", None, None

    # (1) 음성 전사
    try:
        segments, _ = whisper_model.transcribe(
            video_path,
            language="ko",
            beam_size=5,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=500)
        )
        extracted_text = " ".join([seg.text.strip() for seg in segments]).strip()
        if not extracted_text:
            extracted_text = "(음성이 감지되지 않았습니다)"
    except Exception as e:
        extracted_text = f"음성 추출 오류: {str(e)}"

    # (2) 오류 분석 실행
    correction_result = fact_check_and_correct(extracted_text)

    # (3) 비디오 딥페이크 분석
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    duration = (total_frames / fps) if fps > 0 else 0

    best = []
    sample_points = [max(0, duration / 2 - 1.5), 0, max(0, duration - 3)]
    for s in sample_points:
        v = try_extract(cap, fps, s)
        if len(v) > len(best):
            best = v
    cap.release()

    if len(best) < 5:
        return "얼굴 인식 실패 (입 주변을 선명하게 감지할 수 없습니다)", extracted_text, correction_result, None, None

    with torch.no_grad():
        tensors = torch.stack([transform(Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))) for f in best]).to(DEVICE)
        probs = torch.softmax(model(tensors), dim=1)[:, 1].cpu().numpy()

    mean_s, max_s = float(probs.mean()), float(probs.max())
    if decision_rule == "가중치 결합 판정 (Hybrid)":
        final_score = (mean_s * 0.7) + (max_s * 0.3)
    elif decision_rule == "최대 확률 (Max Rule)":
        final_score = max_s
    else:
        final_score = mean_s

    verdict = "[FAKE] 합성 의심 영상" if final_score >= THRESHOLD else "[REAL] 원본 영상"

    # Grad-CAM 생성
    most_idx = int(probs.argmax())
    cam = gradcam(transform(Image.fromarray(cv2.cvtColor(best[most_idx], cv2.COLOR_BGR2RGB))).unsqueeze(0).to(DEVICE))
    heatmap = cv2.applyColorMap((cv2.resize(cam, (224, 224)) * 255).astype(np.uint8), cv2.COLORMAP_JET)
    overlay = (0.5 * heatmap + 0.5 * cv2.cvtColor(cv2.resize(best[most_idx], (224, 224)), cv2.COLOR_BGR2RGB)).astype(np.uint8)

    result_text = (
        f"판정 결과: {verdict}\n\n"
        f"판정 알고리즘: {decision_rule}\n"
        f"최종 위험도: {final_score:.1%}\n"
        f"(평균: {mean_s:.1%}, 최대: {max_s:.1%})"
    )

    return (
        result_text,
        extracted_text,
        correction_result,
        Image.fromarray(cv2.cvtColor(best[most_idx], cv2.COLOR_BGR2RGB)),
        Image.fromarray(overlay)
    )

# 9. Gradio UI
import gradio as gr

# 런타임 오류를 방지하기 위해 가장 안전한 기본 Soft 테마 사용
custom_css = """
.gradio-container {
    background-color: #CFAE78 !important;
}
"""

with gr.Blocks(
    theme=gr.themes.Soft(),
    css=custom_css
) as demo:
    
    # 상단 헤더 영역 (탐정/다크브라운 톤앤매너를 HTML/CSS로 구현)
    gr.HTML("""
        <div style="text-align: center; padding: 30px 20px; background: linear-gradient(135deg, #3F2614 0%, #5C3A21 100%); border-radius: 12px; margin-bottom: 25px; box-shadow: 0 4px 15px rgba(62,38,20,0.2);">
            <h1 style="font-size:42px; font-weight:800; letter-spacing: 3px; color:#E6C280; margin-bottom:10px; font-family: serif;">
                CATCH ME IF YOU CAN
            </h1>
            <p style="font-size:16px; color:#D8C4B6; font-weight: 400; margin: 0; letter-spacing: 0.5px;">
                 교묘한 얼굴 딥페이크 탐지부터 발화 내용의 허위 사실 교정까지 한 번에 판별합니다
            </p>
        </div>
    """)

    with gr.Row(equal_height=True):
        # 왼쪽: 수사 파일 입력 영역
        with gr.Column(scale=1):
            with gr.Group():
                gr.Markdown("###  01. 파일 업로드")
                video_input = gr.Video(label="")
                decision_rule = gr.Radio(
                    ["가중치 결합 판정 (Hybrid)", "단순 평균 (Mean)", "최대 확률 (Max Rule)"],
                    value="가중치 결합 판정 (Hybrid)",
                    label="조사 알고리즘 선택"
                )
                submit_btn = gr.Button("시작", variant="primary", size="lg")

        # 오른쪽: 비주얼 포렌식 결과 영역
        with gr.Column(scale=1):
            with gr.Group():
                gr.Markdown("###  분석 결과")
                res = gr.Textbox(label="합성 판정 요약", lines=3, interactive=False)
                with gr.Row():
                    img1 = gr.Image(label="추적된 입모양 프레임")
                    img2 = gr.Image(label="Grad-CAM 의심 영역 분석")

    gr.Markdown("---")

    # 하단: 오디오 및 팩트체크 수사 영역
    with gr.Row():
        with gr.Column(scale=1):
            with gr.Group():
                gr.Markdown("###  발화 내용")
                txt_raw = gr.Textbox(label="", lines=6, interactive=False, placeholder="음성에서 추출된 텍스트가 여기에 기록됩니다.")
        
        with gr.Column(scale=1):
            with gr.Group():
                gr.Markdown("###  허위사실 교정")
                txt_corrected = gr.Textbox(label="", lines=6, interactive=False, placeholder="발화 내용의 진위 여부 및 오류 분석 결과가 여기에 기록됩니다.")

    # 이벤트 연결 (기존 함수와 연동)
    submit_btn.click(
        predict_video,
        inputs=[video_input, decision_rule],
        outputs=[res, txt_raw, txt_corrected, img1, img2]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, debug=False)