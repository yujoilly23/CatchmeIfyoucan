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

# ===== 1. 디바이스 및 기본 설정 =====
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
COMPUTE_TYPE = "float16" if torch.cuda.is_available() else "int8"
CHECKPOINT_PATH = "resnet50_frame_v3_best.pth"
MODEL_PATH = "face_landmarker.task"
THRESHOLD = 0.40

# ===== 2. 구글 Gemini 클라이언트 (보안 환경변수 자동 연동) =====
GEMINI_KEY = os.environ.get("GEMINI_API_KEY")
if GEMINI_KEY:
    ai_client = genai.Client(api_key=GEMINI_KEY)
else:
    ai_client = None
    print("경고: GEMINI_API_KEY 환경변수가 설정되지 않았습니다.")

# ===== 3. Faster-Whisper 로드 (한국어 STT) =====
print(f"Faster-Whisper 모델 로딩 중... (Device: {DEVICE})")
whisper_model = WhisperModel("deepdml/faster-whisper-large-v3-turbo-ct2", device=str(DEVICE), compute_type=COMPUTE_TYPE)

# ===== 4. MediaPipe 얼굴 검출 준비 =====
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

# ===== 5. 비전 전처리 함수들 =====
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

# ===== 6. 비전 모델 및 Grad-CAM 로드 =====
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

# ===== 7. Gemini 기반 고정밀 오류 분석 함수 =====
def fact_check_and_correct(transcribed_text):
    if not transcribed_text or transcribed_text.startswith("음성 추출"):
        return "분석할 음성 텍스트가 없습니다."

    if not ai_client:
        return "오류: GEMINI_API_KEY가 설정되지 않아 사실 검증을 수행할 수 없습니다. Space Settings에서 Secret을 등록해주세요."

    prompt = f"""
당신은 대한민국 최고의 전문 팩트체커입니다.
다음은 영상 속 음성을 텍스트로 추출한 내용입니다:
"{transcribed_text}"

위 내용의 사실 관계를 정확히 검증하여, 사실과 다르거나 왜곡된 잘못된 내용이 있는지 분석해 주세요.

[작성 규칙]
1. 왜곡되거나 사실과 다른 내용이 있다면, 어떤 부분이 왜 틀렸는지 핵심 근거와 함께 명확하게 설명해 주세요.
2. 사실 관계에 이상이 없고 모두 맞는 내용이라면 오직 '발견된 오류 없음 (사실과 일치함)'이라고만 출력해 주세요.
3. 불필요한 서론이나 인사말 없이 결과만 바로 작성해 주세요.
"""

    try:
        response = ai_client.models.generate_content(
            model="gemini-3.6-flash",
            contents=prompt
        )
        return response.text.strip()
    except Exception as e:
        return f"Gemini 분석 처리 중 오류 발생: {str(e)}"

# ===== 8. 메인 통합 추론 함수 =====
def predict_video(video_path, decision_rule):
    if video_path is None:
        return "영상을 업로드해주세요", "", "", None, None

    # (1) Faster-Whisper 한국어 전사
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

    # (2) Gemini 초정밀 팩트체크 실행
    correction_result = fact_check_and_correct(extracted_text)

    # (3) 비디오 딥페이크 판정
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

# ===== 9. Gradio UI =====
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 딥페이크 탐지 및 AI 발화 오류 분석 시스템")
    gr.Markdown("영상 속 **얼굴 합성(딥페이크)** 여부를 판정하고, **음성 인식 텍스트 내 사실 오류 및 왜곡을 검증**합니다.")

    with gr.Row():
        with gr.Column(scale=1):
            video_input = gr.Video(label="검증할 영상 업로드")
            decision_rule = gr.Radio(
                ["가중치 결합 판정 (Hybrid)", "단순 평균 (Mean)", "최대 확률 (Max Rule)"],
                value="가중치 결합 판정 (Hybrid)",
                label="영상 판정 알고리즘"
            )
            submit_btn = gr.Button("영상 분석 및 오류 검증 시작", variant="primary")

        with gr.Column(scale=1):
            res = gr.Textbox(label="영상 딥페이크 판정 결과", lines=4)
            with gr.Row():
                img1 = gr.Image(label="분석 프레임")
                img2 = gr.Image(label="Grad-CAM 의심 영역")

    gr.Markdown("---")
    with gr.Row():
        with gr.Column(scale=1):
            txt_raw = gr.Textbox(label="음성 인식 원문 (Faster-Whisper STT)", lines=6)
        with gr.Column(scale=1):
            txt_corrected = gr.Textbox(label="발견된 오류 분석 결과 (Gemini AI)", lines=6)

    submit_btn.click(
        predict_video,
        inputs=[video_input, decision_rule],
        outputs=[res, txt_raw, txt_corrected, img1, img2]
    )

if __name__ == "__main__":
    demo.launch(share=True)