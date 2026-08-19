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
import whisper

# ===== 1. 모델 로드 (Whisper base로 격상 및 정확도 옵션 적용) =====
print("Whisper 모델 로딩 중... (한국어 전용 base 모델)")
whisper_model = whisper.load_model("base")

# ===== 2. 기본 설정 =====
CHECKPOINT_PATH = "resnet50_frame_v3_best.pth"
MODEL_PATH = "face_landmarker.task"
THRESHOLD = 0.40
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# MediaPipe 얼굴 검출 준비
if not os.path.exists(MODEL_PATH):
    url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
    urllib.request.urlretrieve(url, MODEL_PATH)
base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.FaceLandmarkerOptions(base_options=base_options, num_faces=1, min_face_detection_confidence=0.3)
detector = vision.FaceLandmarker.create_from_options(options)

MOUTH_LANDMARKS = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 13, 312, 311, 310, 415, 0, 267, 269, 270, 409]

# ===== 3. 전처리 함수들 =====
def apply_mouth_crop(frame, img_size=224, padding_ratio=0.3, detect_max_dim=480):
    h_orig, w_orig = frame.shape[:2]
    small = cv2.resize(frame, (int(w_orig * (detect_max_dim/max(h_orig, w_orig))), int(h_orig * (detect_max_dim/max(h_orig, w_orig))))) if max(h_orig, w_orig) > detect_max_dim else frame
    rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
    results = detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_small))
    if not results.face_landmarks: return None
    landmarks = results.face_landmarks[0]
    mouth_points = np.array([(int(landmarks[idx].x * w_orig), int(landmarks[idx].y * h_orig)) for idx in MOUTH_LANDMARKS])
    x_min, y_min = mouth_points.min(axis=0); x_max, y_max = mouth_points.max(axis=0)
    pad_x, pad_y = int((x_max - x_min) * padding_ratio), int((y_max - y_min) * padding_ratio)
    x1, y1 = max(0, x_min - pad_x), max(0, y_min - pad_y)
    x2, y2 = min(w_orig, x_max + pad_x), min(h_orig, y_max + pad_y)
    crop_w, crop_h = x2 - x1, y2 - y1
    if crop_w > crop_h: y1, y2 = max(0, y1 - (crop_w - crop_h)//2), min(h_orig, y2 + (crop_w - crop_h + 1)//2)
    elif crop_h > crop_w: x1, x2 = max(0, x1 - (crop_h - crop_w)//2), min(w_orig, x2 + (crop_h - crop_w + 1)//2)
    return cv2.resize(frame[y1:y2, x1:x2], (img_size, img_size)) if (x2-x1)*(y2-y1) > 0 else None

def try_extract(cap, fps, start_sec):
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(fps * start_sec))
    valid = []
    for _ in range(8):
        ret, frame = cap.read()
        if not ret: break
        proc = apply_mouth_crop(frame)
        if proc is not None: valid.append(proc)
    return valid

# ===== 4. 모델 및 Grad-CAM 로드 =====
model = models.resnet50(weights=None)
model.fc = nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=False)["model_state_dict"])
model = model.to(DEVICE).eval()

class GradCAM:
    def __init__(self, model, target):
        self.model = model; self.activations = None; self.gradients = None
        target.register_forward_hook(lambda m, i, o: setattr(self, 'activations', o.detach()))
        target.register_full_backward_hook(lambda m, gi, go: setattr(self, 'gradients', go[0].detach()))
    def __call__(self, x, class_idx=1):
        logits = self.model(x); self.model.zero_grad(); logits[0, class_idx].backward()
        cam = (self.gradients.mean(dim=(2, 3), keepdim=True) * self.activations).sum(dim=1, keepdim=True)
        cam = torch.relu(cam).squeeze().cpu().numpy()
        return (cam - cam.min()) / (cam.max() - cam.min() + 1e-8) if cam.max() > 0 else cam

gradcam = GradCAM(model, model.layer4[-1])
transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

# ===== 5. 메인 추론 함수 =====
def predict_video(video_path, decision_rule):
    if video_path is None: return "영상을 업로드해주세요", "", None, None
    
    # Whisper 한국어 최적화 (beam_size 적용)
    try:
        transcription = whisper_model.transcribe(
            video_path, language="ko", beam_size=5, temperature=0.0, best_of=5
        )
        extracted_text = transcription.get("text", "음성 추출 실패")
    except Exception as e:
        extracted_text = f"음성 추출 오류: {str(e)}"

    cap = cv2.VideoCapture(video_path)
    fps, total_frames = cap.get(cv2.CAP_PROP_FPS), cap.get(cv2.CAP_PROP_FRAME_COUNT)
    best = []
    for s in [max(0, (total_frames/fps)/2 - 1.5), 0, max(0, (total_frames/fps) - 3)]:
        v = try_extract(cap, fps, s)
        if len(v) > len(best): best = v
    cap.release()
    
    if len(best) < 5: return "얼굴 인식 실패", extracted_text, None, None
    
    with torch.no_grad():
        probs = torch.softmax(model(torch.stack([transform(Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))) for f in best]).to(DEVICE)), dim=1)[:, 1].cpu().numpy()
    
    # 하이브리드 가중치 로직 (0.7:0.3)
    mean_s, max_s = float(probs.mean()), float(probs.max())
    if decision_rule == "가중치 결합 판정 (Hybrid)":
        final_score = (mean_s * 0.7) + (max_s * 0.3)
    elif decision_rule == "최대 확률 (Max Rule)":
        final_score = max_s
    else:
        final_score = mean_s

    verdict = "🚨 FAKE (합성 의심)" if final_score >= THRESHOLD else "✅ REAL (진짜)"
    
    most_idx = int(probs.argmax())
    cam = gradcam(transform(Image.fromarray(cv2.cvtColor(best[most_idx], cv2.COLOR_BGR2RGB))).unsqueeze(0).to(DEVICE))
    heatmap = cv2.applyColorMap((cv2.resize(cam, (224, 224)) * 255).astype(np.uint8), cv2.COLORMAP_JET)
    overlay = (0.5 * heatmap + 0.5 * cv2.cvtColor(cv2.resize(best[most_idx], (224, 224)), cv2.COLOR_BGR2RGB)).astype(np.uint8)
    
    result_text = f"{verdict}\n\n로직: {decision_rule}\n최종 스코어: {final_score:.1%}\n(평균: {mean_s:.1%}, 최대: {max_s:.1%})"
    return result_text, extracted_text.strip(), Image.fromarray(cv2.cvtColor(best[most_idx], cv2.COLOR_BGR2RGB)), Image.fromarray(overlay)

# ===== 6. Gradio UI =====
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🕵️‍♂️ 딥페이크 탐지 시스템 (하이브리드 & 한국어 STT)")
    with gr.Row():
        with gr.Column(scale=1):
            video_input = gr.Video(label="검증할 영상")
            decision_rule = gr.Radio(["단순 평균 (Mean)", "최대 확률 (Max Rule)", "가중치 결합 판정 (Hybrid)"], value="가중치 결합 판정 (Hybrid)", label="판정 알고리즘")
            submit_btn = gr.Button("분석 시작", variant="primary")
        with gr.Column(scale=1):
            res = gr.Textbox(label="판정 결과", lines=5)
            txt = gr.Textbox(label="Whisper STT (한국어 전용)", lines=3)
            with gr.Row():
                img1 = gr.Image(label="분석 프레임")
                img2 = gr.Image(label="Grad-CAM 결과")
    submit_btn.click(predict_video, [video_input, decision_rule], [res, txt, img1, img2])

if __name__ == "__main__":
    demo.launch(share=True)