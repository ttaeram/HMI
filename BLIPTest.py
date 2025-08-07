import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from ultralytics import YOLO
import cv2
import time

# [1] Device 설정
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print(f"[INFO] Using device: {device}")

# [2] 모델 로드
print("[INFO] Loading YOLOv8...")
yolo = YOLO("yolov8s.pt")

print("[INFO] Loading BLIP...")
blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)

# [3] 이미지 로드
image_path = "/home/taeram/Desktop/HMI/assets/test_image.jpg"
original_image = Image.open(image_path).convert("RGB")
img_cv = cv2.imread(image_path)

# [4] YOLO로 객체 탐지
print("[INFO] Running YOLO detection...")
results = yolo(image_path)
objects_info = []

for box in results[0].boxes:
    cls_id = int(box.cls[0])
    label = yolo.names[cls_id]
    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
    print(f"[DEBUG] Object detected: {label} at ({x1},{y1}),({x2},{y2})")
    objects_info.append((label, (x1, y1, x2, y2)))

# [5] BLIP으로 전체 장면 설명 (시간 측정 포함)
print("[INFO] Generating caption with BLIP...")
blip_inputs = blip_processor(images=original_image, return_tensors="pt").to(device)

start_blip = time.time()
out = blip_model.generate(**blip_inputs)
if device == "cuda":
    torch.cuda.synchronize()  # GPU 대기 (정확한 시간 측정)
end_blip = time.time()

caption = blip_processor.decode(out[0], skip_special_tokens=True)

# [6] 결과 출력
print("=" * 60)
print("[YOLO OBJECTS]")
for label, (x1, y1, x2, y2) in objects_info:
    print(f"- {label} at ({x1},{y1}) → ({x2},{y2})")
print()

print("[BLIP SCENE DESCRIPTION]")
print(caption)
print()
print(f"[TIME] BLIP captioning time: {end_blip - start_blip:.2f} seconds")
print("=" * 60)
