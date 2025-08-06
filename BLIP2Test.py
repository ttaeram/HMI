import torch
from PIL import Image
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from ultralytics import YOLO
import cv2
import time
from collections import Counter

# [1] Device 설정
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print(f"[INFO] Using device: {device}")

# [2] 모델 로드
print("[INFO] Loading YOLOv8...")
yolo = YOLO("yolov8s.pt")

print("[INFO] Loading BLIP2-Flan-T5-XL...")
blip2_processor = Blip2Processor.from_pretrained("Salesforce/blip2-flan-t5-xl", use_fast=True)
blip2_model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-flan-t5-xl").to(device)

# [3] 이미지 로드
image_path = "/Users/ttaeram/Desktop/LLVM-Drone/assets/test_image.jpg"
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
    objects_info.append(label)

# [5] 객체 요약
label_counts = Counter(objects_info)
summary_text = ", ".join([f"{count} {label}(s)" for label, count in label_counts.items()])

# [6] BLIP2 프롬프트 생성 및 추론
prompt = f"The image contains {summary_text}. Based on this, describe the scene specifically."

print("=" * 60)
print("[PROMPT]")
print(prompt)
print("=" * 60)

blip2_inputs = blip2_processor(images=original_image, text=prompt, return_tensors="pt").to(device)

start_blip2 = time.time()
out = blip2_model.generate(**blip2_inputs, max_new_tokens=50)
if device == "cuda":
    torch.cuda.synchronize()
end_blip2 = time.time()

caption = blip2_processor.tokenizer.decode(out[0], skip_special_tokens=True)

# [7] 결과 출력
print("[YOLO OBJECT SUMMARY]")
for label, count in label_counts.items():
    print(f"- {label}: {count}")
print()

print("[BLIP2 SCENE DESCRIPTION]")
print(caption)
print()
print(f"[TIME] BLIP2 captioning time: {end_blip2 - start_blip2:.2f} seconds")
print("=" * 60)
