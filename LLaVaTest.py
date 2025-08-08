import torch
from PIL import Image
from transformers import LlavaProcessor, LlavaForConditionalGeneration
from ultralytics import YOLO
from collections import Counter
import time

# [1] 디바이스 설정
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[INFO] Using device: {device}")

# [2] 모델 로드
print("[INFO] Loading YOLOv8...")
yolo = YOLO("yolov8s.pt")

print("[INFO] Loading LLaVA model...")
model_id = "llava-hf/llava-1.5-7b-hf"
processor = LlavaProcessor.from_pretrained(model_id)
model = LlavaForConditionalGeneration.from_pretrained(model_id).to(device)
print("[SUCCESS] All models loaded.")

# [3] 이미지 처리 및 설명 생성
def process_image_with_llava(image_path):
    # 이미지 열기
    image = Image.open(image_path).convert("RGB")

    # YOLO 객체 탐지
    print("[INFO] Running YOLO object detection...")
    results = yolo(image_path)
    labels = [yolo.names[int(box.cls[0])] for box in results[0].boxes]
    label_counts = Counter(labels)

    # 객체 요약 텍스트 생성
    if label_counts:
        summary = ", ".join(f"{count} {label}(s)" for label, count in label_counts.items())
    else:
        summary = "no specific objects detected"

    # 프롬프트 생성
    prompt = (
        f"The image contains {summary}. "
        f"Describe the scene in detail, including object actions, appearance, and background."
    )

    print("=" * 60)
    print("[YOLO DETECTION SUMMARY]")
    for label, count in label_counts.items():
        print(f"- {label}: {count}")
    print("\n[LLaVA PROMPT]")
    print(prompt)
    print("=" * 60)

    # 입력 구성 및 추론
    inputs = processor(text=prompt, images=image, return_tensors="pt").to(device)
    start = time.time()
    outputs = model.generate(**inputs)
    end = time.time()

    # 결과 출력
    caption = processor.tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("[LLaVA DESCRIPTION]")
    print(caption)
    print(f"\n[INFO] LLaVA inference time: {end - start:.2f} sec")
    print("=" * 60)

    return caption, label_counts

# [4] 실행부
if __name__ == "__main__":
    image_path = "/home/taeram/Desktop/HMI/assets/test_image.jpg"  # 이미지 경로 수정
    caption, objects = process_image_with_llava(image_path)
    print("[SUCCESS] Image processed.")

    if device == "cuda":
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"[GPU MEMORY] Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB")
