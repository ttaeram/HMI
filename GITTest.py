import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM
from ultralytics import YOLO
import time
from collections import Counter

# [1] Device 설정
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print(f"[INFO] Using device: {device}")

# [2] GIT 모델 로드 함수
def load_git_model(model_name="microsoft/git-large"):
    print(f"[INFO] Loading {model_name}...")
    try:
        processor = AutoProcessor.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            low_cpu_mem_usage=True,
        ).to(device)
        print(f"[SUCCESS] Model {model_name} loaded successfully!")
        return processor, model
    except Exception as e:
        print(f"[ERROR] Failed to load GIT model: {e}")
        raise e

# [3] YOLOv8 모델 로드
print("[INFO] Loading YOLOv8...")
yolo = YOLO("yolov8s.pt")

# [4] GIT 모델 로딩
GIT_MODEL = "microsoft/git-large"  # 또는 microsoft/git-base
git_processor, git_model = load_git_model(GIT_MODEL)

# [5] GPU 메모리 정리 함수
def clear_gpu_memory():
    if device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

# [6] 이미지 처리 및 캡셔닝
def process_image_with_memory_optimization(image_path):
    # 이미지 로드 및 크기 조정
    original_image = Image.open(image_path).convert("RGB")
    max_size = 512
    if max(original_image.size) > max_size:
        original_image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
        print(f"[INFO] Image resized to: {original_image.size}")
    
    # YOLO 객체 탐지
    print("[INFO] Running YOLO detection...")
    results = yolo(image_path)
    objects_info = []
    
    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        label = yolo.names[cls_id]
        objects_info.append(label)
    
    # 객체 요약
    label_counts = Counter(objects_info)
    if label_counts:
        summary_text = ", ".join([f"{count} {label}(s)" for label, count in label_counts.items()])
    else:
        summary_text = "no specific objects detected"

    # GIT 프롬프트
    prompt = f"The image contains {summary_text}. Describe this scene in detail, including objects, their appearance, and the environment."
    
    print("=" * 60)
    print("[YOLO DETECTION SUMMARY]")
    if label_counts:
        for label, count in label_counts.items():
            print(f"- {label}: {count}")
    else:
        print("- No objects detected")
    print()
    print("[GIT PROMPT]")
    print(prompt)
    print("=" * 60)
    
    clear_gpu_memory()

    # GIT 추론
    try:
        with torch.no_grad():
            inputs = git_processor(
                images=original_image,
                text=prompt,
                return_tensors="pt"
            ).to(device)

            start_time = time.time()
            output = git_model.generate(
                pixel_values=inputs["pixel_values"],
                input_ids=inputs.get("input_ids"),
                attention_mask=inputs.get("attention_mask"),
                max_new_tokens=80,
                do_sample=True,
                temperature=0.7,
                top_k=50,
                top_p=0.9,
                use_cache=False,
                num_beams=1,
            )
            if device == "cuda":
                torch.cuda.synchronize()
            end_time = time.time()

            caption = git_processor.batch_decode(output, skip_special_tokens=True)[0]

            print("[GIT SCENE DESCRIPTION]")
            print(caption)
            print()
            print(f"[TIME] GIT processing time: {end_time - start_time:.2f} seconds")
            print("=" * 60)

            return caption, label_counts

    except RuntimeError as e:
        print(f"[ERROR] {e}")
        if "out of memory" in str(e).lower():
            print(f"[ERROR] CUDA out of memory with {GIT_MODEL} model.")
            print(f"[TIP] Try using a smaller model (e.g., microsoft/git-base) or reduce image size.")
            print(f"[TIP] Current image size: {original_image.size}")
        raise e
    finally:
        clear_gpu_memory()

# [7] 메인 실행
if __name__ == "__main__":
    image_path = "/home/taeram/Desktop/HMI/assets/test_image.jpg"

    caption, objects = process_image_with_memory_optimization(image_path)
    print(f"\n[SUCCESS] Image processed successfully!")

    if device == "cuda":
        memory_allocated = torch.cuda.memory_allocated() / 1024**3
        memory_reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"[MEMORY] Allocated: {memory_allocated:.2f}GB, Reserved: {memory_reserved:.2f}GB")
