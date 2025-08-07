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

# [2] 실제 존재하는 BLIP2 모델들로 로드
def load_blip2_model(model_option="opt-2.7b"):    
    model_configs = {
        "opt-2.7b": "Salesforce/blip2-opt-2.7b",           # 권장: 가장 가벼움
        "opt-6.7b": "Salesforce/blip2-opt-6.7b",           # 중간 크기
        "flan-xl": "Salesforce/blip2-flan-t5-xl",          # 원래 모델
    }
    
    if model_option not in model_configs:
        print(f"[WARNING] Unknown option '{model_option}'. Using 'opt-2.7b' instead.")
        model_option = "opt-2.7b"
    
    model_name = model_configs[model_option]
    print(f"[INFO] Loading {model_name}...")
    
    try:
        processor = Blip2Processor.from_pretrained(model_name, use_fast=True)
        
        # 메모리 최적화를 위한 설정
        model_kwargs = {
            "torch_dtype": torch.float16 if device == "cuda" else torch.float32,
            "low_cpu_mem_usage": True,
        }
        
        # CUDA 사용시 device_map 자동 설정
        if device == "cuda":
            model_kwargs["device_map"] = "auto"
        
        model = Blip2ForConditionalGeneration.from_pretrained(
            model_name,
            **model_kwargs
        )
        
        # device_map을 사용하지 않는 경우 수동으로 이동
        if device != "cuda":
            model = model.to(device)
        
        print(f"[SUCCESS] Model {model_name} loaded successfully!")
        return processor, model
        
    except Exception as e:
        print(f"[ERROR] Failed to load model {model_name}: {e}")
        raise e

# YOLO 로드
print("[INFO] Loading YOLOv8...")
yolo = YOLO("yolov8s.pt")

# BLIP2 모델 선택
MODEL_OPTION = "flan-xl"
blip2_processor, blip2_model = load_blip2_model(MODEL_OPTION)
# blip2_processor = Blip2Processor.from_pretrained("Salesforce/blip2-flan-t5-xl", use_fast=True)
# blip2_model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-flan-t5-xl").to(device)

# [3] 메모리 최적화 함수
def clear_gpu_memory():
    """GPU 메모리 정리"""
    if device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

def process_image_with_memory_optimization(image_path):
    """메모리 효율적인 이미지 처리"""
    
    # 이미지 로드 및 크기 조정
    original_image = Image.open(image_path).convert("RGB")
    
    # 메모리 절약을 위해 이미지 크기 제한
    max_size = 512  # 더 작게 설정하여 메모리 절약
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
    
    # BLIP2 프롬프트 생성
    prompt = f"The image contains {summary_text}. Describe this scene in detail, including what each object is doing, their appearance, colors, and the background environment."
    
    print("=" * 60)
    print("[YOLO DETECTION SUMMARY]")
    if label_counts:
        for label, count in label_counts.items():
            print(f"- {label}: {count}")
    else:
        print("- No objects detected")
    print()
    print("[BLIP2 PROMPT]")
    print(prompt)
    print("=" * 60)
    
    # 메모리 정리
    clear_gpu_memory()
    
    # BLIP2 추론
    try:
        with torch.no_grad():  # 메모리 절약을 위해 그래디언트 비활성화
            blip2_inputs = blip2_processor(
                images=original_image, 
                text=prompt, 
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=50  # 프롬프트 길이 제한
            )
            
            # GPU로 입력 이동 (device_map 사용시 자동으로 처리되므로 조건부)
            if device == "cuda" and not hasattr(blip2_model, 'hf_device_map'):
                blip2_inputs = blip2_inputs.to(device)
            elif device != "cuda":
                blip2_inputs = blip2_inputs.to(device)
            
            start_time = time.time()
            
            # 생성 파라미터 - 메모리 효율적으로 설정
            generation_kwargs = {
                "max_new_tokens": 80,  # 더 짧게 설정
                "do_sample": True,
                "temperature": 0.7,
                "top_k": 50,
                "top_p": 0.9,
                "use_cache": False,  # 메모리 절약
                "num_beams": 1,      # beam search 비활성화로 메모리 절약
            }
            
            out = blip2_model.generate(**blip2_inputs, **generation_kwargs)
            
            if device == "cuda":
                torch.cuda.synchronize()
            
            end_time = time.time()
            
            # 결과 디코딩
            caption = blip2_processor.tokenizer.decode(out[0], skip_special_tokens=True)
            
            print("[BLIP2 SCENE DESCRIPTION]")
            print(caption)
            print()
            print(f"[TIME] BLIP2 processing time: {end_time - start_time:.2f} seconds")
            print("=" * 60)
            
            return caption, label_counts
            
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print(f"[ERROR] CUDA out of memory with {MODEL_OPTION} model.")
            print(f"[TIP] Try using 'opt-2.7b' model or reduce image size further.")
            print(f"[TIP] Current image size: {original_image.size}")
        raise e
    finally:
        clear_gpu_memory()

# 메인 실행부
if __name__ == "__main__":
    # 이미지 경로 수정 필요
    image_path = "/home/taeram/Desktop/HMI/assets/test_image.jpg"  # 경로를 실제 경로로 수정

    caption, objects = process_image_with_memory_optimization(image_path)
    print(f"\n[SUCCESS] Image processed successfully!")
    
    # GPU 메모리 사용량 출력
    if device == "cuda":
        memory_allocated = torch.cuda.memory_allocated() / 1024**3
        memory_reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"[MEMORY] Allocated: {memory_allocated:.2f}GB, Reserved: {memory_reserved:.2f}GB")

# 추가: 사용 가능한 모델 확인 함수
def check_available_models():
    models = [
        "Salesforce/blip2-opt-2.7b",
        "Salesforce/blip2-opt-6.7b", 
        "Salesforce/blip2-flan-t5-xl",
    ]
    
    print("[INFO] Checking available BLIP2 models...")
    for model_name in models:
        try:
            from transformers import AutoConfig
            config = AutoConfig.from_pretrained(model_name)
            print(f"✓ {model_name} - Available")
        except Exception as e:
            print(f"✗ {model_name} - Error: {e}")
