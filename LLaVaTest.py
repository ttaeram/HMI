import os
import time
from collections import Counter

import torch
from PIL import Image
from ultralytics import YOLO
from transformers import AutoModel, AutoProcessor, BitsAndBytesConfig

# Config
MODEL_ID = "visheratin/MC-LLaVA-3b"
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "80"))
USE_4BIT = os.getenv("USE_4BIT", "true").lower() in ["1", "true", "yes"]

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[INFO] Using device: {device}")

# Load models
print("[INFO] Loading YOLOv8...")
yolo = YOLO("yolov8s.pt")

print("[INFO] Loading MC-LLaVA-3b...")
processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)

dtype = torch.float16 if device == "cuda" else torch.float32
if device == "cuda" and USE_4BIT:
    try:
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        model = AutoModel.from_pretrained(
            MODEL_ID,
            trust_remote_code=True,
            quantization_config=bnb,
            device_map="auto",
            attn_implementation="eager",
        )
        print("[INFO] Loaded 4bit.")
    except Exception as e:
        print(f"[WARN] 4bit failed: {e}\n[INFO] Fallback to {dtype}.")
        model = AutoModel.from_pretrained(
            MODEL_ID,
            trust_remote_code=True,
            torch_dtype=dtype,
            attn_implementation="eager",
        ).to(device)
else:
    model = AutoModel.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
        torch_dtype=dtype,
        attn_implementation="eager",
    ).to(device)

# disable KV cache (버전 충돌 우회)
model.generation_config.use_cache = False
model.config.use_cache = False

# 이미지 토큰 결정
extra = getattr(processor.tokenizer, "additional_special_tokens", None)
image_token = None
if extra:
    for t in extra:
        if "image" in t.lower():
            image_token = t
            break
if image_token is None:
    image_token = "<image>"
print(f"[INFO] Image token: {image_token}")

print("[SUCCESS] All models loaded.")

# Inference
@torch.no_grad()
def process_image_with_mcllava(image_path: str, max_new_tokens: int = MAX_NEW_TOKENS):
    image = Image.open(image_path).convert("RGB")

    # YOLO
    print("[INFO] Running YOLO object detection...")
    results = yolo(image_path)
    labels = [yolo.names[int(b.cls[0])] for b in results[0].boxes]
    counts = Counter(labels)
    summary = ", ".join(f"{c} {lbl}(s)" for lbl, c in counts.items()) if counts else "no specific objects detected"

    # ChatML + 이미지 토큰 포함
    user_text = (
        f"The image contains {summary}. "
        f"Describe the scene in detail, including object actions, appearance, and background."
    )
    prompt = (
        f"<|im_start|>user\n"
        f"{image_token}\n"
        f"{user_text}\n"
        f"<|im_end|>\n"
        f"<|im_start|>assistant"
    )

    print("=" * 60)
    print("[YOLO DETECTION SUMMARY]")
    for lbl, c in counts.items():
        print(f"- {lbl}: {c}")
    print("\n[MCLLAVA PROMPT]")
    print(user_text)
    print("=" * 60)

    # processor(prompt, [image], model, ...)
    inputs = processor(
        prompt,
        [image],
        model,
        max_crops=100,
        num_tokens=728,
        return_tensors="pt",
    )

    # to(device) if needed
    if device == "cuda":
        inputs = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in inputs.items()}

    start = time.time()
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        use_cache=False,
        eos_token_id=processor.tokenizer.eos_token_id,
        pad_token_id=processor.tokenizer.eos_token_id,
    )
    end = time.time()

    text = processor.tokenizer.decode(outputs[0])
    text = text.replace(prompt, "").replace("<|im_end|>", "").strip()

    print("[MCLLAVA DESCRIPTION]")
    print(text)
    print(f"\n[INFO] MC-LLaVA inference time: {end - start:.2f} sec")
    print("=" * 60)

    return text, counts

# Main
if __name__ == "__main__":
    image_path = "/Users/ttaeram/Desktop/HMI/assets/test_image.jpg"
    caption, objects = process_image_with_mcllava(image_path, max_new_tokens=MAX_NEW_TOKENS)
    print("[SUCCESS] Image processed.")

    if device == "cuda":
        alloc = torch.cuda.memory_allocated() / 1024**3
        reserv = torch.cuda.memory_reserved() / 1024**3
        print(f"[GPU MEMORY] Allocated: {alloc:.2f} GB, Reserved: {reserv:.2f} GB")
