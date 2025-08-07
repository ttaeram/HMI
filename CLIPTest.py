import torch
import time
import cv2
from PIL import Image
from transformers import AutoTokenizer, AutoModelForCausalLM
from ultralytics import YOLO
import clip

# [1] Load YOLOv8
print("[INFO] Loading YOLOv8...")
yolo = YOLO('yolov8s.pt')

# [2] Load CLIP
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print(f"[INFO] Using device: {device}")
clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)

# [3] Load lightweight LLM: Phi-2
print("[INFO] Loading Phi-2 model...")
tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2")
llm_model = AutoModelForCausalLM.from_pretrained("microsoft/phi-2").to(device)

# [4] Load input image
image_path = "/Users/ttaeram/Desktop/LLVM-Drone/assets/test_image.jpg"
print(f"[INFO] Loading image: {image_path}")
original_image = Image.open(image_path).convert("RGB")
img_cv = cv2.imread(image_path)

# [5] Object detection with YOLO
print("[INFO] Running YOLO detection...")
start_yolo = time.time()
results = yolo(image_path)
if device == "cuda":
    torch.cuda.synchronize()
end_yolo = time.time()
print(f"[DEBUG] Detected {len(results[0].boxes)} objects.")
print(f"[TIME] YOLO inference time: {end_yolo - start_yolo:.2f} seconds")

# [6] Object classification with CLIP
objects_info = []
clip_times = []

for box in results[0].boxes:
    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
    cropped = original_image.crop((x1, y1, x2, y2))
    clip_input = clip_preprocess(cropped).unsqueeze(0).to(device)

    start_clip = time.time()
    with torch.no_grad():
        image_features = clip_model.encode_image(clip_input)
        possible_labels = ["a person", "a dog", "a car", "a bicycle", "a bus", "a backpack"]
        text_tokens = clip.tokenize(possible_labels).to(device)
        text_features = clip_model.encode_text(text_tokens)
        similarity = image_features @ text_features.T
        best_label = possible_labels[similarity.argmax().item()]
    if device == "cuda":
        torch.cuda.synchronize()
    end_clip = time.time()

    clip_times.append(end_clip - start_clip)
    print(f"[DEBUG] Object detected: {best_label} at ({x1},{y1}),({x2},{y2})")
    print(f"[TIME] CLIP inference time (1 object): {end_clip - start_clip:.2f} seconds")
    objects_info.append((best_label, (x1, y1, x2, y2)))

if clip_times:
    avg_clip_time = sum(clip_times) / len(clip_times)
    print(f"[TIME] Average CLIP inference time per object: {avg_clip_time:.2f} seconds")

# [7] Prompt construction
description = "This image contains the following objects:\n"
for label, (x1, y1, x2, y2) in objects_info:
    description += f"- {label} at location ({x1}, {y1}), ({x2}, {y2})\n"
description += "\nDescribe the scene briefly:"

print("=" * 60)
print("[PROMPT to LLM]")
print(description)
print("=" * 60)

# [8] LLM inference (Phi-2)
inputs = tokenizer(description, return_tensors="pt")
inputs = {k: v.to(device) for k, v in inputs.items()}

start_llm = time.time()
with torch.no_grad():
    outputs = llm_model.generate(
        **inputs,
        max_new_tokens=50,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id
    )
if device == "cuda":
    torch.cuda.synchronize()
end_llm = time.time()
print(f"[TIME] LLM (Phi-2) generation time: {end_llm - start_llm:.2f} seconds")

# [9] Result decoding
result = tokenizer.decode(outputs[0], skip_special_tokens=True)

print("=" * 60)
print("[LLM OUTPUT]")
print(result)
print("=" * 60)
