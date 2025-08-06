import torch
import cv2
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from ultralytics import YOLO

# [1] Device 설정
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print(f"[INFO] Using device: {device}")

# [2] 모델 로드
print("[INFO] Loading YOLOv8...")
yolo = YOLO("yolov8s.pt")

print("[INFO] Loading BLIP...")
blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)

# [3] 비디오 로드
video_path = "/Users/ttaeram/Desktop/LLVM-Drone/assets/3683883-hd_1920_1080_30fps.mp4"
cap = cv2.VideoCapture(video_path)

frame_idx = 0
frame_interval = 30  # 30프레임마다 BLIP 실행

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_idx += 1
    print(f"\n[INFO] Processing frame {frame_idx}")

    # YOLO 객체 탐지
    results = yolo.predict(source=frame, verbose=False)
    objects_info = []

    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        label = yolo.names[cls_id]
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        objects_info.append((label, (x1, y1, x2, y2)))

        # 객체 탐지 결과 화면에 그리기
        cv2.rectangle(frame, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=2)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 255, 0), 2)

    # 일정 프레임마다 BLIP 실행
    if frame_idx % frame_interval == 0:
        print("[INFO] Generating caption with BLIP...")
        pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        blip_inputs = blip_processor(images=pil_image, return_tensors="pt").to(device)

        with torch.no_grad():
            out = blip_model.generate(**blip_inputs)
        caption = blip_processor.decode(out[0], skip_special_tokens=True)
    else:
        caption = "(skipped BLIP for speed)"

    # BLIP 캡션도 영상에 표시 (좌상단)
    cv2.putText(frame, caption, (30, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (255, 255, 255), 2, cv2.LINE_AA)

    # 결과 출력
    print("[YOLO OBJECTS]")
    for label, (x1, y1, x2, y2) in objects_info:
        print(f"- {label} at ({x1},{y1}) → ({x2},{y2})")
    print("[BLIP SCENE DESCRIPTION]")
    print(caption)

    # 영상 프레임 표시
    cv2.imshow("YOLO + BLIP Video", frame)

    # q 키 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
