# src/realtime_detect.py
# -----------------------------------------------------
# Phát hiện mũ bảo hiểm theo thời gian thực và lưu video
# -----------------------------------------------------

from ultralytics import YOLO
import cv2
import time
import os

# 1. Nạp mô hình
model = YOLO("models/best.pt")

# 2. Mở webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Không thể mở webcam. Thử cap = cv2.VideoCapture(1)")
    exit()

print("✅ Webcam đã mở thành công.")
print("🚀 Nhấn 'q' để dừng quay và lưu video...")

# 3. Tạo thư mục lưu video
save_dir = "runs/realtime_video"
os.makedirs(save_dir, exist_ok=True)

# 4. Chuẩn bị file video output
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
timestamp = time.strftime("%Y%m%d-%H%M%S")
output_path = os.path.join(save_dir, f"helmet_realtime_{timestamp}.mp4")

# Lấy kích thước khung hình từ webcam
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
out = cv2.VideoWriter(output_path, fourcc, 20.0, (frame_width, frame_height))

# 5. Chạy phát hiện và ghi video
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Dự đoán
    results = model(frame, conf=0.25)
    annotated_frame = results[0].plot()

    # Hiển thị kết quả lên màn hình
    cv2.imshow("Helmet Detection - Realtime", annotated_frame)

    # Ghi frame có bounding box vào video
    out.write(annotated_frame)

    # Nhấn 'q' để dừng
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 6. Giải phóng tài nguyên
cap.release()
out.release()
cv2.destroyAllWindows()

print(f"✅ Video đã lưu tại: {output_path}")
