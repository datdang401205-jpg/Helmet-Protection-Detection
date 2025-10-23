# src/video_detect.py
# -----------------------------------------------------
# Phát hiện mũ bảo hiểm trong video có sẵn
# -----------------------------------------------------

from ultralytics import YOLO
import os

# 1. Nạp mô hình
model = YOLO("models/best.pt")

# 2. Đường dẫn đến video nguồn
video_path = "data/demo1.mp4"  # đổi tên nếu bạn có video khác

# 3. Tạo thư mục lưu kết quả
save_dir = "runs/video_detect"
os.makedirs(save_dir, exist_ok=True)

# 4. Thực hiện phát hiện và lưu video đầu ra
results = model.predict(
    source=video_path,       # video đầu vào
    conf=0.25,               # ngưỡng confidence
    save=True,               # lưu video kết quả
    project=save_dir,        # thư mục gốc để lưu
    name="helmet_demo"       # tên thư mục con
)

print("✅ Xử lý xong video demo!")
print("📂 Kết quả nằm trong thư mục:")
print(f"   {os.path.join(save_dir, 'helmet_demo')}")
