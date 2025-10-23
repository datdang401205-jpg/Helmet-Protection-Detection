# train_yolov8.py
# -------------------------------------
# Huấn luyện mô hình YOLOv8 phát hiện người đội / không đội mũ bảo hiểm
# (Chạy được local, gọn nhẹ, phù hợp cho bài tập lớn)
# -------------------------------------

from ultralytics import YOLO
import os

# --------------------------
# 1. Cấu hình tham số huấn luyện
# --------------------------
DATA_PATH = "data/data.yaml"     # File cấu hình dữ liệu
MODEL_NAME = "yolov8n.pt"        # YOLOv8 nhỏ (nhẹ, nhanh)
EPOCHS = 10                      # Giảm số epoch để test nhanh
BATCH_SIZE = 8
IMG_SIZE = 640

# --------------------------
# 2. Khởi tạo mô hình
# --------------------------
print("🚀 Khởi tạo mô hình YOLOv8...")
model = YOLO(MODEL_NAME)

# --------------------------
# 3. Bắt đầu huấn luyện
# --------------------------
print("📦 Bắt đầu huấn luyện...")
results = model.train(
    data=DATA_PATH,
    epochs=EPOCHS,
    batch=BATCH_SIZE,
    imgsz=IMG_SIZE,
    workers=1
)

# --------------------------
# 4. Lưu mô hình sau khi train
# --------------------------
os.makedirs("models", exist_ok=True)
model.export(format="pt")  # lưu dạng .pt

print("✅ Huấn luyện hoàn tất. Mô hình đã được lưu trong thư mục models/")

# --------------------------
# 5. (Tùy chọn) Kiểm tra nhanh bằng ảnh demo
# --------------------------
TEST_IMAGE = "data/demo1.png"
if os.path.exists(TEST_IMAGE):
    print("🔍 Kiểm tra nhanh mô hình trên ảnh demo...")
    trained_model = YOLO("models/best.pt") if os.path.exists("models/best.pt") else model
    trained_model.predict(source=TEST_IMAGE, save=True)
    print("✅ Ảnh kết quả đã lưu trong runs/detect/predict/")
else:
    print("⚠️ Không tìm thấy ảnh demo để test nhanh.")
