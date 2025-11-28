# -*- coding: utf-8 -*-
# detect_helmet.py
# ------------------------------------------------
# Phát hiện mũ bảo hiểm bằng mô hình YOLOv8 (local)

from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os

# ------------------------------------------------
# 1. Cấu hình đường dẫn và tham số
# ------------------------------------------------
MODEL_PATH = "/Users/Admin/Documents/clone_AI/Helmet-Detection/models/best.pt"   # Mô hình YOLO đã huấn luyện
TEST_IMAGE = "/Users\Admin/Documents/clone_AI/Helmet-Detection/data/demo5.png"   # Ảnh test
CONF_THRES = 0.25               # Ngưỡng confidence

# ------------------------------------------------
# 2. Nạp mô hình
# ------------------------------------------------
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError("❌ Không tìm thấy file best.pt trong thư mục models/")

model = YOLO(MODEL_PATH)

# ------------------------------------------------
# 3. Dự đoán trên ảnh
# ------------------------------------------------
print("🚀 Đang thực hiện phát hiện mũ bảo hiểm...")
results = model(source=TEST_IMAGE, conf=CONF_THRES, save=True)
print("✅ Dự đoán hoàn tất. Kết quả lưu trong thư mục runs/detect/")

# ------------------------------------------------
# 4. Hiển thị ảnh kết quả
# ------------------------------------------------
result_img_path = os.path.join("runs", "detect", "predict", os.path.basename(TEST_IMAGE))
if os.path.exists(result_img_path):
    img = cv2.imread(result_img_path)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()
else:
    print("⚠️ Không tìm thấy ảnh kết quả trong thư mục runs/detect/predict/")
