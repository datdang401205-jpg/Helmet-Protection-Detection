# Helmet-Protection-Detection

Học phần: MAT3508 – Nhập môn trí tuệ nhân tạo

Học kỳ: Học kỳ 1, Năm học 2025-2026

Trường: VNU-HUS (Đại học Quốc gia Hà Nội – Trường Đại học Khoa học Tự nhiên)

Tên dự án: Helmet Protection Detection AI

Ngày nộp: 30/11/2025

Báo cáo PDF: https://github.com/datdang401205-jpg/Helmet-Protection-Detection/blob/main/B%C3%A1o%20c%C3%A1o%20nh%E1%BA%ADp%20m%C3%B4n%20Tr%C3%AD%20tu%E1%BB%87%20Nh%C3%A2n%20t%E1%BA%A1o.pdf

Slide thuyết trình: https://github.com/datdang401205-jpg/Helmet-Protection-Detection/blob/main/slide%20thuy%E1%BA%BFt%20tr%C3%ACnh%20nh%E1%BA%ADp%20m%C3%B4n%20Tr%C3%AD%20tu%E1%BB%87%20Nh%C3%A2n%20t%E1%BA%A1o.pdf

Kho GitHub: https://github.com/datdang401205-jpg/Helmet-Detection
## 🧩 Giới thiệu
Dự án phát hiện người **có hoặc không đội mũ bảo hiểm** sử dụng mô hình **YOLOv8** của Ultralytics.  
Ứng dụng có thể nhận dạng **trên ảnh, video, hoặc webcam realtime**, hỗ trợ cho hệ thống giám sát giao thông thông minh.

---

## 👥 Nhóm thực hiện
**Nhóm 15 – Môn Nhập môn Trí Tuệ Nhân tạo**

| Họ và tên          | MSSV      | GitHub username          |
|--------------------|-----------|--------------------------|
| Đặng Khánh Đạt     | 23001514  | [@datdang401205-jpg](https://github.com/datdang401205-jpg) |
| Nguyễn Hải Đăng     | 23001516  | [@dawn-ds-15](https://github.com/dawn-ds-15) |
| Trương Mậu Anh     | 23001538  | [@truongmauanh](https://github.com/truongmauanh) |
| Bùi Phương Nam     | 23001498  | [@buiphuongnam23001538](https://github.com/buiphuongnam23001538) |

---

## 🚀 Cấu trúc thư mục
Helmet-Detection/
├── data/ # Datasets và video/ảnh demo
├── models/ # Trọng số mô hình (best.pt)
├── src/ # Code huấn luyện và nhận dạng
│ ├── train_yolov8.py # Huấn luyện YOLOv8
│ ├── detect_helmet.py # Dự đoán ảnh
│ ├── realtime_detect.py # Phát hiện realtime qua webcam
│ └── video_detect.py # Phát hiện trên video
├── app/ # (Tuỳ chọn) Web demo bằng Streamlit
│ ├── app.py
│ └── requirements.txt
└── README.md


---

## ⚙️ Cách chạy dự án

### 1️⃣ Cài đặt môi trường
```bash
pip install -r app/requirements.txt

### 2️⃣ Huấn luyện mô hình (tùy chọn)
```bash
python src/train_yolov8.py

### 3️⃣ Phát hiện trên ảnh
```bash
python src/detect_helmet.py

### 4️⃣ Phát hiện realtime qua webcam
```bash
python src/realtime_detect.py

### 5️⃣ Phát hiện trong video có sẵn
```bash
python src/video_detect.py

### 6️⃣ Phát hiện trên livestream YouTube 🆕
#Chạy file này và cung cấp URL livestream cùng đường dẫn file cookies (nếu cần xác thực).
```bash
python src/youtube_detect.py
🧠 Công nghệ sử dụng

YOLOv8 (Ultralytics)

OpenCV

Python

Streamlit (Web demo)

📚 Tài liệu tham khảo

Ultralytics YOLOv8 Docs

Kaggle: Helmet Detection Dataset

Google Colab

🏁 Mục tiêu

Ứng dụng AI trong nhận dạng hình ảnh nhằm phát hiện hành vi không đội mũ bảo hiểm,
góp phần hỗ trợ giám sát giao thông thông minh và nâng cao an toàn đường bộ.
