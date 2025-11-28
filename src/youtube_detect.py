import cv2
import yt_dlp
import sys
import os
import time
from ultralytics import YOLO

# ==========================================================
# ⚠️ CÁC THAM SỐ CẦN CHỈNH SỬA
# ==========================================================

# 1. Đường dẫn đến file mô hình đã huấn luyện (best.pt)
# Giả sử file nằm cùng thư mục với file code này
MODEL_PATH = 'models/best.pt' 

# 2. Link YouTube livestream cần xử lý
YOUTUBE_URL = 'https://www.youtube.com/watch?v=muijHPW82vI'

# 3. Đường dẫn đến file cookies (nếu cần xác thực)
# Nếu bạn không gặp lỗi đăng nhập, có thể để là None
COOKIES_FILE_PATH = None 

CONFIDENCE_THRESHOLD = 0.5  # Ngưỡng tin cậy tối thiểu
FPS_TO_PROCESS = 5          # Số frame mong muốn xử lý mỗi giây để giảm tải CPU/GPU
MAX_FRAMES_TO_PROCESS = 1000 # Số lượng frame xử lý tối đa (hoặc None để chạy liên tục)

# ==========================================================
# HÀM TRÍCH XUẤT STREAM URL
# ==========================================================

def get_youtube_stream_url(url, cookie_file=None):
    """Sử dụng yt-dlp để trích xuất URL luồng trực tiếp từ YouTube."""
    ydl_opts = {
        'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]', 
        'quiet': True,
        'noplaylist': True,
        'skip_download': True,
        'force_generic_extractor': True,
    }
    
    if cookie_file and os.path.exists(cookie_file):
        ydl_opts['cookiefile'] = cookie_file
        print(f"Sử dụng cookies từ: {cookie_file}")

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info_dict = ydl.extract_info(url, download=False)
            stream_url = info_dict.get('url', None)
            
            if not stream_url and info_dict.get('entries'):
                 stream_url = info_dict['entries'][0].get('url', None)
            
            if stream_url:
                print(f"✅ Đã lấy được Stream URL. Bắt đầu tải mô hình...")
            return stream_url

    except Exception as e:
        print(f"❌ Lỗi khi trích xuất URL: {e}")
        # In thông báo lỗi cụ thể để người dùng biết phải làm gì
        if "Sign in to confirm you’re not a bot" in str(e) and not cookie_file:
             print("GỢI Ý: Lỗi do YouTube yêu cầu đăng nhập. Vui lòng cung cấp file cookies.")
        return None

# ==========================================================
# HÀM XỬ LÝ CHÍNH
# ==========================================================

def run_detection_on_stream():
    # 1. Trích xuất Stream URL
    stream_url = get_youtube_stream_url(YOUTUBE_URL, COOKIES_FILE_PATH)

    if not stream_url:
        print("🛑 Không thể tiếp tục do không lấy được Stream URL.")
        sys.exit()

    # 2. Tải Mô Hình YOLOv8
    try:
        model = YOLO(MODEL_PATH)
        print(f"✅ Đã tải mô hình thành công từ {MODEL_PATH}")
    except Exception as e:
        print(f"❌ Lỗi khi tải mô hình: {e}")
        sys.exit()

    # 3. Khởi tạo VideoCapture và Vòng Lặp Xử Lý
    # cv2.CAP_FFMPEG thường ổn định hơn khi đọc luồng mạng
    cap = cv2.VideoCapture(stream_url, cv2.CAP_FFMPEG) 

    if not cap.isOpened():
        print("❌ Lỗi: Không thể mở luồng video từ URL. Luồng có thể bị lỗi hoặc chặn.")
        sys.exit()

    print("🚀 Bắt đầu xử lý livestream...")

    # Tính toán khoảng thời gian chờ giữa các frame để kiểm soát FPS xử lý
    wait_time_ms = int(1000 / FPS_TO_PROCESS)
    frame_counter = 0

    while True:
        # Đọc frame
        ret, frame = cap.read()

        if not ret:
            print("🛑 Luồng video kết thúc hoặc bị ngắt kết nối.")
            break
        
        frame_counter += 1
        
        start_time = time.time() # Bắt đầu tính thời gian xử lý frame

        # --- Chạy Nhận Diện (Inference) ---
        # `tracker` chỉ nên dùng nếu bạn muốn theo dõi đối tượng giữa các frame (Không bắt buộc)
        results = model.predict(frame, conf=CONFIDENCE_THRESHOLD, verbose=False)
        
        # Lấy frame đã được vẽ bounding boxes và nhãn
        annotated_frame = results[0].plot()

        # Hiển thị FPS xử lý
        end_time = time.time()
        fps = 1 / (end_time - start_time)
        cv2.putText(annotated_frame, f'FPS: {fps:.2f}', (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # --- HIỂN THỊ KẾT QUẢ ---
        # Chỉ hiển thị kết quả nếu đang chạy trên máy tính
        cv2.imshow("Helmet Detection Live Stream", annotated_frame)
        
        # Dừng nếu nhấn phím 'q' hoặc đạt giới hạn frame
        if cv2.waitKey(wait_time_ms) & 0xFF == ord('q'):
            break
        
        if MAX_FRAMES_TO_PROCESS is not None and frame_counter >= MAX_FRAMES_TO_PROCESS:
            print(f"Đã đạt giới hạn xử lý {MAX_FRAMES_TO_PROCESS} frame.")
            break

    # 4. Dọn dẹp
    cap.release()
    cv2.destroyAllWindows()
    print("Hoàn tất xử lý video stream.")

if __name__ == "__main__":
    run_detection_on_stream()