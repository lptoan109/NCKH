# ==== 1️⃣ CÀI ĐẶT THƯ VIỆN (CHỈ CHẠY MỘT LẦN) ====
!pip install tensorflow tqdm

# ==== 2️⃣ IMPORT CÁC THƯ VIỆN CẦN THIẾT ====
import tensorflow as tf                            # Thư viện chính xử lý AI
import numpy as np                                # Xử lý ma trận, vector
import os                                         # Làm việc với thư mục, file
import shutil                                     # Copy file, thư mục
from tensorflow.keras.preprocessing import image  # Tiện ích xử lý ảnh
from tqdm.notebook import tqdm                    # Hiển thị progress bar đẹp cho Colab

# ==== 3️⃣ KẾT NỐI GOOGLE DRIVE ====
from google.colab import drive
drive.mount('/content/drive')

# ==== 4️⃣ CẤU HÌNH CÁC ĐƯỜNG DẪN ====
IMG_SIZE = (128, 128)  # Kích thước ảnh phù hợp với mô hình EfficientNet
MODEL_PATH = '/content/drive/MyDrive/AIData/efficientnet_cough_model_20250620_1830.keras'  # ✅ Đường dẫn file model đã train
UNLABELED_DIR = '/content/drive/MyDrive/DuLieuHo/unlabeled_23000'                          # ✅ Thư mục chứa ảnh chưa gán nhãn
PSEUDO_LABEL_DIR = '/content/drive/MyDrive/DuLieuHo/pseudo_label'                          # ✅ Thư mục lưu ảnh đã gán nhãn giả
CONFIDENCE_THRESHOLD = 0.9                                                                 # ✅ Ngưỡng confidence để chọn ảnh
ORIGINAL_TRAIN_DIR = '/content/drive/MyDrive/DuLieuHo/dataset/train'                       # ✅ Thư mục chứa dữ liệu train gốc (ảnh đã gán nhãn)
COMBINED_TRAIN_DIR = '/content/drive/MyDrive/DuLieuHo/combined_train'                      # ✅ Thư mục kết hợp ảnh gốc + ảnh gán nhãn giả

# ==== 5️⃣ TẢI MÔ HÌNH ====
model = tf.keras.models.load_model(MODEL_PATH)              # Tải mô hình .keras đã huấn luyện
class_names = sorted(os.listdir(ORIGINAL_TRAIN_DIR))        # Lấy tên các lớp (label) từ thư mục train
print("📂 Các lớp nhận dạng của mô hình:", class_names)

# ==== 6️⃣ TẠO CÁC THƯ MỤC ĐẦU RA PSEUDO LABEL (MỖI CLASS 1 FOLDER) ====
for cls in class_names:
    os.makedirs(os.path.join(PSEUDO_LABEL_DIR, cls), exist_ok=True)  # Tạo folder class nếu chưa có

# ==== 7️⃣ TIẾN HÀNH DỰ ĐOÁN VÀ GÁN NHÃN GIẢ ====
for fname in tqdm(os.listdir(UNLABELED_DIR), desc="Gán nhãn giả"):
    if not fname.lower().endswith('.png'):  # Bỏ qua các file không phải .png
        continue

    fpath = os.path.join(UNLABELED_DIR, fname)

    # Load ảnh, resize đúng kích thước input mô hình
    img = image.load_img(fpath, target_size=IMG_SIZE)
    img_array = image.img_to_array(img) / 255.0        # Chuẩn hóa về [0, 1]
    img_array = np.expand_dims(img_array, axis=0)      # Thêm chiều batch

    # Dự đoán nhãn
    preds = model.predict(img_array, verbose=0)[0]     # Mảng xác suất (ví dụ: [0.01, 0.97, 0.02])
    max_prob = np.max(preds)                           # Lấy xác suất cao nhất
    predicted_label = class_names[np.argmax(preds)]    # Lấy tên class tương ứng

    # Nếu xác suất cao hơn CONFIDENCE_THRESHOLD → lưu vào folder đó
    if max_prob >= CONFIDENCE_THRESHOLD:
        out_path = os.path.join(PSEUDO_LABEL_DIR, predicted_label, fname)
        shutil.copy(fpath, out_path)                   # Copy file vào thư mục đúng class

print("✅ Hoàn tất gán nhãn giả.")

# ==== 8️⃣ TRỘN ẢNH GỐC + ẢNH GÁN NHÃN GIẢ VÀO combined_train ====
for cls in class_names:
    orig_cls_path = os.path.join(ORIGINAL_TRAIN_DIR, cls)       # Đường dẫn thư mục ảnh gốc
    pseudo_cls_path = os.path.join(PSEUDO_LABEL_DIR, cls)       # Đường dẫn thư mục ảnh gán nhãn giả
    combined_cls_path = os.path.join(COMBINED_TRAIN_DIR, cls)   # Thư mục đầu ra kết hợp

    os.makedirs(combined_cls_path, exist_ok=True)               # Tạo folder đích nếu chưa có

    # 📥 Copy ảnh gốc
    for fname in os.listdir(orig_cls_path):
        src = os.path.join(orig_cls_path, fname)
        dst = os.path.join(combined_cls_path, "orig_" + fname)  # Đổi tên ảnh gốc → 'orig_<tên>.png'
        shutil.copy(src, dst)

    # 📥 Copy ảnh gán nhãn giả (nếu có)
    if os.path.exists(pseudo_cls_path):
        for fname in os.listdir(pseudo_cls_path):
            src = os.path.join(pseudo_cls_path, fname)
            dst = os.path.join(combined_cls_path, "pseudo_" + fname)  # Đổi tên ảnh giả → 'pseudo_<tên>.png'
            shutil.copy(src, dst)

print("✅ Đã hoàn tất trộn dữ liệu gốc + dữ liệu gán nhãn giả.")
