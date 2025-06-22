# === THƯ VIỆN ===
import tensorflow as tf                                 # Thư viện chính cho deep learning
import numpy as np                                      # Xử lý ma trận, vector
import os                                               # Làm việc với file/thư mục
import shutil                                           # Sao chép file giữa các thư mục
from tensorflow.keras.preprocessing import image        # Dùng để load ảnh .png và chuyển thành mảng
from tqdm import tqdm                                   # Hiển thị progress bar

# === CẤU HÌNH THÔNG SỐ ===
IMG_SIZE = (128, 128)                                   # Kích thước ảnh đầu vào phù hợp với EfficientNet
MODEL_PATH = r"G:\My Drive\Tài liệu NCKH\AIData\efficientnet_cough_model_20250620_1830.keras"  # Đường dẫn mô hình đã huấn luyện
UNLABELED_DIR = r"D:\DuLieuHo\unlabeled_23000"          # Thư mục chứa ảnh chưa có nhãn (ảnh spectrogram)
PSEUDO_LABEL_DIR = r"D:\DuLieuHo\pseudo_label"          # Thư mục để lưu ảnh sau khi gán nhãn giả
CONFIDENCE_THRESHOLD = 0.9                              # Ngưỡng xác suất để chấp nhận nhãn dự đoán

ORIGINAL_TRAIN_DIR = r"D:\DuLieuHo\dataset\train"       # Dữ liệu huấn luyện gốc có nhãn thật
COMBINED_TRAIN_DIR = r"D:\DuLieuHo\combined_train"      # Thư mục trộn ảnh gốc + ảnh gán nhãn giả

# === TẢI MÔ HÌNH ===
model = tf.keras.models.load_model(MODEL_PATH)          # Load mô hình đã huấn luyện từ file .keras

# Lấy danh sách tên lớp từ thư mục train gốc (dùng làm nhãn)
class_names = sorted(os.listdir(ORIGINAL_TRAIN_DIR))    # Sắp xếp để đảm bảo đúng thứ tự nhãn

# Tạo thư mục đầu ra tương ứng với từng class (nếu chưa tồn tại)
for cls in class_names:
    os.makedirs(os.path.join(PSEUDO_LABEL_DIR, cls), exist_ok=True)

# === GÁN NHÃN GIẢ CHO ẢNH CHƯA CÓ NHÃN ===
for fname in tqdm(os.listdir(UNLABELED_DIR), desc="Đang gán nhãn giả"):
    if not fname.lower().endswith('.png'):              # Bỏ qua các file không phải ảnh PNG
        continue

    fpath = os.path.join(UNLABELED_DIR, fname)          # Đường dẫn đến ảnh đầu vào

    # Load ảnh và resize về đúng kích thước
    img = image.load_img(fpath, target_size=IMG_SIZE)
    img_array = image.img_to_array(img) / 255.0         # Chuyển ảnh thành mảng và chuẩn hóa [0, 1]
    img_array = np.expand_dims(img_array, axis=0)       # Thêm chiều batch (1, 128, 128, 3)

    # Dự đoán nhãn bằng mô hình
    preds = model.predict(img_array, verbose=0)[0]      # Trả về mảng xác suất với độ dài = số lớp
    max_prob = np.max(preds)                            # Lấy xác suất lớn nhất
    predicted_label = class_names[np.argmax(preds)]     # Lấy nhãn ứng với xác suất lớn nhất

    # Nếu xác suất dự đoán > ngưỡng, lưu ảnh vào thư mục nhãn tương ứng
    if max_prob >= CONFIDENCE_THRESHOLD:
        out_path = os.path.join(PSEUDO_LABEL_DIR, predicted_label, fname)
        shutil.copy(fpath, out_path)                    # Sao chép ảnh vào thư mục đúng nhãn

# === TRỘN DỮ LIỆU GỐC VỚI DỮ LIỆU GIẢ ===
for cls in class_names:
    orig_cls_path = os.path.join(ORIGINAL_TRAIN_DIR, cls)       # Thư mục ảnh gốc
    pseudo_cls_path = os.path.join(PSEUDO_LABEL_DIR, cls)       # Thư mục ảnh gán nhãn giả
    combined_cls_path = os.path.join(COMBINED_TRAIN_DIR, cls)   # Thư mục đầu ra đã trộn

    os.makedirs(combined_cls_path, exist_ok=True)               # Tạo thư mục nếu chưa có

    # ✅ Copy ảnh gốc (có tiền tố 'orig_')
    for fname in tqdm(os.listdir(orig_cls_path), desc=f"Gốc/{cls}"):
        src = os.path.join(orig_cls_path, fname)
        dst = os.path.join(combined_cls_path, "orig_" + fname)
        shutil.copy(src, dst)

    # ✅ Copy ảnh gán nhãn giả (có tiền tố 'pseudo_'), nếu có
    if os.path.exists(pseudo_cls_path):
        for fname in tqdm(os.listdir(pseudo_cls_path), desc=f"Pseudo/{cls}"):
            src = os.path.join(pseudo_cls_path, fname)
            dst = os.path.join(combined_cls_path, "pseudo_" + fname)
            shutil.copy(src, dst)

# === HOÀN TẤT ===
print("\n✅ Đã gán nhãn và trộn dữ liệu hoàn tất. Dữ liệu đã sẵn sàng tại:", COMBINED_TRAIN_DIR)
