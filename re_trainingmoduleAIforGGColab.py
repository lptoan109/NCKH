# ====== THƯ VIỆN ======
import tensorflow as tf                                # Thư viện chính để xây dựng và huấn luyện mô hình deep learning
from tensorflow.keras import layers, models            # Để tạo các layer và mô hình Sequential
from tensorflow.keras.applications import EfficientNetB0  # Import EfficientNetB0 pretrained
import matplotlib.pyplot as plt                        # Thư viện để vẽ biểu đồ
import pandas as pd                                    # Thư viện xử lý dữ liệu dạng bảng
import os                                              # Làm việc với hệ thống file
from datetime import datetime                          # Để tạo tên file có timestamp

# ====== CÀI ĐẶT GOOGLE DRIVE (NẾU LƯU TRÊN DRIVE) ======
from google.colab import drive
drive.mount('/content/drive')

# ====== CẤU HÌNH ======
IMG_SIZE = (128, 128)                                  # Kích thước ảnh đầu vào của EfficientNet
BATCH_SIZE = 32                                        # Số lượng ảnh trong mỗi batch
EPOCHS = 15                                            # Số epoch để fine-tune
COMBINED_TRAIN_DIR = '/content/drive/MyDrive/NCKH/combined_train'  # Thư mục chứa ảnh train (gốc + pseudo)
VAL_DIR = '/content/drive/MyDrive/NCKH/dataset/val'                # Thư mục chứa tập validation
MODEL_INPUT = '/content/drive/MyDrive/NCKH/AIData/efficientnet_cough_model_20250620_1830.keras'  # Đường dẫn mô hình .keras đã huấn luyện trước
OUTPUT_FOLDER = '/content/drive/MyDrive/NCKH/AIData'                # Thư mục để lưu mô hình mới, log, biểu đồ
MODEL_NAME = "efficientnet_cough_finetune"                          # Tên mô hình lưu

# ====== TẢI DỮ LIỆU ======
# Load dữ liệu huấn luyện từ thư mục combined_train
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    COMBINED_TRAIN_DIR,                     # Thư mục dữ liệu
    image_size=IMG_SIZE,                    # Resize về đúng kích thước cho EfficientNet
    batch_size=BATCH_SIZE                   # Kích thước batch
)

# Load dữ liệu validation từ thư mục val
val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    VAL_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

# Lấy danh sách tên lớp từ tập huấn luyện
class_names = train_ds.class_names
num_classes = len(class_names)

# ====== PREFETCH GIÚP TĂNG HIỆU NĂNG ======
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)

# ====== LOAD MÔ HÌNH ĐÃ HUẤN LUYỆN TRƯỚC ======
model = tf.keras.models.load_model(MODEL_INPUT)     # Load mô hình đã train lần 1 từ file .keras

# ====== MỞ KHÓA EfficientNetB0 ĐỂ FINE-TUNE ======
base_model = model.layers[1]                        # EfficientNetB0 là layer thứ 2 trong mô hình Sequential
base_model.trainable = True                         # Cho phép cập nhật trọng số (fine-tune)

# Đóng băng khoảng 70% số tầng đầu tiên để tránh overfitting, chỉ fine-tune phần sâu của mô hình
FINE_TUNE_AT = int(len(base_model.layers) * 0.7)
for layer in base_model.layers[:FINE_TUNE_AT]:
    layer.trainable = False                         # Các tầng đầu vẫn giữ nguyên trọng số pretrained

# ====== COMPILE LẠI MÔ HÌNH ======
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-5),       # Dùng learning rate nhỏ để fine-tune từ từ, tránh làm hỏng pretrained weights
    loss='sparse_categorical_crossentropy',         # Do nhãn dạng số nguyên
    metrics=['accuracy']                            # Đánh giá theo độ chính xác
)

# ====== TIẾN HÀNH FINE-TUNE ======
history_finetune = model.fit(
    train_ds,                                       # Dữ liệu huấn luyện
    validation_data=val_ds,                         # Dữ liệu kiểm tra
    epochs=EPOCHS                                   # Số vòng lặp huấn luyện
)

# ====== GHI LOG VÀ VẼ BIỂU ĐỒ KẾT QUẢ ======
def save_training_log(history, output_folder=OUTPUT_FOLDER, prefix=MODEL_NAME):
    hist = history.history                                       # Lấy toàn bộ log lịch sử huấn luyện
    df = pd.DataFrame({                                          # Tạo DataFrame lưu các giá trị log
        'Epoch': list(range(1, len(hist['accuracy']) + 1)),      # Danh sách số thứ tự các epoch
        'Accuracy': hist['accuracy'],                            # Độ chính xác trên tập train
        'Loss': hist['loss'],                                    # Loss trên tập train
        'Val Accuracy': hist['val_accuracy'],                    # Độ chính xác trên tập validation
        'Val Loss': hist['val_loss']                             # Loss trên tập validation
    })

    os.makedirs(output_folder, exist_ok=True)                    # Tạo thư mục lưu log nếu chưa có
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")         # Tạo tên file gắn timestamp
    excel_path = os.path.join(output_folder, f"{prefix}_finetune_log_{timestamp}.xlsx")
    acc_plot = os.path.join(output_folder, f"{prefix}_finetune_acc_{timestamp}.png")
    loss_plot = os.path.join(output_folder, f"{prefix}_finetune_loss_{timestamp}.png")

    df.to_excel(excel_path, index=False)                         # Lưu log thành file Excel

    # Vẽ biểu đồ Accuracy theo Epoch
    plt.figure(figsize=(8, 5))
    plt.plot(df['Epoch'], df['Accuracy'], label='Train Acc')
    plt.plot(df['Epoch'], df['Val Accuracy'], label='Val Acc')
    plt.title('Accuracy theo Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.savefig(acc_plot)
    plt.close()

    # Vẽ biểu đồ Loss theo Epoch
    plt.figure(figsize=(8, 5))
    plt.plot(df['Epoch'], df['Loss'], label='Train Loss')
    plt.plot(df['Epoch'], df['Val Loss'], label='Val Loss')
    plt.title('Loss theo Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.savefig(loss_plot)
    plt.close()

# Ghi log và biểu đồ
save_training_log(history_finetune)

# ====== LƯU MÔ HÌNH SAU FINE-TUNE ======
model_path = os.path.join(OUTPUT_FOLDER, f"{MODEL_NAME}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.keras")
model.save(model_path)                                           # Lưu mô hình .keras đã fine-tune

