# ====== CÀI ĐẶT THƯ VIỆN ======
!pip install tensorflow pandas matplotlib

# ====== THƯ VIỆN ======
import tensorflow as tf                                # Thư viện chính để xây dựng và huấn luyện mô hình deep learning
from tensorflow.keras import layers, models            # Để tạo các layer và mô hình Sequential
from tensorflow.keras.applications import EfficientNetB0  # Import EfficientNetB0 pretrained
import matplotlib.pyplot as plt                        # Thư viện để vẽ biểu đồ
import pandas as pd                                    # Thư viện xử lý dữ liệu dạng bảng
import os                                              # Làm việc với hệ thống file
from datetime import datetime                          # Để tạo tên file có timestamp

# ====== KẾT NỐI GOOGLE DRIVE ======
from google.colab import drive
drive.mount('/content/drive')

# ====== CẤU HÌNH ======
IMG_SIZE = (128, 128)                                  # Kích thước ảnh đầu vào của EfficientNet
BATCH_SIZE = 32                                        # Số lượng ảnh trong mỗi batch
EPOCHS = 15                                            # Số epoch để fine-tune
COMBINED_TRAIN_DIR = '/content/drive/MyDrive/NCKH/combined_train'  # Thư mục chứa ảnh train (gốc + pseudo)
VAL_DIR = '/content/drive/MyDrive/NCKH/dataset/val'                # Thư mục chứa tập validation
MODEL_INPUT = '/content/drive/MyDrive/NCKH/AIData/efficientnet_cough_model_20250620_1830.keras'  # Mô hình đã train trước đó
OUTPUT_FOLDER = '/content/drive/MyDrive/NCKH/AIData'                # Thư mục để lưu mô hình mới, log, biểu đồ
MODEL_NAME = "efficientnet_cough_finetune"                          # Tên mô hình lưu
TENSORBOARD_LOG_DIR = '/content/drive/MyDrive/NCKH/tensorboard_logs'  # Thư mục log TensorBoard

# ====== TẢI DỮ LIỆU ======
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    COMBINED_TRAIN_DIR,                     # Thư mục dữ liệu huấn luyện
    image_size=IMG_SIZE,                    # Resize về đúng kích thước cho EfficientNet
    batch_size=BATCH_SIZE                   # Kích thước batch
)
val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    VAL_DIR,                                # Thư mục dữ liệu validation
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

# Lấy tên class
class_names = train_ds.class_names
num_classes = len(class_names)

# ====== PREFETCH GIÚP TĂNG HIỆU NĂNG ======
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)

# ====== LOAD MÔ HÌNH ĐÃ HUẤN LUYỆN TRƯỚC ======
model = tf.keras.models.load_model(MODEL_INPUT)     # Load mô hình .keras đã huấn luyện trước đó

# ====== MỞ KHÓA EfficientNetB0 ĐỂ FINE-TUNE ======
base_model = model.layers[1]                        # EfficientNetB0 là layer thứ 2 trong mô hình Sequential
base_model.trainable = True                         # Cho phép fine-tune lại backbone

# Đóng băng 70% tầng đầu tiên để tránh overfitting
FINE_TUNE_AT = int(len(base_model.layers) * 0.7)
for layer in base_model.layers[:FINE_TUNE_AT]:
    layer.trainable = False

# ====== CALLBACK TENSORBOARD ======
# Tạo thư mục lưu log TensorBoard
log_dir = os.path.join(TENSORBOARD_LOG_DIR, datetime.now().strftime("%Y%m%d-%H%M%S"))
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

# ====== COMPILE LẠI MÔ HÌNH ======
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-5),       # Learning rate thấp để fine-tune nhẹ nhàng
    loss='sparse_categorical_crossentropy',         # Dùng cho nhãn dạng số nguyên
    metrics=['accuracy']                            # Đánh giá bằng độ chính xác
)

# ====== FINE-TUNE VỚI TENSORBOARD ======
history_finetune = model.fit(
    train_ds,                                       # Tập huấn luyện (gốc + pseudo)
    validation_data=val_ds,                         # Tập kiểm tra
    epochs=EPOCHS,                                  # Số epoch fine-tune
    callbacks=[tensorboard_callback]                # Ghi log vào TensorBoard
)

# ====== MỞ TENSORBOARD TRÊN COLAB ======
%load_ext tensorboard
%tensorboard --logdir $TENSORBOARD_LOG_DIR

# ====== LƯU LOG RA EXCEL + VẼ BIỂU ĐỒ ======
def save_training_log(history, output_folder=OUTPUT_FOLDER, prefix=MODEL_NAME):
    hist = history.history
    df = pd.DataFrame({
        'Epoch': list(range(1, len(hist['accuracy']) + 1)),
        'Accuracy': hist['accuracy'],
        'Loss': hist['loss'],
        'Val Accuracy': hist['val_accuracy'],
        'Val Loss': hist['val_loss']
    })

    os.makedirs(output_folder, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    excel_path = os.path.join(output_folder, f"{prefix}_finetune_log_{timestamp}.xlsx")
    acc_plot = os.path.join(output_folder, f"{prefix}_finetune_acc_{timestamp}.png")
    loss_plot = os.path.join(output_folder, f"{prefix}_finetune_loss_{timestamp}.png")

    df.to_excel(excel_path, index=False)

    # Biểu đồ Accuracy
    plt.figure(figsize=(8, 5))
    plt.plot(df['Epoch'], df['Accuracy'], label='Train Acc')
    plt.plot(df['Epoch'], df['Val Accuracy'], label='Val Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy theo Epoch')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(acc_plot)
    plt.close()

    # Biểu đồ Loss
    plt.figure(figsize=(8, 5))
    plt.plot(df['Epoch'], df['Loss'], label='Train Loss')
    plt.plot(df['Epoch'], df['Val Loss'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss theo Epoch')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(loss_plot)
    plt.close()

save_training_log(history_finetune)

# ====== LƯU MÔ HÌNH SAU FINE-TUNE ======
model_path = os.path.join(OUTPUT_FOLDER, f"{MODEL_NAME}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.keras")
model.save(model_path)
