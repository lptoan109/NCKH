# ==== THƯ VIỆN ====
import tensorflow as tf
from tensorflow.keras import layers, models                     # Dùng để xây mô hình Keras
from tensorflow.keras.applications import EfficientNetB0       # Import EfficientNetB0 đã huấn luyện sẵn
import matplotlib.pyplot as plt                                 # Vẽ biểu đồ
import pandas as pd                                             # Ghi log kết quả dạng bảng
import os                                                       # Làm việc với file/thư mục
from datetime import datetime                                   # Để đặt timestamp file

# ==== CẤU HÌNH ====
IMG_SIZE = (128, 128)                                           # Kích thước ảnh đầu vào
BATCH_SIZE = 32                                                 # Số lượng ảnh mỗi batch khi huấn luyện
EPOCHS = 30                                                     # Số vòng lặp train ban đầu
DATASET_DIR = r"D:\DuLieuHo\dataset"                            # Đường dẫn đến dataset đã chia sẵn train/val/test
MODEL_NAME = "efficientnet_cough_model"                         # Tên mô hình
OUTPUT_FOLDER = r"G:\My Drive\Tài liệu NCKH\AIData"             # Thư mục lưu kết quả, log, model

# ==== TẢI DỮ LIỆU ====
# Load dữ liệu từ thư mục train
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    os.path.join(DATASET_DIR, "train"),
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

# Load dữ liệu từ thư mục val
val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    os.path.join(DATASET_DIR, "val"),
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

# Load dữ liệu từ thư mục test
test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    os.path.join(DATASET_DIR, "test"),
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

# Lấy danh sách tên lớp (class) và số lượng lớp
class_names = train_ds.class_names
num_classes = len(class_names)
print("📂 Các lớp:", class_names)

# ==== TỐI ƯU DỮ LIỆU ====
AUTOTUNE = tf.data.AUTOTUNE
# Sử dụng tiền xử lý song song để tăng tốc train
train_ds = train_ds.prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)
test_ds = test_ds.prefetch(AUTOTUNE)

# ==== KHỞI TẠO MÔ HÌNH EfficientNetB0 ====
# Tải EfficientNetB0 pretrained từ ImageNet, không bao gồm layer đầu ra cuối (include_top=False)
base_model = EfficientNetB0(include_top=False, weights='imagenet', input_shape=(128, 128, 3))
base_model.trainable = False   # Freeze toàn bộ mô hình ban đầu (chưa fine-tune)

# Xây dựng mô hình mới trên nền EfficientNet
model = models.Sequential([
    layers.Rescaling(1./255),                      # Chuẩn hóa ảnh từ [0–255] về [0–1]
    base_model,                                    # Thêm EfficientNet làm feature extractor
    layers.GlobalAveragePooling2D(),               # Giảm chiều feature map
    layers.Dropout(0.2),                           # Tránh overfitting bằng dropout
    layers.Dense(num_classes, activation='softmax')  # Layer phân loại đầu ra
])

# Biên dịch mô hình lần đầu
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',   # Do label là dạng số
              metrics=['accuracy'])                     # Đánh giá theo độ chính xác

# ==== HUẤN LUYỆN LẦN 1 ====
history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS)  # Train mô hình

# ==== FINE-TUNE ====
base_model.trainable = True  # Mở khóa mô hình EfficientNet
# Chỉ fine-tune tầng cuối (20 layer cuối cùng), còn lại giữ frozen
for layer in base_model.layers[:-20]:
    layer.trainable = False

# Biên dịch lại mô hình sau khi fine-tune
model.compile(optimizer=tf.keras.optimizers.Adam(1e-5),   # Learning rate thấp hơn
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train lại mô hình thêm 10 epoch (fine-tune)
history_fine = model.fit(train_ds, validation_data=val_ds, epochs=10)

# ==== GHI LOG KẾT QUẢ ====
def save_training_log(history, history_fine=None, output_folder=OUTPUT_FOLDER, prefix=MODEL_NAME):
    # Gộp lịch sử train & fine-tune
    history_data = history.history
    if history_fine:
        for key in history_fine.history:
            history_data[key].extend(history_fine.history[key])

    # Tạo DataFrame chứa log
    df = pd.DataFrame({
        'Epoch': list(range(1, len(history_data['accuracy']) + 1)),
        'Accuracy': history_data['accuracy'],
        'Loss': history_data['loss'],
        'Val Accuracy': history_data['val_accuracy'],
        'Val Loss': history_data['val_loss']
    })

    os.makedirs(output_folder, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")  # Tạo timestamp hiện tại
    # Tạo tên file log
    excel_path = os.path.join(output_folder, f"{prefix}_trainlog_{timestamp}.xlsx")
    acc_plot = os.path.join(output_folder, f"{prefix}_accuracy_{timestamp}.png")
    loss_plot = os.path.join(output_folder, f"{prefix}_loss_{timestamp}.png")

    df.to_excel(excel_path, index=False)  # Ghi file Excel
    print("✅ Đã lưu file Excel log:", excel_path)

    # Vẽ biểu đồ Accuracy
    plt.figure(figsize=(8,5))
    plt.plot(df['Epoch'], df['Accuracy'], label='Train Acc')
    plt.plot(df['Epoch'], df['Val Accuracy'], label='Val Acc')
    plt.title('Accuracy theo Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(acc_plot)   # Lưu ảnh
    plt.close()
    print("✅ Đã lưu biểu đồ Accuracy:", acc_plot)

    # Vẽ biểu đồ Loss
    plt.figure(figsize=(8,5))
    plt.plot(df['Epoch'], df['Loss'], label='Train Loss')
    plt.plot(df['Epoch'], df['Val Loss'], label='Val Loss')
    plt.title('Loss theo Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(loss_plot)
    plt.close()
    print("✅ Đã lưu biểu đồ Loss:", loss_plot)

# Gọi hàm lưu log và biểu đồ
save_training_log(history, history_fine)

# ==== LƯU MÔ HÌNH ====
MODEL_FILE = os.path.join(OUTPUT_FOLDER, f"{MODEL_NAME}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.keras")
model.save(MODEL_FILE)
print("✅ Đã lưu mô hình:", MODEL_FILE)
