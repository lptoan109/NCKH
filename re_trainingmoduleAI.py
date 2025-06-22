import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB0
import matplotlib.pyplot as plt
import pandas as pd
import os
from datetime import datetime

# ====== 🔧 CẤU HÌNH ======
IMG_SIZE = (128, 128)                            # Kích thước ảnh đầu vào của mạng
BATCH_SIZE = 32                                  # Số lượng ảnh mỗi batch
EPOCHS = 15                                      # Số epoch cho fine-tuning
COMBINED_TRAIN_DIR = r"D:\DuLieuHo\combined_train"   # 📁 Thư mục chứa dữ liệu đã trộn (gốc + pseudo)
VAL_DIR = r"D:\DuLieuHo\dataset\val"                  # 📁 Tập validation ban đầu
MODEL_INPUT = r"G:\My Drive\Tài liệu NCKH\AIData\efficientnet_cough_model_20250620_1830.keras"  # ✅ Mô hình đã huấn luyện trước đó
OUTPUT_FOLDER = r"G:\My Drive\Tài liệu NCKH\AIData"      # 📁 Nơi lưu log, biểu đồ, mô hình mới
MODEL_NAME = "efficientnet_cough_finetune"               # 📛 Tên mô hình

# ====== 📂 TẢI DỮ LIỆU ======
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    COMBINED_TRAIN_DIR,                         # Load ảnh từ combined_train
    image_size=IMG_SIZE,                        # Resize ảnh
    batch_size=BATCH_SIZE                       # Batch size
)
val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    VAL_DIR,                                    # Load ảnh validation cũ
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

class_names = train_ds.class_names              # Lấy tên các class
num_classes = len(class_names)                  # Đếm số lớp
print("📂 Lớp:", class_names)

# ====== ⚙️ PREFETCH GIÚP TĂNG HIỆU NĂNG TRAINING ======
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)

# ====== 🧠 LOAD MÔ HÌNH ĐÃ HUẤN LUYỆN LẦN 1 ======
model = tf.keras.models.load_model(MODEL_INPUT)  # Load mô hình .keras đã huấn luyện trước
print("✅ Đã load mô hình:", MODEL_INPUT)

# ====== 🔓 MỞ KHÓA CÁC TẦNG ĐỂ FINE-TUNE ======
base_model = model.layers[1]                     # EfficientNetB0 là layer thứ 2 trong mô hình Sequential
base_model.trainable = True                      # Cho phép huấn luyện lại

# ❄️ Đóng băng khoảng 70% tầng đầu tiên để tránh overfitting
FINE_TUNE_AT = int(len(base_model.layers) * 0.7)
for layer in base_model.layers[:FINE_TUNE_AT]:
    layer.trainable = False

# ====== 🔁 COMPILE LẠI MÔ HÌNH ======
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-5),    # Dùng learning rate thấp để fine-tune nhẹ nhàng
    loss='sparse_categorical_crossentropy',      # Dùng cho nhãn dạng số (int)
    metrics=['accuracy']
)

# ====== 🚀 TIẾN HÀNH FINE-TUNE ======
history_finetune = model.fit(
    train_ds,                      # Dữ liệu huấn luyện (gốc + pseudo)
    validation_data=val_ds,        # Dữ liệu đánh giá
    epochs=EPOCHS                  # Số vòng lặp
)

# ====== 📈 GHI LOG & VẼ BIỂU ĐỒ ======
def save_training_log(history, output_folder=OUTPUT_FOLDER, prefix=MODEL_NAME):
    hist = history.history                                     # Lấy lịch sử training
    df = pd.DataFrame({                                        # Tạo DataFrame để xuất Excel
        'Epoch': list(range(1, len(hist['accuracy']) + 1)),
        'Accuracy': hist['accuracy'],
        'Loss': hist['loss'],
        'Val Accuracy': hist['val_accuracy'],
        'Val Loss': hist['val_loss']
    })

    os.makedirs(output_folder, exist_ok=True)                  # Tạo thư mục nếu chưa có
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")       # Gắn timestamp để phân biệt
    excel_path = os.path.join(output_folder, f"{prefix}_finetune_log_{timestamp}.xlsx")
    acc_plot = os.path.join(output_folder, f"{prefix}_finetune_acc_{timestamp}.png")
    loss_plot = os.path.join(output_folder, f"{prefix}_finetune_loss_{timestamp}.png")

    df.to_excel(excel_path, index=False)                       # Lưu log dưới dạng Excel
    print("✅ Đã lưu log:", excel_path)

    # 🎯 Accuracy Plot
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
    print("✅ Đã lưu biểu đồ Accuracy:", acc_plot)

    # 🎯 Loss Plot
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
    print("✅ Đã lưu biểu đồ Loss:", loss_plot)

# Ghi log ra file Excel + ảnh biểu đồ
save_training_log(history_finetune)

# ====== 💾 LƯU MÔ HÌNH FINE-TUNED ======
model_path = os.path.join(OUTPUT_FOLDER, f"{MODEL_NAME}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.keras")
model.save(model_path)                        # Lưu mô hình mới ở định dạng .keras
print("✅ Đã lưu mô hình fine-tuned:", model_path)
