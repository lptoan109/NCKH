import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB0
import matplotlib.pyplot as plt
import pandas as pd
import os
from datetime import datetime

# ====== 🔧 CẤU HÌNH ======
IMG_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = 30
DATASET_DIR = r"D:\DuLieuHo\dataset"  # 📁 Đường dẫn tới dataset đã chia train/val/test
MODEL_NAME = "efficientnet_cough_model"
OUTPUT_FOLDER = r"G:\My Drive\Tài liệu NCKH\AIData"

# ====== 📂 TẢI DỮ LIỆU ======
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    os.path.join(DATASET_DIR, "train"),
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)
val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    os.path.join(DATASET_DIR, "val"),
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)
test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    os.path.join(DATASET_DIR, "test"),
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

class_names = train_ds.class_names
num_classes = len(class_names)
print("📂 Các lớp:", class_names)

# ====== ⚙️ Tiền xử lý
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)
test_ds = test_ds.prefetch(AUTOTUNE)

# ====== 🧠 MÔ HÌNH EfficientNetB0
base_model = EfficientNetB0(include_top=False, weights='imagenet', input_shape=(128, 128, 3))
base_model.trainable = False  # Freeze ban đầu

model = models.Sequential([
    layers.Rescaling(1./255),
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.2),
    layers.Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# ====== 🚀 TRAIN BAN ĐẦU
history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS)

# ====== 🔁 FINE-TUNE TẦNG CUỐI
base_model.trainable = True
for layer in base_model.layers[:-20]:
    layer.trainable = False

model.compile(optimizer=tf.keras.optimizers.Adam(1e-5),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

history_fine = model.fit(train_ds, validation_data=val_ds, epochs=10)

# ====== 📈 GHI LOG KẾT QUẢ
def save_training_log(history, history_fine=None, output_folder=OUTPUT_FOLDER, prefix=MODEL_NAME):
    history_data = history.history
    if history_fine:
        for key in history_fine.history:
            history_data[key].extend(history_fine.history[key])

    df = pd.DataFrame({
        'Epoch': list(range(1, len(history_data['accuracy']) + 1)),
        'Accuracy': history_data['accuracy'],
        'Loss': history_data['loss'],
        'Val Accuracy': history_data['val_accuracy'],
        'Val Loss': history_data['val_loss']
    })

    os.makedirs(output_folder, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    excel_path = os.path.join(output_folder, f"{prefix}_trainlog_{timestamp}.xlsx")
    acc_plot = os.path.join(output_folder, f"{prefix}_accuracy_{timestamp}.png")
    loss_plot = os.path.join(output_folder, f"{prefix}_loss_{timestamp}.png")

    df.to_excel(excel_path, index=False)
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
    plt.savefig(acc_plot)
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

# Ghi log
save_training_log(history, history_fine)

# ====== 💾 LƯU MÔ HÌNH
MODEL_FILE = os.path.join(OUTPUT_FOLDER, f"{MODEL_NAME}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.h5")
model.save(MODEL_FILE)
print("✅ Đã lưu mô hình:", MODEL_FILE)
