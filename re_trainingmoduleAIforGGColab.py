# ====== CÀI ĐẶT THƯ VIỆN ======
!pip install tensorflow pandas matplotlib

# ====== KẾT NỐI GOOGLE DRIVE ======
from google.colab import drive
drive.mount('/content/drive')

# ====== THƯ VIỆN ======
import os                                # Thư viện thao tác file, thư mục
import tensorflow as tf                  # Thư viện chính để xây dựng deep learning model
from tensorflow.keras import layers, models  # Để xây dựng các lớp CNN, Transformer
import matplotlib.pyplot as plt          # Thư viện vẽ biểu đồ
import pandas as pd                      # Thư viện xử lý dữ liệu dạng bảng
from datetime import datetime            # Thư viện tạo timestamp cho tên file
from sklearn.utils.class_weight import compute_class_weight  # Tính toán class weight để cân bằng dữ liệu

# ====== CẤU HÌNH ======
IMG_SIZE = (224, 224)                                           # Kích thước ảnh input
BATCH_SIZE = 32                                                 # Số ảnh mỗi batch
EPOCHS = 30                                                     # Số epoch huấn luyện
DATASET_DIR = '/content/drive/MyDrive/Tài liệu NCKH/coughvid_dataset'  # Dataset ảnh spectrogram
RESULT_DIR = '/content/drive/MyDrive/NCKH/RESULT'               # Thư mục lưu model, log, biểu đồ
MODEL_NAME = "cnn_transformer_cough"                            # Tên model
TENSORBOARD_LOG_DIR = '/content/drive/MyDrive/NCKH/tensorboard_logs'  # Thư mục log TensorBoard

# ====== TẢI DATASET ======
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    os.path.join(DATASET_DIR, "train"), image_size=IMG_SIZE, batch_size=BATCH_SIZE
)
val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    os.path.join(DATASET_DIR, "val"), image_size=IMG_SIZE, batch_size=BATCH_SIZE
)
test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    os.path.join(DATASET_DIR, "test"), image_size=IMG_SIZE, batch_size=BATCH_SIZE
)

class_names = train_ds.class_names
num_classes = len(class_names)

# ====== PREFETCH ĐỂ TĂNG HIỆU NĂNG ======
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)
test_ds = test_ds.prefetch(AUTOTUNE)

# ====== TÍNH CLASS WEIGHTS ĐỂ CÂN BẰNG ======
labels = []
for _, y in train_ds.unbatch():
    labels.append(int(y.numpy()))

class_weights = compute_class_weight("balanced", classes=np.arange(num_classes), y=labels)
class_weight_dict = {i: w for i, w in enumerate(class_weights)}
print("Class Weights:", class_weight_dict)

# ====== XÂY DỰNG MÔ HÌNH CNN + TRANSFORMER ======
def build_cnn_transformer_model(input_shape=(224, 224, 3), num_classes=3, num_heads=4, transformer_units=[128, 64]):
    inputs = layers.Input(shape=input_shape)

    # Khối CNN feature extractor
    x = layers.Rescaling(1./255)(inputs)            # Normalize ảnh về [0,1]
    x = layers.Conv2D(32, 3, padding="same", activation="relu")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(128, 3, padding="same", activation="relu")(x)
    x = layers.MaxPooling2D()(x)

    # Reshape về (batch_size, sequence_length, feature_dim) cho Transformer
    x = layers.Reshape((-1, x.shape[-1]))(x)

    # Khối Transformer
    attention_output = layers.MultiHeadAttention(num_heads=num_heads, key_dim=x.shape[-1])(x, x)
    x = layers.Add()([x, attention_output])         # Residual connection
    x = layers.LayerNormalization(epsilon=1e-6)(x)

    for dim in transformer_units:                   # Feed-Forward Network (FFN)
        x_ffn = layers.Dense(dim, activation="relu")(x)
        x = layers.Add()([x, x_ffn])                # Residual connection
        x = layers.LayerNormalization(epsilon=1e-6)(x)

    x = layers.GlobalAveragePooling1D()(x)          # Pooling toàn bộ sequence
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs, outputs)
    return model

model = build_cnn_transformer_model(input_shape=(224, 224, 3), num_classes=num_classes)
model.summary()  # Hiển thị cấu trúc model

# ====== COMPILE MÔ HÌNH ======
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-4),       # LR thấp vì có Transformer
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# ====== CALLBACKS ======
log_dir = os.path.join(TENSORBOARD_LOG_DIR, datetime.now().strftime("%Y%m%d-%H%M%S"))
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
earlystop_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# ====== HUẤN LUYỆN ======
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=[tensorboard_callback, earlystop_callback],
    class_weight=class_weight_dict
)

# ====== VẼ BIỂU ĐỒ ACCURACY & LOSS ======
def save_training_log(history, output_folder=RESULT_DIR, prefix=MODEL_NAME):
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
    excel_path = os.path.join(output_folder, f"{prefix}_trainlog_{timestamp}.xlsx")
    acc_plot = os.path.join(output_folder, f"{prefix}_accuracy_{timestamp}.png")
    loss_plot = os.path.join(output_folder, f"{prefix}_loss_{timestamp}.png")

    df.to_excel(excel_path, index=False)

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

save_training_log(history)

# ====== LƯU MÔ HÌNH ======
MODEL_FILE = os.path.join(RESULT_DIR, f"{MODEL_NAME}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.keras")
model.save(MODEL_FILE)
print("✅ Đã lưu mô hình:", MODEL_FILE)

# ====== CHẠY TENSORBOARD ======
%load_ext tensorboard
%tensorboard --logdir $TENSORBOARD_LOG_DIR
