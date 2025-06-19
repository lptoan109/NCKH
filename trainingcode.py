import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Ẩn warning
import warnings
warnings.filterwarnings("ignore")         # Ẩn warning


from tensorflow.keras.mixed_precision import set_global_policy
set_global_policy('mixed_float16')
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, GlobalAveragePooling2D
from pr_procescode import val_generator, train_generator
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import pandas as pd
from datetime import datetime
from collections import Counter
import numpy as np

#Tiền xử lý các ảnh
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(100, 40, 1)),#Cắt ảnh ra 32 filter và lấy đặc trưng
    BatchNormalization(),
    MaxPooling2D((2, 2)),#Lọc và giảm kích thước ảnh xuống 1 nửa

    Conv2D(64, (3, 3), activation='relu',kernel_regularizer=l2(0.001)), #Cắt ảnh ra 64 filter
    BatchNormalization(),
    MaxPooling2D((2, 2)),

    Conv2D(128, (3, 3), activation='relu',kernel_regularizer=l2(0.001)), #Cắt ảnh ra 128 filter
    BatchNormalization(),
    MaxPooling2D((2, 2)),

    Conv2D(256, (3, 3), activation='relu',kernel_regularizer=l2(0.001)),   # Cắt ảnh ra 256 filter 
    BatchNormalization(),

    GlobalAveragePooling2D(),#Chuyển xuống dạng 1D
    Dense(256, activation='relu'),#Tạo ra 256 neuron để học 
    Dropout(0.3),#Tự động tắt 30% neuron để thử thách
    Dense(5, activation='softmax', dtype='float32')  #Phân 5 lớp bệnh thành dạng xác xuất sao cho tổng bằng 1 (mixed precision yêu cầu float32 đầu ra)
])

# Đếm số mẫu mỗi lớp từ train_generator
counter = Counter(train_generator.classes)
total = sum(counter.values())
num_classes = len(counter)

# Tính class_weight
class_weight = {cls: total / (num_classes * count) for cls, count in counter.items()}
print("Class weights:", class_weight)

#Khởi tạo học
model.compile(
    optimizer='adam', #Tối ưu hóa mô hình bằng cách cập nhật đặc trọng số sau mỗi lần học
    loss='categorical_crossentropy', #Lấy thông số sai lệch sau mỗi lần học
    metrics=['accuracy'] #Lấy chỉ số độ chính xác
)

# Đường dẫn thư mục lưu
folder = r"D:\ModelAI"
os.makedirs(folder, exist_ok=True)

# Tên file lưu
filepath = os.path.join(folder, "best_model.keras")

# Callback lưu mô hình
checkpoint = ModelCheckpoint(
    filepath=filepath,
    monitor='val_accuracy',
    save_best_only=True,
    verbose=1
)

print("Mô hình sẽ được lưu vào:", filepath)

# Callback dừng sớm nếu val_accuracy không cải thiện
earlystop = EarlyStopping(
    monitor='val_accuracy',
    patience=5,
    restore_best_weights=True,
    verbose=1
)

#Lấy dữ liệu và in kết quả
history = model.fit(
    train_generator,              # Lấy dữ liệu huấn luyện
    validation_data=val_generator, # Lấy dữ liệu kiểm tra (validation)
    epochs=30,                    # Số vòng lặp (lần quét toàn bộ dữ liệu)
    verbose=1,                     # Hiện tiến trình huấn luyện
    class_weight=class_weight,      #Truyền dữ liệu class_weight
    callbacks=[checkpoint, earlystop]
)

# Tạo DataFrame từ history
history_data = history.history
num_epochs = len(history_data['accuracy'])

df = pd.DataFrame({
    'Epoch': list(range(1, num_epochs + 1)),
    'Accuracy': history_data['accuracy'],
    'Loss': history_data['loss'],
    'Val Accuracy': history_data['val_accuracy'],
    'Val Loss': history_data['val_loss']
})

# Đặt thư mục lưu tự động
folder = r"G:\My Drive\Tài liệu NCKH\AIData"
os.makedirs(folder, exist_ok=True)  # Tạo thư mục nếu chưa tồn tại

# Tạo tên file với timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
excel_filename = f"cnn_trainlog_{timestamp}.xlsx"
accuracy_plot_filename = f"accuracy_plot_{timestamp}.png"
loss_plot_filename = f"loss_plot_{timestamp}.png"

# Lưu Excel
excel_path = os.path.join(folder, excel_filename)
df.to_excel(excel_path, index=False)
print(f"Đã lưu file Excel tại: {excel_path}")

# Vẽ và lưu biểu đồ Accuracy
plt.figure(figsize=(10, 5))
plt.plot(df['Epoch'], df['Accuracy'], label='Training Accuracy', marker='o')
plt.plot(df['Epoch'], df['Val Accuracy'], label='Validation Accuracy', marker='o')
plt.title('Accuracy theo Epoch')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.tight_layout()
acc_plot_path = os.path.join(folder, accuracy_plot_filename)
plt.savefig(acc_plot_path)
plt.close()
print(f"Đã lưu biểu đồ Accuracy tại: {acc_plot_path}")

# Vẽ và lưu biểu đồ Loss
plt.figure(figsize=(10, 5))
plt.plot(df['Epoch'], df['Loss'], label='Training Loss', marker='o')
plt.plot(df['Epoch'], df['Val Loss'], label='Validation Loss', marker='o')
plt.title('Loss theo Epoch')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.tight_layout()
loss_plot_path = os.path.join(folder, loss_plot_filename)
plt.savefig(loss_plot_path)
plt.close()
print(f"Đã lưu biểu đồ Loss tại: {loss_plot_path}")
