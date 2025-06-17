from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Khởi tạo bộ tiền xử lý ảnh: chuẩn hóa và tách tập
datagen = ImageDataGenerator(
    rescale=1./255,          # Chia giá trị pixel từ [0-255] → [0-1]
    validation_split=0.2     # Giữ lại 20% ảnh để làm validation
)

# 👉 Bộ ảnh huấn luyện
train_generator = datagen.flow_from_directory(
    r'D:\Dữ liệu về tiếng ho NCKH\COUGHVID Dataset\coughvid_dataset',               # Thư mục gốc chứa các thư mục con
    target_size=(100, 40),   # Resize ảnh về đúng kích thước input của CNN
    color_mode='grayscale',  # Vì ảnh spectrogram là ảnh xám
    class_mode='categorical',# Trả về nhãn dạng one-hot (phù hợp với softmax)
    batch_size=32,           # Mỗi lần huấn luyện 32 ảnh
    shuffle=True,            # Trộn ngẫu nhiên ảnh mỗi epoch
    subset='training'        # Lấy tập huấn luyện
)

# 👉 Bộ ảnh kiểm tra (validation)
val_generator = datagen.flow_from_directory(
    r'D:\Dữ liệu về tiếng ho NCKH\COUGHVID Dataset\coughvid_validationset',
    target_size=(100, 40),
    color_mode='grayscale',
    class_mode='categorical',
    batch_size=32,
    shuffle=True,
    subset='validation'      # Lấy tập kiểm tra
)
print("Train classes:", train_generator.class_indices)
print("Number of training samples:", train_generator.samples)
print("Number of validation samples:", val_generator.samples)
