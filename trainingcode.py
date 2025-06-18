import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Ẩn warning
import warnings
warnings.filterwarnings("ignore")         # Ẩn warning

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from pr_procescode import val_generator, train_generator
import os


#Tiền xử lý các ảnh
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(100, 40, 1)), #Cắt ảnh ra 32 filter và lấy đặc trưng
    MaxPooling2D(pool_size=(2, 2)), #Lọc và giảm kích thước ảnh xuống 1 nửa
    Conv2D(64, (3, 3), activation='relu'), #Cắt ảnh ra 64 filter và lấy đặc trưng 
    MaxPooling2D(pool_size=(2, 2)), #Lọc và giảm kích thước ảnh xuống 1 nửa
    Flatten(), #Chuyển xuống dạng 1D
    Dense(128, activation='relu'), #Tạo ra 128 neuron để học 
    Dense(5, activation='softmax')  #Phân 5 lớp bệnh thành dạng xác xuất sao cho tổng bằng 1
])

model.compile(
    optimizer='adam', #Tối ưu hóa mô hình bằng cách cập nhật đặc trọng số sau mỗi lần học
    loss='categorical_crossentropy', #Lấy thông số sai lệch sau mỗi lần học
    metrics=['accuracy'] #Lấy chỉ số độ chính xác
)

history = model.fit(
    train_generator,              # Lấy dữ liệu huấn luyện
    validation_data=val_generator, # Lấy dữ liệu kiểm tra (validation)
    epochs=20,                    # Số vòng lặp (lần quét toàn bộ dữ liệu)
    verbose=1                     # Hiện tiến trình huấn luyện
)
#model.save("cough_classifier_model.h5")

