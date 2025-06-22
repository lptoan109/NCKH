import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

# 📁 Thư mục chứa file .wav chuẩn hóa
input_dir = r'D:\Dữ liệu về tiếng ho NCKH\COUGHVID Dataset\coughvid_dataset\covidWAV'
# 📁 Thư mục lưu ảnh spectrogram
output_dir = r'D:\Dữ liệu về tiếng ho NCKH\COUGHVID Dataset\coughvid_dataset\covidSPECTROGRAM'
os.makedirs(output_dir, exist_ok=True)

# 🎯 Kích thước ảnh đầu ra (pixels)
IMG_SIZE = 128

# 🔁 Duyệt qua từng file .wav
for filename in os.listdir(input_dir):
    if filename.endswith('.wav'):
        audio_path = os.path.join(input_dir, filename)
        output_filename = os.path.splitext(filename)[0] + ".png"
        output_path = os.path.join(output_dir, output_filename)

        print(f"🔧 Đang xử lý: {filename}")
        try:
            # 1. Load âm thanh
            y, sr = librosa.load(audio_path, sr=None)

            # (Tùy chọn) Bỏ qua file âm thanh quá ngắn
            if len(y) < sr * 0.5:
                print(f"⚠️ File quá ngắn, bỏ qua: {filename}")
                continue

            # 2. Tính STFT rồi chuyển thành dB
            D = librosa.stft(y, n_fft=1024, hop_length=256)
            S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)

            # 3. Vẽ ảnh không có trục, không padding
            fig = plt.figure(figsize=(IMG_SIZE / 100, IMG_SIZE / 100), dpi=100)
            plt.axis('off')
            librosa.display.specshow(S_db, sr=sr, cmap='magma')
            plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

            # 4. Lưu ảnh
            fig.savefig(output_path, bbox_inches='tight', pad_inches=0)
            plt.close(fig)
            print(f"✅ Đã lưu: {output_filename}")

        except Exception as e:
            print(f"❌ Lỗi khi xử lý {filename}: {e}")

print("\n🎉 Hoàn tất tạo ảnh spectrogram 128x128 từ toàn bộ file .wav")
