import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

# Thư mục chứa các file .wav
input_dir = r'e:\NCKH\Dữ liệu ICBHI 2017 đã lọc\COPD(viêm phổi mãn tính)\Dữ liệu WAV'         # Thay bằng đường dẫn tới thư mục chứa .wav
output_dir = r'e:\NCKH\Output\COPD(viêm phổi mãn tính)'  # Thư mục để lưu ảnh spectrogram

# Lặp qua tất cả các file trong thư mục
for filename in os.listdir(input_dir):
    if filename.endswith('.wav'):
        audio_path = os.path.join(input_dir, filename)
        print(f'Đang xử lý: {filename}')

        # Đọc âm thanh
        y, sr = librosa.load(audio_path, sr=None)

        # Tính STFT và chuyển sang decibel
        D = librosa.stft(y)
        S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)

        # Vẽ và lưu spectrogram
        plt.figure(figsize=(10, 4))
        librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='log', cmap='magma')
        plt.colorbar(format='%+2.0f dB')
        plt.title(f'Spectrogram: {filename}')
        plt.tight_layout()

        # Tên file đầu ra
        output_filename = os.path.splitext(filename)[0] + '.png'
        output_path = os.path.join(output_dir, output_filename)
        plt.savefig(output_path)
        plt.close()  # Giải phóng bộ nhớ

print("Hoàn tất xử lý tất cả các file.")
