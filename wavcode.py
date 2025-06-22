import os
import subprocess

# 📁 Thư mục chứa file .webm gốc
input_dir = r'D:\Dữ liệu về tiếng ho NCKH\COUGHVID Dataset\coughvid_dataset\covid'
# 📁 Thư mục để lưu các file .wav đã chuẩn hóa
output_dir = r'D:\Dữ liệu về tiếng ho NCKH\COUGHVID Dataset\coughvid_dataset\covidWAV'
os.makedirs(output_dir, exist_ok=True)

# 🔁 Duyệt từng file .webm
for filename in os.listdir(input_dir):
    if filename.endswith('.webm'):
        input_path = os.path.join(input_dir, filename)
        output_filename = os.path.splitext(filename)[0] + ".wav"
        output_path = os.path.join(output_dir, output_filename)

        print(f"🔧 Đang xử lý: {filename}")

        # 📦 Lệnh ffmpeg chuyển và chuẩn hóa
        command = [
            "ffmpeg",
            "-y",                     # Ghi đè nếu file đã tồn tại
            "-i", input_path,         # File đầu vào
            "-ar", "22050",           # Sample rate: 22050 Hz
            "-ac", "1",               # Mono
            "-c:a", "pcm_s16le",      # PCM 16-bit
            output_path               # File đầu ra (.wav)
        ]

        try:
            subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            print(f"✅ Xong: {output_filename}")
        except subprocess.CalledProcessError:
            print(f"❌ Lỗi ffmpeg với file: {filename}")

print("\n🎉 Đã chuyển xong toàn bộ .webm sang .wav chuẩn hóa!")
