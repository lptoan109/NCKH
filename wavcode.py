import os
import subprocess

# ✅ Đường dẫn đến thư mục chứa các file .webm
input_folder = r"c:\NCKH\Dữ liệu Coughvid đã lọc\upper_infection"
# ✅ Đường dẫn đến thư mục bạn muốn lưu file .wav đầu ra
output_folder = r"c:\NCKH\Dữ liệu Coughvid đã lọc\upper_infection\Dữ liệu WAV"

# Duyệt tất cả file trong thư mục đầu vào
for filename in os.listdir(input_folder):
    if filename.lower().endswith(".webm"):
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename.replace(".webm", ".wav"))
        
        # ✅ Lệnh ffmpeg chuẩn
        command = [
            r'c:\NCKH\ffmpeg\bin\ffmpeg.exe',
            "-i", input_path,
            "-vn",                    # Bỏ video
            "-acodec", "pcm_s16le",  # Định dạng âm thanh chuẩn AI
            "-ar", "16000",          # Sample rate: 16kHz
            "-ac", "1",              # Mono
            output_path
        ]

        print(f"🎧 Chuyển: {filename} → {os.path.basename(output_path)}")
        subprocess.run(command)
