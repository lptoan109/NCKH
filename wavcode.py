import os
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

# 📁 Thư mục chứa file .webm
input_dir = r'D:\Dữ liệu về tiếng ho NCKH\COUGHVID Dataset\WEBMdata'
# 📁 Thư mục để lưu các file .wav đã chuẩn hóa
output_dir = r'D:\Dữ liệu về tiếng ho NCKH\COUGHVID Dataset\WAVdata'
os.makedirs(output_dir, exist_ok=True)

# ✅ Hàm xử lý 1 file, dùng cho đa tiến trình
def convert_webm_to_wav(filename):
    input_path = os.path.join(input_dir, filename)
    output_filename = os.path.splitext(filename)[0] + ".wav"
    output_path = os.path.join(output_dir, output_filename)

    command = [
        "ffmpeg", "-y", "-i", input_path,
        "-ar", "22050", "-ac", "1", "-c:a", "pcm_s16le",
        output_path
    ]

    try:
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return filename, True  # ✅ Thành công
    except subprocess.CalledProcessError:
        return filename, False  # ❌ Lỗi

if __name__ == "__main__":
    file_list = [f for f in os.listdir(input_dir) if f.endswith('.webm')]
    errors = []

    with ProcessPoolExecutor() as executor:
        futures = {executor.submit(convert_webm_to_wav, filename): filename for filename in file_list}

        for future in tqdm(as_completed(futures), total=len(futures), desc="Chuyển đổi"):
            filename, success = future.result()
            if not success:
                errors.append(filename)

    if errors:
        with open("error_convert_webm_to_wav.txt", "w", encoding="utf-8") as f:
            for err_file in errors:
                f.write(f"{err_file}\n")

    print("\n🎉 Đã chuyển xong toàn bộ .webm sang .wav chuẩn hóa!")
    if errors:
        print(f"❗ Có {len(errors)} file lỗi → xem file 'error_convert_webm_to_wav.txt'")
