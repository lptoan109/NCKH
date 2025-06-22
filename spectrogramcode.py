import os
import subprocess
import concurrent.futures
from tqdm import tqdm

input_dir = r'D:\Dữ liệu về tiếng ho NCKH\COUGHVID Dataset\WAVdata'
output_dir = r'D:\Dữ liệu về tiếng ho NCKH\COUGHVID Dataset\SPECTROGRAMdata'
os.makedirs(output_dir, exist_ok=True)

IMG_SIZE = 128
TIMEOUT_SECONDS = 10

def process_file(filename):
    try:
        # Gọi file xử lý riêng (sử dụng subprocess)
        command = [
            "python", "single_proces.py",  # 📌 Bạn cần tạo file này bên ngoài (viết bên dưới)
            filename, input_dir, output_dir, str(IMG_SIZE)
        ]
        subprocess.run(command, timeout=TIMEOUT_SECONDS, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return filename, True, None
    except subprocess.TimeoutExpired:
        return filename, False, "Timeout"
    except subprocess.CalledProcessError:
        return filename, False, "Subprocess error"

if __name__ == "__main__":
    files = [f for f in os.listdir(input_dir) if f.endswith('.wav')]
    errors = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(process_file, filename): filename for filename in files}

        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Chuyển đổi"):
            filename = futures[future]
            result = future.result()
            filename, success, error = result
            if not success:
                errors.append((filename, error))

    if errors:
        with open("error_spectrogram_log.txt", "w", encoding="utf-8") as f:
            for filename, err in errors:
                f.write(f"{filename}: {err}\n")

    print("\n🎉 Hoàn tất!")
