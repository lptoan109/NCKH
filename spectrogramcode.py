<<<<<<< Updated upstream
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
=======
# ====== CÀI ĐẶT THƯ VIỆN ======
!pip install librosa matplotlib pillow tqdm

# ====== THƯ VIỆN ======
import os                                       # Thao tác thư mục
import librosa                                  # Xử lý âm thanh
import librosa.display                          # Hiển thị spectrogram
import matplotlib.pyplot as plt                 # Vẽ biểu đồ
from PIL import Image                          # Xử lý ảnh
import numpy as np                             # Thư viện toán học
from tqdm import tqdm                          # Hiển thị progress bar

# ====== CẤU HÌNH ======
INPUT_WAV_DIR = '/content/drive/MyDrive/DATASET_WAV'        # Thư mục chứa các file .wav gốc (phân lớp theo folder con)
OUTPUT_IMG_DIR = '/content/drive/MyDrive/DATASET_SPECTRO'   # Thư mục lưu ảnh spectrogram
IMG_SIZE = (224, 224)                                       # Kích thước chuẩn hóa (EfficientNetB3 khuyến nghị 224x224)

os.makedirs(OUTPUT_IMG_DIR, exist_ok=True)                  # Tạo thư mục nếu chưa có

# ====== HÀM CHUYỂN ĐỔI WAV → SPECTROGRAM ẢNH ======
def save_spectrogram(wav_path, output_path, img_size=(224, 224)):
    y, sr = librosa.load(wav_path, sr=None)                 # Đọc file .wav, sampling rate gốc
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)  # Tạo Mel Spectrogram
    S_dB = librosa.power_to_db(S, ref=np.max)               # Chuyển sang dB

    # Vẽ và lưu ảnh spectrogram tạm thời
    fig = plt.figure(figsize=(3, 3), dpi=100)
    plt.axis('off')
    librosa.display.specshow(S_dB, sr=sr, cmap='magma')
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.close(fig)

    # Mở ảnh, convert sang RGB, resize chính xác, rồi lưu đè
    img = Image.open(output_path).convert("RGB")
    img = img.resize(img_size)
    img.save(output_path)

# ====== XỬ LÝ TOÀN BỘ FILE .WAV GIỮ NGUYÊN CẤU TRÚC THƯ MỤC ======
for root, _, files in os.walk(INPUT_WAV_DIR):
    for file in tqdm(files, desc=f"Processing {root}"):
        if file.lower().endswith('.wav'):
            wav_path = os.path.join(root, file)

            # Tạo đường dẫn output tương ứng, giữ nguyên phân lớp thư mục
            rel_path = os.path.relpath(root, INPUT_WAV_DIR)            # VD: 'class_A/'
            output_subdir = os.path.join(OUTPUT_IMG_DIR, rel_path)
            os.makedirs(output_subdir, exist_ok=True)

            output_img_path = os.path.join(output_subdir, file.replace('.wav', '.png'))
            save_spectrogram(wav_path, output_img_path, IMG_SIZE)

print("✅ Đã hoàn thành chuyển .wav → spectrogram theo cấu trúc phân lớp.")
>>>>>>> Stashed changes
