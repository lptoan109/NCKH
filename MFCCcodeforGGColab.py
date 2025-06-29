# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Cài đặt thư viện
!pip install librosa numpy

# Import module
import os
import numpy as np
import librosa

# Hàm xử lý MFCC + delta + delta2
def extract_mfcc_features(file_path, 
                          sr=16000, 
                          n_mfcc=20, 
                          n_fft=512, 
                          hop_length=160, 
                          n_mels=40):
    y, sr = librosa.load(file_path, sr=sr)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, 
                                n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)
    mfcc = (mfcc - np.mean(mfcc)) / (np.std(mfcc)+1e-6)
    delta = (delta - np.mean(delta)) / (np.std(delta)+1e-6)
    delta2 = (delta2 - np.mean(delta2)) / (np.std(delta2)+1e-6)
    stacked = np.stack([mfcc, delta, delta2], axis=0)
    stacked = np.transpose(stacked, (2,1,0))
    return stacked

# Main process: quét toàn bộ folder
input_dir = "/content/drive/MyDrive/cough_dataset/data"
output_dir = "/content/drive/MyDrive/cough_dataset/output_mfcc"
os.makedirs(output_dir, exist_ok=True)

for label_name in os.listdir(input_dir):
    label_path = os.path.join(input_dir, label_name)
    if not os.path.isdir(label_path):
        continue

    out_label_path = os.path.join(output_dir, label_name)
    os.makedirs(out_label_path, exist_ok=True)

    for file_name in os.listdir(label_path):
        if not file_name.lower().endswith(".wav"):
            continue
        in_file_path = os.path.join(label_path, file_name)
        try:
            features = extract_mfcc_features(in_file_path)
            out_file_name = os.path.splitext(file_name)[0] + ".npy"
            out_file_path = os.path.join(out_label_path, out_file_name)
            np.save(out_file_path, features)
            print(f"Processed: {in_file_path} -> shape={features.shape}")
        except Exception as e:
            print(f"Error processing {in_file_path}: {e}")

print("Done. MFCC files saved to:", output_dir)
