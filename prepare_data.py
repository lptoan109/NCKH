import os                     # Xử lý thư mục và đường dẫn
import shutil                 # Dùng để sao chép file
from tqdm import tqdm         # Tiện ích để hiển thị progress bar khi lặp

# === 📁 CẤU HÌNH ĐƯỜNG DẪN ===
original_train_dir = r"D:\DuLieuHo\dataset\train"        # ✅ Thư mục chứa dữ liệu huấn luyện gốc (đã gán nhãn tay)
pseudo_label_dir = r"D:\DuLieuHo\pseudo_label"           # ✅ Thư mục chứa ảnh được gán nhãn giả (bởi AI)
combined_train_dir = r"D:\DuLieuHo\combined_train"       # ✅ Thư mục đầu ra – nơi bạn sẽ có ảnh gốc + ảnh gán nhãn

# === 🔁 LẶP QUA TỪNG CLASS (VD: 'covid', 'healthy', ...)
for cls in os.listdir(original_train_dir):
    orig_cls_path = os.path.join(original_train_dir, cls)         # Đường dẫn tới class gốc
    pseudo_cls_path = os.path.join(pseudo_label_dir, cls)         # Đường dẫn tới class pseudo (có thể không tồn tại)
    combined_cls_path = os.path.join(combined_train_dir, cls)     # Thư mục đầu ra chứa cả ảnh gốc và giả

    # === 📂 TẠO FOLDER ĐÍCH (nếu chưa có)
    if not os.path.exists(combined_cls_path):
        os.makedirs(combined_cls_path)

    # === ✅ SAO CHÉP ẢNH GỐC
    for fname in tqdm(os.listdir(orig_cls_path), desc=f"📂 Gốc/{cls}"):
        src = os.path.join(orig_cls_path, fname)                                # Đường dẫn ảnh gốc
        dst = os.path.join(combined_cls_path, "orig_" + fname)                  # Đổi tên ảnh (tiền tố 'orig_') để phân biệt
        shutil.copy(src, dst)                                                   # Copy vào thư mục đích

    # === ✅ SAO CHÉP ẢNH GÁN NHÃN GIẢ (NẾU CÓ)
    if os.path.exists(pseudo_cls_path):                                         # Kiểm tra nếu class đó có ảnh gán nhãn giả
        for fname in tqdm(os.listdir(pseudo_cls_path), desc=f"🤖 Pseudo/{cls}"):
            src = os.path.join(pseudo_cls_path, fname)                          # Đường dẫn ảnh giả
            dst = os.path.join(combined_cls_path, "pseudo_" + fname)           # Đổi tên ảnh (tiền tố 'pseudo_')
            shutil.copy(src, dst)                                               # Copy vào thư mục đích

# === ✅ THÔNG BÁO HOÀN TẤT
print("✅ Hoàn tất trộn dữ liệu gốc + pseudo-label.")
