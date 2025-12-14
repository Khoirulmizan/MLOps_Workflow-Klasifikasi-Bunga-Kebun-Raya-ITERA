# File: merge_data.py
import os
import shutil
import json

# Konfigurasi
SOURCE_ROOT = "collected_data"
DEST_ROOT = "Dataset Bunga"

# Muat daftar kelas untuk validasi
try:
    with open('class_names.json', 'r') as f:
        VALID_CLASSES = json.load(f)
except:
    print("Error: class_names.json tidak ditemukan.")
    exit()

def merge_datasets():
    print(f"--- Memulai Penggabungan Data dari {SOURCE_ROOT} ke {DEST_ROOT} ---")
    
    # Loop semua folder tanggal (misal: 2025-12-14)
    for date_folder in os.listdir(SOURCE_ROOT):
        date_path = os.path.join(SOURCE_ROOT, date_folder)
        
        if not os.path.isdir(date_path):
            continue
            
        print(f"\nMemproses tanggal: {date_folder}")
        
        # Loop semua gambar di dalam folder tanggal
        images = [f for f in os.listdir(date_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        if not images:
            print("  Folder kosong.")
            continue

        print(f"  Ditemukan {len(images)} gambar baru. Silakan review manual.")
        print("  Pindahkan gambar ke folder kelas yang sesuai di Dataset Bunga.")
        print("  Sistem ini tidak memindahkan otomatis karena butuh validasi manusia (apakah gambar benar bunga X?)")
        
        # --- LOGIKA BANTUAN ---
        # Karena kita tidak tahu gambar user itu bunga apa (label ground truthnya belum ada),
        # Script ini akan membuat folder "Unlabeled" di dalam dataset untuk Anda sortiri nanti.
        
        unlabeled_dir = os.path.join(DEST_ROOT, "_UNLABELED_SORT_ME")
        os.makedirs(unlabeled_dir, exist_ok=True)
        
        for img in images:
            src_file = os.path.join(date_path, img)
            # Rename file agar unik (tambah tanggal)
            new_name = f"{date_folder}_{img}"
            dst_file = os.path.join(unlabeled_dir, new_name)
            
            shutil.move(src_file, dst_file)
            print(f"  -> {img} dipindah ke _UNLABELED_SORT_ME")
            
        # Hapus folder tanggal jika sudah kosong
        if not os.listdir(date_path):
            os.rmdir(date_path)
            print(f"  Folder {date_folder} dihapus (sudah bersih).")

if __name__ == "__main__":
    choice = input("PENTING: Script ini akan memindahkan file dari collected_data ke folder sementara di Dataset Bunga. Lanjutkan? (y/n): ")
    if choice.lower() == 'y':
        merge_datasets()
        print("\nSelesai! Cek folder 'Dataset Bunga/_UNLABELED_SORT_ME' dan pindahkan gambar ke folder bunga yang benar secara manual.")