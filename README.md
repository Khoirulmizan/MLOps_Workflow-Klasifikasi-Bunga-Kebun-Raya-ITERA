Anggota Kelompok :
1. Ferdy Kevin Naibaho (122450107)
2. Ibrahim Al-kahfi (122450100)
3. Khoirul Mizan Abdullah (122450010)
4. Haikal Dwi Syaputra (122450067)

# MLOps Workflow — Klasifikasi Bunga Kebun Raya ITERA

Repositori ini merupakan **Tugas Besar Mata Kuliah Machine Learning Operations (MLOps)** yang bertujuan untuk menerapkan seluruh tahapan dalam proses pengembangan model Machine Learning dalam format pipeline yang berulang, terdokumentasi, dan terotomasi. Proyek ini fokus pada **klasifikasi bunga** menggunakan dataset bunga yang telah dikumpulkan dan disiapkan, serta menerapkan praktik MLOps modern seperti tracking dengan MLflow, containerization (Docker), dan deployment API.

##  Fitur Utama

* **Deep Learning Models**: Menggunakan arsitektur CNN kustom dan MobileNet (Nano) untuk efisiensi.
* **Blue-Green Deployment**: Mekanisme *serving* ganda yang memungkinkan peralihan instan antara model stabil ("Blue") dan model eksperimental ("Green") via API.
* **MLflow Integration**: Pelacakan metrik pelatihan (accuracy, loss) dan parameter model secara otomatis.
* **Data Collection Pipeline**: Sistem otomatis yang menyimpan gambar unggahan pengguna berdasarkan tanggal untuk keperluan *retraining*.
* **FastAPI Backend**: API performa tinggi untuk melayani prediksi model.
* **Dockerized**: Seluruh aplikasi dibungkus dalam Docker Container untuk kemudahan deployment (Cloud/On-Premise).

##  Struktur Proyek

```text
project_root/
project_root/
├── .dvc/                  # Folder konfigurasi DVC (Data Versioning)
├── collected_data/        # Folder penampung data dari user
├── dataset/               # Folder Dataset Bunga asli
├── models/                # Tempat menyimpan file .keras (Blue & Green)
├── train_with_mlflow.py   # Script training baru
├── api.py                 # Backend FastAPI (Serving & Logic)
├── informasi_bunga.py     # Data informasi (tetap dipakai)
├── Dockerfile             # Konfigurasi Image Docker
├── docker-compose.yml     # Untuk menjalankan API & MLflow lokal
└── requirements.txt       # Dependencies
---

## 1. Dataset Bunga

Dataset ini berisi data bunga yang menjadi dasar klasifikasi. Dataset ini bisa berupa **fitur numerik** (misalnya panjang/lebar sepal & petal seperti pada dataset bunga Iris klasik) atau **gambar bunga** yang dikategorikan menurut kelasnya. :contentReference[oaicite:0]{index=0}

Beberapa informasi umum yang biasanya ada dalam dataset semacam ini:
- Fitur morfologi bunga seperti panjang dan lebar sepal serta petal (angka).
- Label kelas/species bunga sebagai target prediksi.
- Data bisa berupa CSV atau direktori gambar per kelas.

---

##  2. Data Collection & Preprocessing

###  Penggabungan Data
Skrip `merge_data.py` digunakan untuk:
- Menggabungkan dataset mentah yang tersebar di folder `collected_data/`
- Membersihkan missing values
- Menyimpan hasil merge sebagai dataset siap training.

```python
# contoh sederhana merge_data.py
import pandas as pd

df1 = pd.read_csv("collected_data/data1.csv")
df2 = pd.read_csv("collected_data/data2.csv")
df = pd.concat([df1, df2])
df.to_csv("Dataset Bunga/flowers_dataset.csv", index=False)

```

## 3. Training & Eksperimen Model (train_with_mlflow.py)

Skrip ini adalah komponen utama dalam workflow MLOps. Fungsinya:

- Menyiapkan dataset
- Melakukan pembagian data train/test
- Melatih model Machine Learning
- Melacak eksperimen dengan MLflow (metrics, parameters, model artifacts)

## 4.  Evaluasi Model
Setelah training, evaluasi dilakukan untuk:

- Menilai akurasi, precision, recall, F1-score
- Menentukan apakah model sudah layak dipakai produksi

Hasil evaluasi biasanya dilacak di UI MLflow untuk membandingkan performa berbagai eksperimen.

## 5.  Dokumentasi API & Workflow
Sistem backend menggunakan FastAPI. Berikut adalah endpoint utamanya:

1. Prediksi Bunga (POST /predict)
Mengunggah gambar untuk diklasifikasikan oleh model yang sedang aktif.

- Input: File gambar (multipart/form-data).
- Output: JSON berisi kelas prediksi, confidence score, dan info botani.
- Side Effect: Gambar disimpan ke collected_data/YYYY-MM-DD/ untuk data versioning.

2. Switch Model (Blue-Green) (POST /admin/switch-model)
Endpoint admin untuk mengganti model yang digunakan secara live.

- Parameter: color ("blue" atau "green").
- Contoh: http://localhost:8000/admin/switch-model?color=green
- Kegunaan: Jika model "Green" (baru) berkinerja buruk, admin dapat melakukan rollback ke "Blue" dalam hitungan detik.
  
## 6.  Containerization (Docker)
Dockerfile dan docker-compose.yml digunakan untuk:

- Menjalankan environment Python isolasi
- Menyusun service API
- Mempermudah deployment ke layanan hosting seperti Render/App platform

## 7.  Monitoring dengan MLflow
Kami menggunakan MLflow untuk mencatat eksperimen. Setiap kali train_with_mlflow.py dijalankan, sistem mencatat:

- Parameters: Epochs, Batch Size, Learning Rate.
- Metrics: Accuracy, Loss, Validation Accuracy.
- Artifacts: Model file, Confusion Matrix, dan Class Names.

## 8. Deployment (Cloud)
Proyek ini siap di-deploy ke platform container-based seperti Render, AWS ECS, atau GCP Cloud Run.

Contoh Render (render.yaml): Repository ini menyertakan render.yaml untuk deployment instan sebagai Web Service di Render. Cukup hubungkan repository GitHub Anda ke Render, dan layanan akan otomatis berjalan.


# Langkah - Langkah Menjalankan Kode Proyek Ini

## Langkah 1: Persiapan Proyek (Setup)
1. Siapkan Folder Proyek Buat folder baru di komputer, lalu susun file-file yang sudah kita bahas sebelumnya ( api.py, train_with_mlflow.py, docker-compose.yml, dll) sesuai struktur folder ini:

my_flower_project/
├── dataset/                <-- Isi dengan folder-folder bunga 
├── models/                 <-- Buat folder kosong ini dulu
├── collected_data/         <-- Buat folder kosong ini dulu
├── api.py
├── train_with_mlflow.py
├── merge_data.py
├── informasi_bunga.py
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── .dockerignore

2. Buka Terminal Buka terminal (Command Prompt/PowerShell/Terminal) di dalam folder my_flower_project.

## Langkah 2: Pelatihan Model Pertama (Training)
Sebelum menjalankan API, kita butuh file model (.keras) dan file class_names.json. Kita akan melatihnya di lokal (tanpa Docker dulu agar lebih cepat memanfaatkan resource laptop langsung).
1. Buat Virtual Environment (Opsional tapi disarankan):
'''
python -m venv env
# Windows:
.\env\Scripts\activate
# Mac/Linux:
source env/bin/activate
'''
2. Install Library:
'''
pip install -r requirements.txt
'''
4. Jalankan Training: Kita akan melatih model awal. Pastikan folder Dataset Bunga sudah ada isinya.

python train_with_mlflow.py

Tunggu hingga proses training selesai.

4. Cek Hasil: Setelah selesai, periksa folder models/. Harusnya sekarang muncul file baru, misalnya flower_classifier_new.keras (atau nama sesuai script).

- Ubah nama file tersebut menjadi flower_classifier_model.keras (ini akan jadi model Blue).
- Copy file tersebut dan beri nama flower_classifier_nano.keras (ini akan jadi dummy model Green).
- Pastikan file class_names.json juga sudah muncul di root folder.

## Langkah 3: Menjalankan Aplikasi dengan Docker
Sekarang kita akan menyalakan sistem "Mesin Produksi" kita menggunakan Docker Compose.

1. Build & Run: Di terminal, jalankan perintah:

docker-compose up --build

Proses ini akan memakan waktu cukup lama di awal karena harus mendownload image Python dan menginstall library.

2. Verifikasi: Jika berhasil, Anda akan melihat log berjalan terus menerus. Jangan tutup terminal ini.

- Buka browser dan akses: http://localhost:8000/docs
- Jika muncul tampilan Swagger UI (halaman dokumentasi API yang interaktif), berarti server sudah jalan!
- Buka tab baru: http://localhost:5000
- Jika muncul dashboard MLflow, berarti server tracking juga jalan.

## Langkah 4: Simulasi Penggunaan (Testing API)
Mari kita coba apakah API berfungsi dengan benar.

1. Di halaman http://localhost:8000/docs:
2. Cari endpoint berwarna hijau bertuliskan POST /predict.
3. Klik tombol Try it out (sebelah kanan).
4. Klik Choose File, pilih sembarang gambar bunga dari komputer Anda.
5. Klik tombol biru besar Execute.
6. Lihat Hasil: Scroll ke bawah sedikit. Di bagian "Server response", Anda akan melihat JSON seperti ini:

{
  "model_used": "blue",
  "prediction": {
    "class": "Mawar",
    "confidence": "95.20%",
    ...
  },
  "data_collected_at": "collected_data/2023-12-15/user_upload_..."
}

7. Cek Data Collection: Buka File Explorer di laptop, masuk ke folder collected_data. Kita akan melihat folder tanggal hari ini, dan di dalamnya ada gambar yang barusan Anda upload. Sistem Data Versioning berhasil!

## Langkah 5: Simulasi Blue-Green Deployment
Bayangkan Anda ingin mengganti model tanpa mematikan server.

1. Buka tab baru di browser atau gunakan fitur "Try it out" di Swagger UI pada endpoint POST /admin/switch-model.
2. Isi parameter color dengan green.
3. Klik Execute.
4. Responnya harus: {"status": "success", "active_model_now": "green"}.
5. Coba lakukan prediksi lagi di langkah 4. Perhatikan JSON response-nya, field "model_used" sekarang harusnya bernilai "green".

## Langkah 6: Simulasi Retraining (Siklus MLOps)
Ini adalah langkah terakhir untuk mensimulasikan "pembelajaran berkelanjutan".

1. Validasi Data: Buka terminal baru (terminal Docker biarkan jalan). Jalankan:

python merge_data.py

Script ini akan memindahkan gambar dari collected_data ke folder sementara di Dataset Bunga.

2. Sortir Manual: Buka folder Dataset Bunga/_UNLABELED_SORT_ME. Pindahkan gambar-gambar tersebut ke folder bunga yang benar (misal pindahkan foto mawar ke folder dataset/Mawar).

3. Training Ulang: Jalankan lagi training:

python train_with_mlflow.py

Sekarang model dilatih dengan data yang lebih banyak.

4. Cek MLflow: Buka http://localhost:5000. Klik pada experiment list. Anda akan melihat dua "Run" (satu yang lama, satu yang barusan). Anda bisa klik keduanya dan pilih "Compare" untuk melihat grafik kenaikan akurasinya.


## Cara Mematikan Server
Jika sudah selesai, kembali ke terminal tempat Docker berjalan, lalu tekan: Ctrl + C Atau ketik perintah:

docker-compose down
