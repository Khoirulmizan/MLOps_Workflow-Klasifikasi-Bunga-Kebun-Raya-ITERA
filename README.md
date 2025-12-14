# MLOps Workflow — Klasifikasi Bunga Kebun Raya ITERA

Repositori ini merupakan **Tugas Besar Mata Kuliah Machine Learning Operations (MLOps)** yang bertujuan untuk menerapkan seluruh tahapan dalam proses pengembangan model Machine Learning dalam format pipeline yang berulang, terdokumentasi, dan terotomasi. Proyek ini fokus pada **klasifikasi bunga** menggunakan dataset bunga yang telah dikumpulkan dan disiapkan, serta menerapkan praktik MLOps modern seperti tracking dengan MLflow, containerization (Docker), dan deployment API.


---

## 📊 1. 📦 Dataset Bunga

Dataset ini berisi data bunga yang menjadi dasar klasifikasi. Dataset ini bisa berupa **fitur numerik** (misalnya panjang/lebar sepal & petal seperti pada dataset bunga Iris klasik) atau **gambar bunga** yang dikategorikan menurut kelasnya. :contentReference[oaicite:0]{index=0}

Beberapa informasi umum yang biasanya ada dalam dataset semacam ini:
- Fitur morfologi bunga seperti panjang dan lebar sepal serta petal (angka).
- Label kelas/species bunga sebagai target prediksi.
- Data bisa berupa CSV atau direktori gambar per kelas. :contentReference[oaicite:1]{index=1}

---

## 🧹 2. 🔄 Data Collection & Preprocessing

### ✨ Penggabungan Data
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
<<<<<<< HEAD
```
=======
>>>>>>> 46b2bbfe1a45c88562d3bc33eba556cca2f5da03


## 🤖 3. 📈 Training & Eksperimen Model (train_with_mlflow.py)

Skrip ini adalah komponen utama dalam workflow MLOps. Fungsinya:

- Menyiapkan dataset
- Melakukan pembagian data train/test
- Melatih model Machine Learning
- Melacak eksperimen dengan MLflow (metrics, parameters, model artifacts)

## 4. 📊 Evaluasi Model
Setelah training, evaluasi dilakukan untuk:

- Menilai akurasi, precision, recall, F1-score
- Menentukan apakah model sudah layak dipakai produksi

Hasil evaluasi biasanya dilacak di UI MLflow untuk membandingkan performa berbagai eksperimen.

## 5. 📡 API Deployment (api.py)
File api.py berisi server API untuk melakukan inference model yang telah dilatih. Biasanya dibuat dengan FastAPI atau Flask.

## 6. 🧱 Containerization (Docker)
Dockerfile dan docker-compose.yml digunakan untuk:

- Menjalankan environment Python isolasi
- Menyusun service API
- Mempermudah deployment ke layanan hosting seperti Render/App platform

## 7. 📦 Deployment
Setelah container siap :

- Deploy di Render, Heroku, AWS ECS/Fargate
- Menggunakan render.yaml sebagai konfigurasi deployment otomatis
