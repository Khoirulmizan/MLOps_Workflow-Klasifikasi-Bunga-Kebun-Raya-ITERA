# File: api.py
import os
import io
import json
import shutil
from datetime import datetime
from typing import Optional

# Library FastAPI
from fastapi import FastAPI, File, UploadFile, HTTPException, Query, Request
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

# Library Data Science
import tensorflow as tf
import numpy as np
from PIL import Image

# Import Data Bunga
try:
    from informasi_bunga import DATA_BUNGA
except ImportError:
    DATA_BUNGA = {}

# --- Konfigurasi Awal ---
app = FastAPI(
    title="API Klasifikasi Bunga",
    description="API Klasifikasi Bunga dengan fitur Hot-Swapping Model dan Data Collection.",
    version="2.0"
)

# Mount folder static agar script.js bisa diakses
app.mount("/static", StaticFiles(directory="static"), name="static")

# Setup folder templates untuk file HTML
templates = Jinja2Templates(directory="templates")

# Direktori & Path
MODELS_DIR = "models"
COLLECTED_DATA_DIR = "collected_data"
CLASS_NAMES_PATH = "class_names.json"

# Konfigurasi Model Blue-Green
# Blue: Model Lama/Stabil
# Green: Model Baru/Eksperimental (Nano)
MODEL_PATHS = {
    "blue": os.path.join(MODELS_DIR, "flower_classifier_model.keras"),
    "green": os.path.join(MODELS_DIR, "flower_classifier_nano.keras")
}

# State Global (Menyimpan model di memori RAM)
LOADED_MODELS = {}
ACTIVE_MODEL_KEY = "blue"  # Default model yang digunakan
CLASS_NAMES = []

# --- Event saat Server Menyala (Startup) ---
@app.on_event("startup")
async def load_resources():
    global LOADED_MODELS, CLASS_NAMES
    
    print("--- [SYSTEM] Memulai Server API ---")

    # 1. Muat Nama Kelas
    if os.path.exists(CLASS_NAMES_PATH):
        with open(CLASS_NAMES_PATH, 'r') as f:
            CLASS_NAMES = json.load(f)
        print(f"--- [INFO] {len(CLASS_NAMES)} kelas bunga dimuat.")
    else:
        print("--- [WARNING] class_names.json tidak ditemukan! Prediksi akan error.")

    # 2. Muat Kedua Model ke RAM (Pre-loading)
    for color, path in MODEL_PATHS.items():
        if os.path.exists(path):
            print(f"--- [INFO] Memuat Model {color.upper()} dari {path}...")
            try:
                LOADED_MODELS[color] = tf.keras.models.load_model(path)
                print(f"--- [SUCCESS] Model {color.upper()} siap.")
            except Exception as e:
                print(f"--- [ERROR] Gagal memuat {color}: {e}")
                LOADED_MODELS[color] = None
        else:
            print(f"--- [WARNING] File model {color} tidak ditemukan di {path}.")
            LOADED_MODELS[color] = None

# --- Fungsi Helper (Pembantu) ---

def save_image_for_retraining(image_bytes):
    """
    Menyimpan gambar user ke folder terpisah berdasarkan tanggal.
    Contoh: collected_data/2025-12-15/img_12345.jpg
    """
    try:
        # Buat nama folder tanggal hari ini (YYYY-MM-DD)
        today_date = datetime.now().strftime("%Y-%m-%d")
        target_folder = os.path.join(COLLECTED_DATA_DIR, today_date)
        
        os.makedirs(target_folder, exist_ok=True)
        
        # Buat nama file unik
        filename = f"user_{datetime.now().strftime('%H%M%S_%f')}.jpg"
        file_path = os.path.join(target_folder, filename)
        
        # Simpan file
        with open(file_path, "wb") as f:
            f.write(image_bytes)
            
        return file_path
    except Exception as e:
        print(f"--- [ERROR] Gagal menyimpan data user: {e}")
        return None

def preprocess_image(image_bytes):
    """Mengubah bytes gambar menjadi array numpy siap prediksi"""
    img = Image.open(io.BytesIO(image_bytes))
    
    # Pastikan RGB
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    # Resize ke 224x224 (Sesuai training)
    img = img.resize((224, 224))
    
    img_array = np.asarray(img)
    img_array = np.expand_dims(img_array, axis=0) # Tambah dimensi batch
    img_array = img_array / 255.0  # Normalisasi
    return img_array

# --- Endpoints API ---

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {
        "request": request,
        "active_model": ACTIVE_MODEL_KEY
    })

@app.post("/predict")
async def predict_flower(file: UploadFile = File(...)):
    """
    Endpoint utama: Menerima gambar -> Prediksi dengan Model Aktif
    """
    global ACTIVE_MODEL_KEY
    
    # Cek apakah model aktif tersedia
    model = LOADED_MODELS.get(ACTIVE_MODEL_KEY)
    if model is None:
        raise HTTPException(status_code=503, detail=f"Model {ACTIVE_MODEL_KEY} is not available/loaded.")

    # 1. Baca File
    image_bytes = await file.read()
    
    # 2. Simpan Data (Data Versioning Pipeline)
    saved_path = save_image_for_retraining(image_bytes)
    
    # 3. Preprocessing
    try:
        processed_img = preprocess_image(image_bytes)
    except Exception as e:
        raise HTTPException(status_code=400, detail="Invalid image file.")

    # 4. Prediksi
    predictions = model.predict(processed_img)[0]
    
    # Ambil Top 1
    top_index = np.argmax(predictions)
    confidence = float(predictions[top_index])
    class_name = CLASS_NAMES[top_index] if top_index < len(CLASS_NAMES) else "Unknown"
    
    # Ambil Info Detail Bunga
    info = DATA_BUNGA.get(class_name, {})
    
    return {
        "model_used": ACTIVE_MODEL_KEY,
        "prediction": {
            "class_name": class_name,
            "confidence": f"{confidence * 100:.2f}%",
            "common_name": info.get("nama_umum", "Tidak diketahui"),
            "latin_name": info.get("nama_latin", "-"),
            "description": info.get("deskripsi", "Tidak ada deskripsi")
        },
        "data_storage": {
            "saved": True if saved_path else False,
            "path": saved_path
        }
    }

@app.post("/admin/switch-model")
def switch_model(color: str = Query(..., regex="^(blue|green)$")):
    """
    Fitur Blue-Green Deployment: Mengganti model aktif secara instan.
    Hanya menerima query parameter 'blue' atau 'green'.
    """
    global ACTIVE_MODEL_KEY
    
    if LOADED_MODELS.get(color) is None:
        raise HTTPException(status_code=400, detail=f"Model {color} belum dimuat atau file tidak ada.")
    
    ACTIVE_MODEL_KEY = color
    return {
        "message": f"Successfully switched to {color.upper()} model",
        "current_active_model": ACTIVE_MODEL_KEY
    }