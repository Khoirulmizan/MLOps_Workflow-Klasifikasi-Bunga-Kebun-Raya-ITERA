# File: api.py
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.responses import JSONResponse
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os
import json
import shutil
from datetime import datetime
from informasi_bunga import DATA_BUNGA

app = FastAPI(title="ITERA Flower Classification API (Blue-Green)")

# --- Konfigurasi Global ---
MODELS = {}
CLASS_NAMES = []
# Default Active Model
ACTIVE_MODEL_KEY = "blue" 
MODEL_PATHS = {
    "blue": "models/flower_classifier_model.keras",  # Model Lama/Stabil
    "green": "models/flower_classifier_nano.keras"   # Model Baru/Eksperimental
}
COLLECTED_DATA_DIR = "collected_data"

# --- Startup Event (Load Models) ---
@app.on_event("startup")
async def startup_event():
    global MODELS, CLASS_NAMES
    
    # Load Class Names
    try:
        with open('class_names.json', 'r') as f:
            CLASS_NAMES = json.load(f)
    except FileNotFoundError:
        print("WARNING: class_names.json tidak ditemukan. Jalankan training dulu.")
        CLASS_NAMES = ["Unknown"] * 22

    # Load Kedua Model (Blue & Green)
    for color, path in MODEL_PATHS.items():
        if os.path.exists(path):
            print(f"Loading {color} model from {path}...")
            MODELS[color] = tf.keras.models.load_model(path)
        else:
            print(f"WARNING: Model {color} tidak ditemukan di {path}")
            MODELS[color] = None

# --- Helper Functions ---
def preprocess_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes))
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize((224, 224))
    img_array = np.asarray(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array

def save_user_image(image_bytes):
    """Menyimpan gambar user ke folder berdasarkan tanggal"""
    today = datetime.now().strftime("%Y-%m-%d")
    target_folder = os.path.join(COLLECTED_DATA_DIR, today)
    
    # Buat folder jika belum ada (Data Versioning Preparation)
    os.makedirs(target_folder, exist_ok=True)
    
    # Generate nama file unik (timestamp)
    timestamp = datetime.now().strftime("%H-%M-%S-%f")
    filename = f"user_upload_{timestamp}.jpg"
    file_path = os.path.join(target_folder, filename)
    
    with open(file_path, "wb") as f:
        f.write(image_bytes)
    
    return file_path

# --- Endpoints ---

@app.get("/")
def home():
    return {
        "message": "Flower Classification API Ready",
        "active_model": ACTIVE_MODEL_KEY,
        "available_models": list(MODELS.keys())
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Endpoint utama untuk prediksi bunga"""
    global ACTIVE_MODEL_KEY
    
    if MODELS[ACTIVE_MODEL_KEY] is None:
        raise HTTPException(status_code=503, detail="Active model is not loaded")

    # Baca file gambar
    image_bytes = await file.read()
    
    # 1. Simpan Data (Data Collection Pipeline)
    saved_path = save_user_image(image_bytes)
    
    # 2. Preprocess
    processed_img = preprocess_image(image_bytes)
    
    # 3. Prediksi menggunakan Model Aktif
    model = MODELS[ACTIVE_MODEL_KEY]
    predictions = model.predict(processed_img)[0]
    
    # Ambil Top 1
    top_idx = np.argmax(predictions)
    top_prob = float(predictions[top_idx])
    predicted_class = CLASS_NAMES[top_idx]
    
    # Ambil Info Detail
    info = DATA_BUNGA.get(predicted_class, {"nama_umum": "Unknown", "deskripsi": "Tidak ada info"})
    
    return {
        "model_used": ACTIVE_MODEL_KEY,
        "prediction": {
            "class": predicted_class,
            "confidence": f"{top_prob*100:.2f}%",
            "common_name": info['nama_umum'],
            "description": info['deskripsi'],
            "latin_name": info.get('nama_latin', '-')
        },
        "data_collected_at": saved_path
    }

@app.post("/admin/switch-model")
def switch_model(color: str = Query(..., regex="^(blue|green)$")):
    """Endpoint Admin untuk mengubah model aktif (Blue/Green)"""
    global ACTIVE_MODEL_KEY
    
    if MODELS.get(color) is None:
        raise HTTPException(status_code=400, detail=f"Model {color} tidak tersedia/gagal dimuat.")
    
    ACTIVE_MODEL_KEY = color
    return {"status": "success", "active_model_now": ACTIVE_MODEL_KEY}