# File: Dockerfile
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install dependencies sistem (untuk OpenCV jika dibutuhkan nnti)
RUN apt-get update && apt-get install -y libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*

# Copy requirements dan install
COPY requirements.txt .
# Tambahkan library untuk API server
RUN pip install --no-cache-dir -r requirements.txt fastapi uvicorn python-multipart mlflow dvc

# Copy seluruh kode
COPY . .

# Buat folder untuk data collection
RUN mkdir -p collected_data models

# Expose port (Render/Heroku biasanya meng-override ini via env var $PORT)
EXPOSE 8000

# Perintah default jalankan API
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]