from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from ultralytics import YOLO  # <-- Import YOLO
import numpy as np
import cv2
import base64
import io
from PIL import Image

app = FastAPI()

# Menambahkan CORS (SANGAT PENTING)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Izinkan semua
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === 1. LOAD MODEL YOLO ANDA ===
# GANTI PATH INI jika 'best.pt' Anda ada di tempat lain
model_path = "best.pt"
try:
    model = YOLO(model_path)
    print(f"Model {model_path} berhasil dimuat.")
except Exception as e:
    print(f"GAGAL memuat model: {e}")
    model = None

# Model untuk menerima data gambar Base64
class ImageRequest(BaseModel):
    image: str

@app.get("/")
def home():
    return {"message": "Server YOLOv8 SIBI Aktif!"}

@app.post("/predict")
async def predict(item: ImageRequest):
    if model is None:
        return {"error": "Model tidak berhasil dimuat"}

    try:
        # === 2. Decode gambar Base64 dari JavaScript ===
        image_data = item.image.split(',')[1]
        image_bytes = base64.b64decode(image_data)
        
        # Konversi bytes ke gambar PIL, lalu ke array OpenCV (BGR)
        pil_image = Image.open(io.BytesIO(image_bytes))
        img = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

        # === 3. Jalankan Prediksi YOLO ===
        results = model.predict(img)

        prediction_text = "-"
        
        # Cek apakah ada deteksi
        if results and results[0].boxes:
            # Ambil deteksi dengan confidence tertinggi (YOLO otomatis mengurutkannya)
            best_box = results[0].boxes[0]
            
            # Ambil nama kelas (huruf) dan confidence
            cls_id = int(best_box.cls[0])
            conf = float(best_box.conf[0])
            cls_name = results[0].names[cls_id]

            # Hanya tampilkan jika confidence > 50% (0.5)
            if conf > 0.5:
                prediction_text = cls_name

        # === 4. Kirim Balasan (HANYA HURUFNYA) ===
        return {"prediction": prediction_text}

    except Exception as e:
        print(f"Error saat prediksi: {e}")
        return {"error": str(e)}
