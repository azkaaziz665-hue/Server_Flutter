"""
Server Relay & YOLO Detector untuk ESP32-CAM (Beda Jaringan)
Dibuat dengan FastAPI & Ultralytics YOLO

Fitur:
1. Endpoint POST /upload: Menerima frame gambar dari ESP32-CAM di luar jaringan
2. Endpoint GET /stream: Menyajikan MJPEG stream real-time untuk aplikasi Flutter
3. Endpoint GET /latest: Menyajikan frame terakhir dan hasil deteksi JSON YOLO
"""

import io
import os
import time
import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, Response
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO

app = FastAPI(title="Coffee Ripeness YOLO Stream Server")

# Izinkan CORS agar aplikasi Flutter dapat mengakses tanpa hambatan
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model YOLO (Gunakan model custom biji kopi Anda atau bawaan YOLOv8n)
try:
    model = YOLO("yolov8n.pt")  # Ganti dengan 'best_coffee.pt' jika sudah punya
    print("[INFO] Model YOLO berhasil dimuat!")
except Exception as e:
    model = None
    print(f"[WARN] Gagal memuat model YOLO: {e}. Menggunakan fallback.")

# Variabel buffer frame global
latest_frame_bytes = None
latest_detections = []
latest_timestamp = 0

@app.post("/upload")
async def upload_frame(file: UploadFile = File(...)):
    """
    Dihubungi oleh ESP32-CAM untuk mengunggah frame kamera
    """
    global latest_frame_bytes, latest_detections, latest_timestamp
    
    contents = await file.read()
    latest_timestamp = time.time()
    
    # Proses deteksi YOLO jika model aktif
    if model is not None:
        try:
            nparr = np.frombuffer(contents, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if img is not None:
                results = model.predict(img, conf=0.4, verbose=False)
                detections = []
                h, w, _ = img.shape
                
                for r in results:
                    for box in r.boxes:
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        conf = float(box.conf[0])
                        cls_id = int(box.cls[0])
                        cls_name = model.names[cls_id]
                        
                        # Normalisasi koordinat 0.0 - 1.0 untuk Flutter
                        detections.append({
                            "label": cls_name,
                            "confidence": conf,
                            "box": [x1 / w, y1 / h, x2 / w, y2 / h]
                        })
                
                latest_detections = detections
        except Exception as err:
            print(f"[ERROR] Inference error: {err}")
    
    latest_frame_bytes = contents
    return {"status": "ok", "timestamp": latest_timestamp}

def generate_mjpeg_stream():
    """Generator multipart stream MJPEG untuk Aplikasi Flutter"""
    global latest_frame_bytes
    while True:
        if latest_frame_bytes is not None:
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + latest_frame_bytes + b"\r\n"
            )
        time.sleep(0.04)  # ~25 FPS

@app.get("/stream")
def get_stream():
    """
    Endpoint MJPEG yang dimasukkan ke Pengaturan Aplikasi Flutter
    """
    return StreamingResponse(
        generate_mjpeg_stream(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

@app.get("/detections")
def get_detections():
    """Mengambil metadata bounding box & status kematangan terkini"""
    return {
        "timestamp": latest_timestamp,
        "count": len(latest_detections),
        "detections": latest_detections
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app:app", host="0.0.0.0", port=port)
