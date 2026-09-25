"""CoffeeVision relay untuk ESP32-CAM, ESP32 servo, dan aplikasi Flutter.

Server ini tidak menjalankan inferensi AI dan tidak memuat best.onnx. ESP32-CAM
mengirim JPEG ke /upload, aplikasi membaca frame dari /stream, sedangkan model
best.tflite pada aplikasi Flutter melakukan deteksi secara lokal di ponsel.
"""

import os
import time
from typing import Optional

from fastapi import FastAPI, File, Header, HTTPException, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

app = FastAPI(title="CoffeeVision ESP32 Relay")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Frame paling baru dari ESP32-CAM. Penyimpanan in-memory ini cocok untuk satu
# instance Railway; gunakan Redis/object storage jika ingin multi-instance.
latest_frame_bytes: Optional[bytes] = None
latest_timestamp = 0.0

# Relay command untuk ESP32 di balik NAT. Perangkat melakukan polling untuk
# mengambil posisi servo terbaru tanpa membutuhkan IP publik.
servo_commands = {}
device_states = {}
DEVICE_TOKEN = os.environ.get("DEVICE_TOKEN", "")


class ServoCommand(BaseModel):
    pan: int = Field(ge=0, le=180)
    tilt: int = Field(ge=0, le=180)


def verify_device_token(token: Optional[str]) -> None:
    """Token opsional saat pengembangan; wajib bila DEVICE_TOKEN di Railway diisi."""
    if DEVICE_TOKEN and token != DEVICE_TOKEN:
        raise HTTPException(status_code=401, detail="Token perangkat tidak valid")


@app.post("/upload")
async def upload_frame(file: UploadFile = File(...)):
    """Menerima JPEG dari ESP32-CAM tanpa diproses oleh server."""
    global latest_frame_bytes, latest_timestamp

    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Frame kosong")

    latest_frame_bytes = contents
    latest_timestamp = time.time()
    return {"status": "ok", "timestamp": latest_timestamp, "bytes": len(contents)}


def generate_mjpeg_stream():
    """Meneruskan frame JPEG terbaru sebagai MJPEG untuk aplikasi Flutter."""
    while True:
        if latest_frame_bytes is not None:
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n"
                + latest_frame_bytes
                + b"\r\n"
            )
        time.sleep(0.04)


@app.get("/stream")
def get_stream():
    return StreamingResponse(
        generate_mjpeg_stream(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "service": "coffeevision-relay",
        "ai_inference": "flutter_tflite_local",
        "has_frame": latest_frame_bytes is not None,
        "time": time.time(),
    }


@app.put("/api/v1/devices/{device_id}/servo/command")
def set_servo_command(
    device_id: str,
    command: ServoCommand,
    x_device_token: Optional[str] = Header(default=None),
):
    """Aplikasi menyimpan target Pan/Tilt; ESP32 akan mengambilnya saat polling."""
    verify_device_token(x_device_token)
    previous = servo_commands.get(device_id, {})
    revision = previous.get("revision", 0) + 1
    payload = {
        "device_id": device_id,
        "pan": command.pan,
        "tilt": command.tilt,
        "revision": revision,
        "updated_at": time.time(),
    }
    servo_commands[device_id] = payload
    return payload


@app.get("/api/v1/devices/{device_id}/servo/command")
def get_servo_command(
    device_id: str,
    after: int = -1,
    x_device_token: Optional[str] = Header(default=None),
):
    """ESP32 melakukan polling; 204 berarti belum ada perintah baru."""
    verify_device_token(x_device_token)
    payload = servo_commands.get(device_id)
    if payload is None or payload["revision"] <= after:
        return Response(status_code=204)
    return payload


@app.post("/api/v1/devices/{device_id}/servo/state")
def update_servo_state(
    device_id: str,
    state: dict,
    x_device_token: Optional[str] = Header(default=None),
):
    verify_device_token(x_device_token)
    device_states[device_id] = {**state, "updated_at": time.time()}
    return {"status": "ok"}


@app.get("/api/v1/devices/{device_id}/state")
def get_device_state(
    device_id: str,
    x_device_token: Optional[str] = Header(default=None),
):
    verify_device_token(x_device_token)
    return device_states.get(device_id, {"status": "offline"})


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8000))
    print(f"Relay berjalan di port {port}; stream: http://localhost:{port}/stream")
    uvicorn.run("app:app", host="0.0.0.0", port=port)
