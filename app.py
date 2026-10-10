"""CoffeeVision relay untuk ESP32-CAM, ESP32 servo, dan aplikasi Flutter.

Server ini tidak menjalankan inferensi AI dan tidak memuat best.onnx. ESP32-CAM
mengirim JPEG ke /upload/{device_id}, aplikasi membaca frame dari /stream/{device_id},
sedangkan model best.tflite pada aplikasi Flutter melakukan deteksi secara lokal
di ponsel. Setiap perangkat diidentifikasi melalui device_id unik yang diatur
oleh pengguna.
"""

import os
import time
from typing import Optional

from fastapi import FastAPI, File, Header, HTTPException, Request, Response, UploadFile
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

# Frame paling baru dari setiap ESP32-CAM. Dict key = device_id.
# Penyimpanan in-memory ini cocok untuk satu instance Railway;
# gunakan Redis/object storage jika ingin multi-instance.
camera_frames: dict[str, dict] = {}

# Default frame untuk backward compatibility (tanpa device_id)
latest_frame_bytes: Optional[bytes] = None
latest_timestamp = 0.0

# Relay command untuk ESP32 di balik NAT. Perangkat melakukan polling untuk
# mengambil posisi servo terbaru tanpa membutuhkan IP publik.
servo_commands = {}
device_states = {}
battery_states: dict[str, dict] = {}
DEVICE_TOKEN = os.environ.get("DEVICE_TOKEN", "")


class ServoCommand(BaseModel):
    pan: int = Field(ge=0, le=180)
    tilt: int = Field(ge=0, le=180)


class BatteryReport(BaseModel):
    battery_voltage: float
    battery_percentage: int = Field(ge=0, le=100)
    raw_adc: Optional[int] = None
    device_id: Optional[str] = None



def verify_device_token(token: Optional[str]) -> None:
    """Token opsional saat pengembangan; wajib bila DEVICE_TOKEN di Railway diisi."""
    if DEVICE_TOKEN and token != DEVICE_TOKEN:
        raise HTTPException(status_code=401, detail="Token perangkat tidak valid")


# === CAMERA UPLOAD & STREAM (PER-DEVICE) ===

@app.post("/upload/{device_id}")
async def upload_frame_device(
    device_id: str,
    request: Request,
    file: Optional[UploadFile] = File(None),
):
    """Menerima JPEG dari ESP32-CAM (mendukung raw binary image/jpeg atau multipart/form-data)."""
    global latest_frame_bytes, latest_timestamp

    contents: Optional[bytes] = None
    if file is not None:
        contents = await file.read()
    else:
        contents = await request.body()

    if not contents:
        raise HTTPException(status_code=400, detail="Frame kosong")

    now = time.time()
    camera_frames[device_id] = {
        "bytes": contents,
        "timestamp": now,
    }
    # Tetap simpan ke global untuk backward compatibility
    latest_frame_bytes = contents
    latest_timestamp = now
    return {"status": "ok", "device_id": device_id, "timestamp": now, "bytes": len(contents)}


@app.post("/upload")
async def upload_frame_legacy(
    request: Request,
    file: Optional[UploadFile] = File(None),
):
    """Backward compatible: Menerima JPEG dari ESP32-CAM tanpa device_id."""
    global latest_frame_bytes, latest_timestamp

    contents: Optional[bytes] = None
    if file is not None:
        contents = await file.read()
    else:
        contents = await request.body()

    if not contents:
        raise HTTPException(status_code=400, detail="Frame kosong")

    latest_frame_bytes = contents
    latest_timestamp = time.time()
    return {"status": "ok", "timestamp": latest_timestamp, "bytes": len(contents)}


def generate_mjpeg_stream_device(device_id: str):
    """Meneruskan frame JPEG terbaru dari device_id tertentu sebagai MJPEG."""
    while True:
        device_data = camera_frames.get(device_id)
        if device_data is not None:
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n"
                + device_data["bytes"]
                + b"\r\n"
            )
        time.sleep(0.04)


def generate_mjpeg_stream():
    """Meneruskan frame JPEG terbaru (legacy, tanpa filter device_id)."""
    while True:
        if latest_frame_bytes is not None:
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n"
                + latest_frame_bytes
                + b"\r\n"
            )
        time.sleep(0.04)


@app.get("/stream/{device_id}")
def get_stream_device(device_id: str):
    """Stream frame MJPEG dari ESP32-CAM tertentu."""
    return StreamingResponse(
        generate_mjpeg_stream_device(device_id),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.get("/stream")
def get_stream():
    """Backward compatible: Stream frame MJPEG tanpa filter device_id."""
    return StreamingResponse(
        generate_mjpeg_stream(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


# === DEVICE INFO ===

@app.get("/devices")
def list_devices():
    """Menampilkan daftar perangkat yang terhubung ke server."""
    devices = []

    # Kamera yang pernah upload frame
    for dev_id, data in camera_frames.items():
        devices.append({
            "device_id": dev_id,
            "type": "camera",
            "has_frame": True,
            "last_update": data.get("timestamp", 0),
        })

    # Servo yang pernah menerima perintah
    for dev_id in servo_commands:
        if dev_id not in [d["device_id"] for d in devices]:
            devices.append({
                "device_id": dev_id,
                "type": "servo",
                "has_frame": False,
                "last_update": servo_commands[dev_id].get("updated_at", 0),
            })

    # Perangkat yang pernah kirim state
    for dev_id in device_states:
        if dev_id not in [d["device_id"] for d in devices]:
            devices.append({
                "device_id": dev_id,
                "type": "unknown",
                "has_frame": False,
                "last_update": device_states[dev_id].get("updated_at", 0),
            })

    # Tambahkan status baterai ke perangkat jika tersedia
    for dev in devices:
        b_info = battery_states.get(dev["device_id"])
        if b_info:
            dev["battery"] = {
                "voltage": b_info.get("battery_voltage"),
                "percentage": b_info.get("battery_percentage"),
                "is_online": b_info.get("is_online", False),
            }

    return {"devices": devices, "count": len(devices)}


@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "service": "coffeevision-relay",
        "ai_inference": "flutter_tflite_local",
        "has_frame": latest_frame_bytes is not None,
        "connected_cameras": len(camera_frames),
        "connected_servos": len(servo_commands),
        "time": time.time(),
    }


# === SERVO RELAY ===

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
    now = time.time()
    device_states[device_id] = {**state, "updated_at": now}
    if "battery_voltage" in state or "battery_percentage" in state:
        battery_states[device_id] = {
            "device_id": device_id,
            "battery_voltage": round(float(state.get("battery_voltage", 0.0)), 2),
            "battery_percentage": int(state.get("battery_percentage", 0)),
            "raw_adc": state.get("raw_adc"),
            "updated_at": now,
            "is_online": True,
        }
    return {"status": "ok"}


# === BATTERY RELAY ===

@app.post("/api/v1/devices/{device_id}/battery")
def update_battery_status(
    device_id: str,
    report: BatteryReport,
    x_device_token: Optional[str] = Header(default=None),
):
    """ESP32 melaporkan tegangan dan persentase baterai via pin ADC."""
    verify_device_token(x_device_token)
    now = time.time()
    payload = {
        "device_id": device_id,
        "battery_voltage": round(report.battery_voltage, 2),
        "battery_percentage": report.battery_percentage,
        "raw_adc": report.raw_adc,
        "updated_at": now,
        "is_online": True,
    }
    battery_states[device_id] = payload
    current_state = device_states.get(device_id, {})
    device_states[device_id] = {**current_state, **payload}
    return {"status": "ok", "battery": payload}


@app.get("/api/v1/devices/{device_id}/battery")
def get_battery_status(
    device_id: str,
    x_device_token: Optional[str] = Header(default=None),
):
    """Aplikasi Flutter mengambil persentase dan voltase baterai ESP32 terkini."""
    verify_device_token(x_device_token)
    payload = battery_states.get(device_id)
    now = time.time()
    if payload is None:
        state = device_states.get(device_id, {})
        if "battery_percentage" in state or "battery_voltage" in state:
            updated_at = state.get("updated_at", 0)
            return {
                "device_id": device_id,
                "battery_voltage": round(float(state.get("battery_voltage", 0.0)), 2),
                "battery_percentage": int(state.get("battery_percentage", 0)),
                "raw_adc": state.get("raw_adc"),
                "updated_at": updated_at,
                "is_online": (now - updated_at) < 30,
            }
        return {
            "device_id": device_id,
            "battery_voltage": 0.0,
            "battery_percentage": 0,
            "raw_adc": None,
            "updated_at": 0,
            "is_online": False,
            "status": "no_data",
        }

    is_online = (now - payload.get("updated_at", 0)) < 30
    return {**payload, "is_online": is_online}


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
    print(f"Per-device stream: http://localhost:{port}/stream/<device_id>")
    print(f"Per-device upload: POST http://localhost:{port}/upload/<device_id>")
    uvicorn.run("app:app", host="0.0.0.0", port=port)
