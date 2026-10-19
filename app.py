"""CoffeeVision relay untuk ESP32-CAM, ESP32 servo, dan aplikasi Flutter.

Server ini tidak menjalankan inferensi AI dan tidak memuat best.onnx. ESP32-CAM
mengirim JPEG ke /upload/{device_id}, aplikasi membaca frame dari /stream/{device_id},
sedangkan model best.tflite pada aplikasi Flutter melakukan deteksi secara lokal
di ponsel. Setiap perangkat diidentifikasi melalui device_id unik yang diatur
oleh pengguna.
"""

import os
import time
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

# Zona waktu Indonesia Barat (UTC+7)
WIB = timezone(timedelta(hours=7))

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
servo_last_seen: dict[str, float] = {}
DEVICE_TOKEN = os.environ.get("DEVICE_TOKEN", "")

# ============================================================
# SCAN SESSION STATE — in-memory per device
# scan_sessions[device_id] = { sesi aktif atau None }
# scan_history[device_id]  = [ list sesi selesai (maks 20) ]
# ============================================================
scan_sessions: dict[str, Optional[dict]] = {}
scan_history: dict[str, list] = {}
# Jadwal scan otomatis per perangkat
device_schedules: dict[str, dict] = {}


# ─────────────────────────────────────────────
# MODELS
# ─────────────────────────────────────────────

class ServoCommand(BaseModel):
    pan: int = Field(ge=0, le=180)
    tilt: int = Field(ge=0, le=180)


class BatteryReport(BaseModel):
    battery_voltage: float
    battery_percentage: int = Field(ge=0, le=100)
    raw_adc: Optional[int] = None
    device_id: Optional[str] = None


class ScanStartRequest(BaseModel):
    pan_min: int = Field(default=0,   ge=0, le=180)
    pan_max: int = Field(default=180, ge=0, le=180)
    tilt_min: int = Field(default=60, ge=0, le=180)
    tilt_max: int = Field(default=120, ge=0, le=180)
    pan_steps: int = Field(default=4, ge=2, le=12)
    tilt_steps: int = Field(default=3, ge=1, le=8)
    settle_ms: int = Field(default=800, ge=200, le=3000)
    label: Optional[str] = None   # Nama sesi opsional


class WaypointDoneRequest(BaseModel):
    waypoint_index: int
    detections: list[dict[str, Any]] = []  # [{ripeness, confidence, bbox}]
    has_detection: bool = False
    frame_base64: Optional[str] = None    # JPEG frame base64 jika ada deteksi


class ScheduleConfigRequest(BaseModel):
    enabled: bool = True
    interval_type: str = "3_days"  # "3_days" | "7_days" | "daily" | "demo_10m" | "demo_1h"
    custom_interval_hours: int = 72
    scan_hour: int = Field(default=8, ge=0, le=23)
    scan_minute: int = Field(default=0, ge=0, le=59)
    min_battery_percent: int = Field(default=0, ge=0, le=100)
    pan_steps: int = Field(default=4, ge=2, le=12)
    tilt_steps: int = Field(default=3, ge=1, le=8)
    settle_ms: int = Field(default=800, ge=200, le=3000)


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def verify_device_token(token: Optional[str]) -> None:
    """Token opsional saat pengembangan; wajib bila DEVICE_TOKEN di Railway diisi."""
    if DEVICE_TOKEN and token != DEVICE_TOKEN:
        raise HTTPException(status_code=401, detail="Token perangkat tidak valid")


def _generate_waypoints(
    pan_min: int, pan_max: int,
    tilt_min: int, tilt_max: int,
    pan_steps: int, tilt_steps: int,
) -> list[dict]:
    """
    Menghasilkan waypoint pola raster zig-zag.
    Baris genap (0,2,...) bergerak kiri→kanan, baris ganjil kanan→kiri.
    """
    pan_gap  = (pan_max  - pan_min)  / max(pan_steps  - 1, 1)
    tilt_gap = (tilt_max - tilt_min) / max(tilt_steps - 1, 1)

    waypoints = []
    idx = 0
    for ti in range(tilt_steps):
        tilt_val = round(tilt_min + ti * tilt_gap)
        pan_sequence = range(pan_steps) if ti % 2 == 0 else range(pan_steps - 1, -1, -1)
        for pi in pan_sequence:
            pan_val = round(pan_min + pi * pan_gap)
            waypoints.append({
                "index": idx,
                "pan": pan_val,
                "tilt": tilt_val,
                "state": "pending",      # pending | current | done | skipped
                "has_detection": False,
                "detections": [],
                "frame_base64": None,
                "completed_at": None,
            })
            idx += 1

    return waypoints


def _build_summary(waypoints: list[dict]) -> dict:
    """Hitung ringkasan kematangan dari semua deteksi dalam satu sesi."""
    counts: dict[str, int] = {}
    total = 0
    for wp in waypoints:
        for det in wp.get("detections", []):
            ripeness = det.get("ripeness", "unknown")
            counts[ripeness] = counts.get(ripeness, 0) + 1
            total += 1
    return {
        "total_detections": total,
        "by_class": counts,
        "waypoints_done": sum(1 for w in waypoints if w["state"] == "done"),
        "waypoints_with_detection": sum(1 for w in waypoints if w["has_detection"]),
    }


def _strip_frames(waypoints: list[dict]) -> list[dict]:
    """Hapus frame_base64 dari waypoint sebelum disimpan ke riwayat (hemat memori)."""
    return [
        {k: v for k, v in wp.items() if k != "frame_base64"}
        for wp in waypoints
    ]


# ─────────────────────────────────────────────
# CAMERA UPLOAD & STREAM (PER-DEVICE)
# ─────────────────────────────────────────────

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
    camera_frames[device_id] = {"bytes": contents, "timestamp": now}
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

    now = time.time()
    latest_frame_bytes = contents
    latest_timestamp = now
    # Otomatis route ke cam-1 untuk kompatibilitas jika sketch lama upload ke /upload
    camera_frames["cam-1"] = {"bytes": contents, "timestamp": now}
    return {"status": "ok", "timestamp": latest_timestamp, "bytes": len(contents)}


def generate_mjpeg_stream_device(device_id: str):
    """Meneruskan frame JPEG terbaru dari device_id tertentu sebagai MJPEG."""
    while True:
        device_data = camera_frames.get(device_id)
        # Jika belum ada frame per-device, gunakan frame dari legacy /upload jika ada
        frame_bytes = device_data["bytes"] if device_data is not None else latest_frame_bytes
        if frame_bytes is not None:
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n"
                + frame_bytes
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


# ─────────────────────────────────────────────
# DEVICE INFO
# ─────────────────────────────────────────────

@app.get("/devices")
def list_devices():
    """Menampilkan daftar perangkat yang terhubung ke server."""
    devices = []

    for dev_id, data in camera_frames.items():
        devices.append({
            "device_id": dev_id,
            "type": "camera",
            "has_frame": True,
            "last_update": data.get("timestamp", 0),
        })

    for dev_id in servo_commands:
        if dev_id not in [d["device_id"] for d in devices]:
            devices.append({
                "device_id": dev_id,
                "type": "servo",
                "has_frame": False,
                "last_update": servo_commands[dev_id].get("updated_at", 0),
            })

    for dev_id in device_states:
        if dev_id not in [d["device_id"] for d in devices]:
            devices.append({
                "device_id": dev_id,
                "type": "unknown",
                "has_frame": False,
                "last_update": device_states[dev_id].get("updated_at", 0),
            })

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
        "active_scans": sum(1 for s in scan_sessions.values() if s is not None),
        "time": time.time(),
    }


# ─────────────────────────────────────────────
# SERVO RELAY
# ─────────────────────────────────────────────

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
    servo_last_seen[device_id] = time.time()
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


# ─────────────────────────────────────────────
# BATTERY RELAY
# ─────────────────────────────────────────────

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
    servo_online = (now - servo_last_seen.get(device_id, 0)) < 15

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
                "is_online": (now - updated_at) < 30 or servo_online,
            }
        return {
            "device_id": device_id,
            "battery_voltage": 0.0,
            "battery_percentage": 0,
            "raw_adc": None,
            "updated_at": servo_last_seen.get(device_id, 0),
            "is_online": servo_online,
            "status": "connected_servo" if servo_online else "no_data",
        }

    is_online = ((now - payload.get("updated_at", 0)) < 30) or servo_online
    return {**payload, "is_online": is_online}


@app.get("/api/v1/devices/{device_id}/state")
def get_device_state(
    device_id: str,
    x_device_token: Optional[str] = Header(default=None),
):
    verify_device_token(x_device_token)
    return device_states.get(device_id, {"status": "offline"})


# ─────────────────────────────────────────────
# SCAN SESSION — state machine di server
# ─────────────────────────────────────────────

@app.post("/api/v1/devices/{device_id}/scan/start")
def start_scan(
    device_id: str,
    req: ScanStartRequest,
    x_device_token: Optional[str] = Header(default=None),
):
    """
    Mulai sesi scan baru. Menghasilkan waypoint raster zig-zag sesuai parameter.
    Mengembalikan sesi lengkap termasuk daftar waypoint dan waypoint pertama.
    """
    verify_device_token(x_device_token)

    current = scan_sessions.get(device_id)
    if current and current.get("state") == "SCANNING":
        raise HTTPException(status_code=409, detail="Scan sedang berjalan. Hentikan dulu.")

    waypoints = _generate_waypoints(
        pan_min=req.pan_min, pan_max=req.pan_max,
        tilt_min=req.tilt_min, tilt_max=req.tilt_max,
        pan_steps=req.pan_steps, tilt_steps=req.tilt_steps,
    )
    if waypoints:
        waypoints[0]["state"] = "current"

    session_id = str(uuid.uuid4())[:8].upper()
    now = time.time()
    session = {
        "session_id": session_id,
        "device_id": device_id,
        "label": req.label or f"Scan #{session_id}",
        "state": "SCANNING",
        "started_at": now,
        "finished_at": None,
        "settle_ms": req.settle_ms,
        "current_waypoint_index": 0,
        "total_waypoints": len(waypoints),
        "waypoints": waypoints,
        "total_detections": 0,
        "summary": {},
    }

    scan_sessions[device_id] = session
    return session


@app.post("/api/v1/devices/{device_id}/scan/stop")
def stop_scan(
    device_id: str,
    x_device_token: Optional[str] = Header(default=None),
):
    """Hentikan scan yang sedang berjalan dan simpan ke riwayat."""
    verify_device_token(x_device_token)

    session = scan_sessions.get(device_id)
    if not session:
        raise HTTPException(status_code=404, detail="Tidak ada sesi scan aktif.")

    session["state"] = "DONE"
    session["finished_at"] = time.time()
    session["summary"] = _build_summary(session["waypoints"])

    history = scan_history.setdefault(device_id, [])
    history.insert(0, {**session, "waypoints": _strip_frames(session["waypoints"])})
    if len(history) > 20:
        history.pop()

    scan_sessions[device_id] = None
    return {"status": "stopped", "summary": session["summary"]}


@app.post("/api/v1/devices/{device_id}/scan/pause")
def pause_scan(
    device_id: str,
    x_device_token: Optional[str] = Header(default=None),
):
    """Jeda scan sementara (servo berhenti di posisi terakhir)."""
    verify_device_token(x_device_token)
    session = scan_sessions.get(device_id)
    if not session or session.get("state") != "SCANNING":
        raise HTTPException(status_code=400, detail="Tidak ada sesi scan yang bisa dijeda.")
    session["state"] = "PAUSED"
    return {"status": "paused", "current_waypoint": session["current_waypoint_index"]}


@app.post("/api/v1/devices/{device_id}/scan/resume")
def resume_scan(
    device_id: str,
    x_device_token: Optional[str] = Header(default=None),
):
    """Lanjutkan scan yang dijeda."""
    verify_device_token(x_device_token)
    session = scan_sessions.get(device_id)
    if not session or session.get("state") != "PAUSED":
        raise HTTPException(status_code=400, detail="Scan tidak dalam keadaan PAUSED.")
    session["state"] = "SCANNING"
    return {"status": "scanning", "current_waypoint": session["current_waypoint_index"]}


@app.post("/api/v1/devices/{device_id}/scan/waypoint_done")
def waypoint_done(
    device_id: str,
    req: WaypointDoneRequest,
    x_device_token: Optional[str] = Header(default=None),
):
    """
    Flutter melaporkan bahwa satu waypoint telah selesai diproses:
    servo sudah bergerak, settle, frame diambil, dan TFLite sudah dijalankan.
    Server memperbarui state waypoint dan menggeser ke waypoint berikutnya.
    Mengembalikan waypoint berikutnya atau status scan_complete.
    """
    verify_device_token(x_device_token)

    session = scan_sessions.get(device_id)
    if not session or session.get("state") not in ("SCANNING", "PAUSED"):
        raise HTTPException(status_code=404, detail="Tidak ada sesi scan aktif.")

    idx = req.waypoint_index
    if idx < 0 or idx >= len(session["waypoints"]):
        raise HTTPException(status_code=400, detail=f"Indeks waypoint {idx} tidak valid.")

    wp = session["waypoints"][idx]
    wp["state"] = "done"
    wp["has_detection"] = req.has_detection
    wp["detections"] = req.detections
    wp["completed_at"] = time.time()

    # Simpan frame hanya jika ada deteksi (hemat memori)
    if req.has_detection and req.frame_base64:
        wp["frame_base64"] = req.frame_base64

    # Hitung total deteksi
    session["total_detections"] = sum(len(w["detections"]) for w in session["waypoints"])

    next_idx = idx + 1
    session["current_waypoint_index"] = next_idx

    # Semua waypoint selesai
    if next_idx >= session["total_waypoints"]:
        session["state"] = "DONE"
        session["finished_at"] = time.time()
        session["summary"] = _build_summary(session["waypoints"])

        history = scan_history.setdefault(device_id, [])
        history.insert(0, {**session, "waypoints": _strip_frames(session["waypoints"])})
        if len(history) > 20:
            history.pop()

        scan_sessions[device_id] = None

        return {
            "status": "scan_complete",
            "next_waypoint": None,
            "summary": session["summary"],
        }

    # Tandai waypoint berikutnya sebagai current
    next_wp = session["waypoints"][next_idx]
    next_wp["state"] = "current"

    return {
        "status": "next_waypoint",
        "next_waypoint": {k: v for k, v in next_wp.items() if k != "frame_base64"},
        "progress": {
            "done": next_idx,
            "total": session["total_waypoints"],
            "percent": round((next_idx / session["total_waypoints"]) * 100),
        },
    }


@app.get("/api/v1/devices/{device_id}/scan/status")
def get_scan_status(
    device_id: str,
    x_device_token: Optional[str] = Header(default=None),
):
    """Polling status scan saat ini oleh Flutter."""
    verify_device_token(x_device_token)

    session = scan_sessions.get(device_id)
    if not session:
        return {"state": "IDLE", "device_id": device_id, "session": None}

    idx = session["current_waypoint_index"]
    current_wp = None
    if idx < session["total_waypoints"]:
        current_wp = {k: v for k, v in session["waypoints"][idx].items() if k != "frame_base64"}

    return {
        "state": session["state"],
        "device_id": device_id,
        "session_id": session["session_id"],
        "label": session["label"],
        "current_waypoint": current_wp,
        "progress": {
            "done": idx,
            "total": session["total_waypoints"],
            "percent": round((idx / max(session["total_waypoints"], 1)) * 100),
        },
        "total_detections": session.get("total_detections", 0),
        "settle_ms": session["settle_ms"],
    }


@app.get("/api/v1/devices/{device_id}/scan/sessions")
def get_scan_history(
    device_id: str,
    limit: int = 10,
    x_device_token: Optional[str] = Header(default=None),
):
    """Riwayat sesi scan yang sudah selesai."""
    verify_device_token(x_device_token)
    history = scan_history.get(device_id, [])
    return {
        "device_id": device_id,
        "sessions": history[:limit],
        "total": len(history),
    }


@app.get("/api/v1/devices/{device_id}/scan/sessions/{session_id}")
def get_scan_session_detail(
    device_id: str,
    session_id: str,
    x_device_token: Optional[str] = Header(default=None),
):
    """Detail lengkap satu sesi scan termasuk semua waypoint dan deteksi."""
    verify_device_token(x_device_token)
    for s in scan_history.get(device_id, []):
        if s["session_id"] == session_id:
            return s
    raise HTTPException(status_code=404, detail="Sesi tidak ditemukan.")


# ─────────────────────────────────────────────
# JADWAL SCANNING OTOMATIS (PRD F-30 s/d F-35)
# ─────────────────────────────────────────────

def _calc_next_run(sched: dict) -> float:
    now = datetime.now(WIB)
    itype = sched.get("interval_type", "3_days")
    shour = sched.get("scan_hour", 8)
    smin = sched.get("scan_minute", 0)

    if itype == "demo_10m":
        return time.time() + 600
    if itype == "demo_1h":
        return time.time() + 3600

    target = now.replace(hour=shour, minute=smin, second=0, microsecond=0)
    if target <= now:
        target += timedelta(days=1)

    if itype == "3_days":
        target += timedelta(days=2)
    elif itype == "7_days":
        target += timedelta(days=6)

    return target.timestamp()


@app.get("/api/v1/devices/{device_id}/schedule")
def get_device_schedule(
    device_id: str,
    x_device_token: Optional[str] = Header(default=None),
):
    """Mengambil konfigurasi jadwal scan perangkat saat ini."""
    verify_device_token(x_device_token)
    sched = device_schedules.get(device_id)
    if not sched:
        # Default sesuai PRD: Tiap 3 hari jam 08:00 pagi
        sched = {
            "device_id": device_id,
            "enabled": True,
            "interval_type": "3_days",
            "custom_interval_hours": 72,
            "scan_hour": 8,
            "scan_minute": 0,
            "min_battery_percent": 0,
            "pan_steps": 4,
            "tilt_steps": 3,
            "settle_ms": 800,
            "next_run_timestamp": _calc_next_run({"interval_type": "3_days", "scan_hour": 8, "scan_minute": 0}),
            "updated_at": time.time(),
        }
        device_schedules[device_id] = sched

    next_ts = sched.get("next_run_timestamp", 0)
    dt_str = datetime.fromtimestamp(next_ts, tz=WIB).strftime("%d/%m/%Y %H:%M WIB") if next_ts > 0 else "-"

    return {
        **sched,
        "next_run_formatted": dt_str,
        "seconds_until_next_run": max(0, int(next_ts - time.time())),
    }


@app.put("/api/v1/devices/{device_id}/schedule")
def update_device_schedule(
    device_id: str,
    req: ScheduleConfigRequest,
    x_device_token: Optional[str] = Header(default=None),
):
    """Menyimpan atau memperbarui konfigurasi jadwal scan otomatis."""
    verify_device_token(x_device_token)

    sched_dict = req.dict()
    next_ts = _calc_next_run(sched_dict) if req.enabled else 0

    saved = {
        **sched_dict,
        "device_id": device_id,
        "next_run_timestamp": next_ts,
        "updated_at": time.time(),
    }
    device_schedules[device_id] = saved

    dt_str = datetime.fromtimestamp(next_ts, tz=WIB).strftime("%d/%m/%Y %H:%M WIB") if next_ts > 0 else "Nonaktif"
    return {
        "status": "success",
        "message": "Jadwal berhasil diperbarui",
        "schedule": {
            **saved,
            "next_run_formatted": dt_str,
            "seconds_until_next_run": max(0, int(next_ts - time.time())),
        },
    }


@app.get("/api/v1/devices/{device_id}/checkin")
def device_checkin(
    device_id: str,
    battery: Optional[int] = None,
    x_device_token: Optional[str] = Header(default=None),
):
    """
    Endpoint Check-in untuk ESP32 bangun dari Deep Sleep:
    Mengecek apakah saat ini sudah masuk jadwal scan dan baterai mencukupi.
    """
    verify_device_token(x_device_token)
    sched = device_schedules.get(device_id)

    if not sched or not sched.get("enabled", False):
        return {
            "should_scan": False,
            "reason": "Jadwal nonaktif",
            "sleep_seconds": 3600,
        }

    # Scanning diizinkan pada berapa pun persentase baterai (tanpa batas minimum)
    next_ts = sched.get("next_run_timestamp", 0)
    now = time.time()

    if now >= next_ts:
        # Perbarui jadwal berikutnya
        sched["next_run_timestamp"] = _calc_next_run(sched)
        return {
            "should_scan": True,
            "reason": "Jadwal scan telah tiba",
            "pan_steps": sched.get("pan_steps", 4),
            "tilt_steps": sched.get("tilt_steps", 3),
            "settle_ms": sched.get("settle_ms", 800),
            "sleep_seconds": 0,
        }

    return {
        "should_scan": False,
        "reason": "Belum saatnya scan",
        "sleep_seconds": max(60, int(next_ts - now)),
    }


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8000))
    print(f"Relay berjalan di port {port}; stream: http://localhost:{port}/stream")
    print(f"Per-device stream: http://localhost:{port}/stream/<device_id>")
    print(f"Per-device upload: POST http://localhost:{port}/upload/<device_id>")
    print(f"Scan start:  POST http://localhost:{port}/api/v1/devices/<id>/scan/start")
    print(f"Scan status: GET  http://localhost:{port}/api/v1/devices/<id>/scan/status")
    uvicorn.run("app:app", host="0.0.0.0", port=port)
