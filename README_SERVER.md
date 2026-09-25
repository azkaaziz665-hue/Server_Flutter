# CoffeeVision Railway Relay

Server ini menerima frame dari ESP32-CAM, menyediakan stream MJPEG untuk aplikasi Flutter, dan menjadi relay perintah servo untuk ESP32 yang berada di jaringan berbeda. Server tidak memuat atau menjalankan `best.onnx`; deteksi dilakukan oleh `best.tflite` secara lokal di aplikasi Flutter.

## Deploy ke Railway

1. Buat service baru dari folder `server`.
2. Railway akan memakai `Dockerfile` yang tersedia.
3. Opsional tetapi disarankan: tambahkan variable environment berikut di Railway.

```text
DEVICE_TOKEN=token-rahasia-yang-panjang
```

4. Setelah deploy, salin domain publik Railway, misalnya `https://coffeevision-production.up.railway.app`.
5. Buka `<domain>/health`; respons `status: ok` menandakan service siap.

## Konfigurasi perangkat

- ESP32-CAM: pada `assets/esp32/esp32_cam_client_push.ino`, isi `serverUploadUrl` menjadi `<domain>/upload`.
- ESP32 servo: gunakan `assets/esp32/esp32_servo_railway_relay.ino`, lalu isi Wi-Fi, `RAILWAY_URL`, `SERVO_DEVICE_ID`, dan token yang sama.
- Aplikasi Flutter: tombol Pengaturan pada header cukup membutuhkan URL Railway, ID servo, dan token.

## Endpoint

| Endpoint | Fungsi |
|---|---|
| `POST /upload` | ESP32-CAM mengunggah frame JPEG. |
| `GET /stream` | Stream MJPEG untuk aplikasi. |
| `GET /detections` | Deteksi terbaru. |
| `GET /health` | Uji koneksi dari aplikasi. |
| `PUT /api/v1/devices/{id}/servo/command` | Aplikasi menyimpan target Pan/Tilt. |
| `GET /api/v1/devices/{id}/servo/command?after=n` | ESP32 mengambil target baru; `204` artinya belum ada perintah baru. |

Relay menyimpan perintah terakhir di memori instance Railway. Setelah service di-redeploy, posisi servo kembali ke posisi awal sampai joystick mengirim perintah baru.
