# Panduan Menjalankan Server Relay & YOLO (Beda Jaringan)

Panduan ini digunakan jika **ESP32-CAM dan HP Anda berada di jaringan WiFi/Internet yang berbeda**.

---

## 1. Persiapan Server di Laptop / PC
Pastikan laptop/PC Anda sudah terpasang **Python 3.9+**.

Buka terminal di folder proyek ini dan install modul yang diperlukan:
```bash
pip install fastapi uvicorn opencv-python ultralytics python-multipart numpy
```

---

## 2. Menjalankan Server Python
Jalankan file server:
```bash
python server/app.py
```
Server akan aktif di port `8000`:
- Upload Endpoint: `http://localhost:8000/upload`
- Stream Endpoint: `http://localhost:8000/stream`
- Detections API: `http://localhost:8000/detections`

---

## 3. Membuat URL Publik Gratis (Agar Dapat Diakses dari Luar Jaringan)

Gunakan **Ngrok** (gratis) untuk membuka port 8000 ke internet publik:
```bash
ngrok http 8000
```
Anda akan mendapatkan link publik HTTPS, contoh:
`https://abcd-1234.ngrok-free.app`

---

## 4. Konfigurasi ke Perangkat

### A. Pada ESP32-CAM (`assets/esp32/esp32_cam_client_push.ino`)
Buka file `esp32_cam_client_push.ino`, lalu masukkan link ngrok upload:
```cpp
const char* serverUploadUrl = "https://abcd-1234.ngrok-free.app/upload";
```
Upload kode ini ke ESP32-CAM Anda.

### B. Pada Aplikasi Flutter di HP Anda
Buka aplikasi Flutter di HP (bisa menggunakan paket data 4G/5G atau WiFi mana pun), masuk ke menu **Pengaturan (Ikon Roda Gigi)**:
- Masukkan URL Stream: `https://abcd-1234.ngrok-free.app/stream`
- Klik **Simpan**.

Sekarang video stream dan deteksi YOLO kematangan kopi akan langsung tampil di HP Anda dari jarak mana pun melalui internet!
