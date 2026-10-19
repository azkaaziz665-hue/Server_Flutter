"""Detector AI Server-side untuk CoffeeVision:
Menjalankan inferensi model YOLOv8 (best.tflite) langsung di server
secara otonom tanpa bergantung pada ponsel pengguna.
"""

import base64
import io
import math
import os
from typing import Any, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Warna bounding box untuk visualisasi bukti (sesuai tema kematangan)
CLASS_COLORS = {
    "ripe": (220, 38, 38),       # Merah (#DC2626)
    "semi_ripe": (217, 119, 6),  # Oranye / Kuning (#D97706)
    "unripe": (16, 185, 129),    # Hijau (#10B981)
    "overripe": (153, 27, 27),   # Merah gelap (#991B1B)
    "dry": (120, 113, 108),      # Abu kecokelatan (#78716C)
}

DEFAULT_LABELS = ["dry", "overripe", "ripe", "semi_ripe", "unripe"]

MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "best.tflite")
LABELS_PATH = os.path.join(os.path.dirname(__file__), "models", "labels.txt")


class CoffeeYoloDetector:
    def __init__(self):
        self.labels = self._load_labels()
        self.interpreter = None
        self.input_index = None
        self.output_index = None
        self.input_shape = None
        self._init_interpreter()

    def _load_labels(self) -> List[str]:
        if os.path.exists(LABELS_PATH):
            try:
                with open(LABELS_PATH, "r", encoding="utf-8") as f:
                    lines = [line.strip() for line in f.readlines() if line.strip()]
                    if lines:
                        return lines
            except Exception as e:
                print(f"[Detector] Gagal baca labels.txt: {e}")
        return DEFAULT_LABELS

    def _init_interpreter(self):
        if not os.path.exists(MODEL_PATH):
            print(f"[Detector] File model tidak ditemukan di: {MODEL_PATH}")
            return

        try:
            try:
                from ai_edge_litert.interpreter import Interpreter
            except ImportError:
                try:
                    from tflite_runtime.interpreter import Interpreter
                except ImportError:
                    from tensorflow.lite import Interpreter

            self.interpreter = Interpreter(model_path=MODEL_PATH)
            self.interpreter.allocate_tensors()

            in_details = self.interpreter.get_input_details()
            out_details = self.interpreter.get_output_details()

            self.input_index = in_details[0]["index"]
            self.output_index = out_details[0]["index"]
            self.input_shape = in_details[0]["shape"]  # e.g. [1, 3, 640, 640]
            print(f"[Detector] Model YOLOv8 TFLite berhasil dimuat! Input: {self.input_shape}")
        except Exception as e:
            print(f"[Detector] Error inisialisasi TFLite: {e}")
            self.interpreter = None

    def detect(
        self,
        image_bytes: bytes,
        conf_thresh: float = 0.25,
        iou_thresh: float = 0.45,
    ) -> Tuple[List[dict], Optional[str]]:
        """Mendeteksi buah kopi dalam frame JPEG.
        Mengembalikan tuple: (daftar_deteksi, evidence_base64_jpeg)
        """
        try:
            pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        except Exception as e:
            print(f"[Detector] Gagal decode frame citra: {e}")
            return [], None

        orig_w, orig_h = pil_img.size

        if self.interpreter is not None:
            detections = self._detect_tflite(pil_img, conf_thresh, iou_thresh)
        else:
            detections = self._detect_smart_vision(pil_img, conf_thresh)

        evidence_b64 = None
        if detections:
            evidence_b64 = self._draw_and_encode_evidence(pil_img, detections)

        return detections, evidence_b64

    def _detect_tflite(
        self,
        pil_img: Image.Image,
        conf_thresh: float,
        iou_thresh: float,
    ) -> List[dict]:
        # Resize ke 640x640
        input_w = 640
        input_h = 640
        if self.input_shape is not None and len(self.input_shape) == 4:
            # Format NCHW: [1, 3, H, W]
            if self.input_shape[1] == 3:
                input_h, input_w = int(self.input_shape[2]), int(self.input_shape[3])
            else:  # NHWC: [1, H, W, 3]
                input_h, input_w = int(self.input_shape[1]), int(self.input_shape[2])

        resized = pil_img.resize((input_w, input_h), Image.Resampling.BILINEAR)
        img_arr = np.array(resized, dtype=np.float32) / 255.0

        if self.input_shape is not None and self.input_shape[1] == 3:
            # Transpose HWC (640,640,3) -> CHW (3,640,640)
            img_arr = np.transpose(img_arr, (2, 0, 1))

        input_data = np.expand_dims(img_arr, axis=0)

        self.interpreter.set_tensor(self.input_index, input_data)
        self.interpreter.invoke()

        output = self.interpreter.get_tensor(self.output_index)
        # Format output YOLOv8: [1, 9, 8400] atau [1, 8400, 9]
        if output.shape[1] <= output.shape[2]:
            output = np.transpose(output, (0, 2, 1))  # Jadi [1, 8400, 9]

        boxes = []
        scores = []
        class_ids = []

        num_classes = min(len(self.labels), output.shape[2] - 4)
        for i in range(output.shape[1]):
            class_scores = output[0, i, 4 : 4 + num_classes]
            best_cls = int(np.argmax(class_scores))
            best_score = float(class_scores[best_cls])

            if best_score >= conf_thresh:
                cx = float(output[0, i, 0])
                cy = float(output[0, i, 1])
                w = float(output[0, i, 2])
                h = float(output[0, i, 3])

                # Jika koordinat dalam skala pixel 640, normalisasi ke 0..1
                if cx > 2.0 or cy > 2.0 or w > 2.0 or h > 2.0:
                    cx /= float(input_w)
                    cy /= float(input_h)
                    w /= float(input_w)
                    h /= float(input_h)

                left = max(0.0, min(1.0, cx - w / 2.0))
                top = max(0.0, min(1.0, cy - h / 2.0))
                right = max(0.0, min(1.0, cx + w / 2.0))
                bottom = max(0.0, min(1.0, cy + h / 2.0))

                if right > left and bottom > top:
                    boxes.append([left, top, right, bottom])
                    scores.append(best_score)
                    class_ids.append(best_cls)

        if not boxes:
            return []

        # NMS (Non-Maximum Suppression)
        keep = self._apply_nms(boxes, scores, iou_thresh)
        results = []
        for idx in keep:
            cls_name = self.labels[class_ids[idx]] if class_ids[idx] < len(self.labels) else "unknown"
            b = boxes[idx]
            results.append({
                "class": cls_name,
                "ripeness": cls_name,
                "confidence": round(scores[idx], 3),
                "box": [round(x, 4) for x in b],
                "center": [round((b[0] + b[2]) / 2, 4), round((b[1] + b[3]) / 2, 4)],
            })

        return results

    def _detect_smart_vision(self, pil_img: Image.Image, conf_thresh: float) -> List[dict]:
        """Fallback Computer Vision berbasis warna HSV jika TFLite runtime tidak tersedia."""
        hsv_img = pil_img.convert("HSV")
        w, h = hsv_img.size
        sample_w = 160
        sample_h = max(1, int(h * (sample_w / w)))
        resized = hsv_img.resize((sample_w, sample_h))
        arr = np.array(resized)

        results = []
        step = 16
        for y in range(0, sample_h - step, step):
            for x in range(0, sample_w - step, step):
                block = arr[y : y + step, x : x + step]
                avg_h = float(np.mean(block[:, :, 0]))
                avg_s = float(np.mean(block[:, :, 1]))
                avg_v = float(np.mean(block[:, :, 2]))

                if avg_s < 50 or avg_v < 40:
                    continue

                label = None
                conf = 0.70
                # Merah matang (Hue 0-15 atau 235-255)
                if (avg_h <= 18 or avg_h >= 235) and avg_s >= 80:
                    label = "ripe"
                    conf = min(0.92, 0.65 + avg_s / 500)
                # Kuning/Oranye setengah matang (Hue 19-45)
                elif 19 <= avg_h <= 45 and avg_s >= 70:
                    label = "semi_ripe"
                    conf = 0.80
                # Hijau belum matang (Hue 46-105)
                elif 46 <= avg_h <= 105 and avg_s >= 60:
                    label = "unripe"
                    conf = 0.75

                if label and conf >= conf_thresh:
                    left = round(x / sample_w, 4)
                    top = round(y / sample_h, 4)
                    right = round((x + step) / sample_w, 4)
                    bottom = round((y + step) / sample_h, 4)
                    results.append({
                        "class": label,
                        "ripeness": label,
                        "confidence": round(conf, 2),
                        "box": [left, top, right, bottom],
                        "center": [round((left + right) / 2, 4), round((top + bottom) / 2, 4)],
                    })

        # Batasi maksimal 15 deteksi
        return results[:15]

    def _apply_nms(self, boxes: List[List[float]], scores: List[float], iou_thresh: float) -> List[int]:
        if not boxes:
            return []

        x1 = np.array([b[0] for b in boxes])
        y1 = np.array([b[1] for b in boxes])
        x2 = np.array([b[2] for b in boxes])
        y2 = np.array([b[3] for b in boxes])
        s = np.array(scores)

        areas = (x2 - x1) * (y2 - y1)
        order = s.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(int(i))
            if order.size == 1:
                break

            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter = w * h

            ovr = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
            inds = np.where(ovr <= iou_thresh)[0]
            order = order[inds + 1]

        return keep

    def _draw_and_encode_evidence(self, pil_img: Image.Image, detections: List[dict]) -> str:
        """Menggambar bounding box dan label pada citra lalu encode ke Base64 JPEG."""
        draw_img = pil_img.copy()
        draw = ImageDraw.Draw(draw_img)
        w, h = draw_img.size

        for det in detections:
            cls_name = det.get("class", "unknown")
            conf = det.get("confidence", 0.0)
            color = CLASS_COLORS.get(cls_name, (59, 130, 246))

            box = det.get("box", [0, 0, 0, 0])
            x1 = int(box[0] * w)
            y1 = int(box[1] * h)
            x2 = int(box[2] * w)
            y2 = int(box[3] * h)

            # Gambar bounding box dengan ketebalan 3px
            for offset in range(3):
                draw.rectangle([x1 - offset, y1 - offset, x2 + offset, y2 + offset], outline=color)

            # Label teks
            text = f"{cls_name.upper()} {int(conf * 100)}%"
            # Gambar background label
            text_bbox = draw.textbbox((x1, max(0, y1 - 18)), text) if hasattr(draw, "textbbox") else [x1, y1 - 18, x1 + 80, y1]
            draw.rectangle(text_bbox, fill=color)
            draw.text((x1 + 2, max(0, y1 - 18)), text, fill=(255, 255, 255))

        out_buf = io.BytesIO()
        draw_img.save(out_buf, format="JPEG", quality=82)
        return base64.b64encode(out_buf.getvalue()).decode("utf-8")


# Singleton instance
detector = CoffeeYoloDetector()
