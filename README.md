# PHP Bill Detector — Web App

A Flask web application that detects old-series ₱500 and ₱1,000 Philippine banknotes
using a YOLOv8m ONNX model with real-time camera support and drag-and-drop image upload.

---

## Quick Start

### 1. Clone / copy the project
```
php_bill_detector/
├── app.py
├── pipeline.py
├── requirements.txt
├── render.yaml
├── Procfile
├── .gitignore
├── model/
│   ├── yolov8m_php_bills.onnx
│   └── model_metadata.json
└── static/
    └── styles.css
└── templates/
    └── index.html
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the app
```bash
python app.py
```

Then open **http://localhost:5000** in your browser.

---

## Features

| Feature | Description |
| Image Upload | Drag-and-drop or click-to-browse (JPG / PNG / WEBP) |
| Camera Capture | Live camera preview with single-frame capture |
| Auto Detection | YOLOv8m detects ₱500 and ₱1000 bills in the image |
| Bill Counting | Automatic per-denomination count and total peso value |
| Annotated View | Returns image with bounding boxes and confidence labels |
| Responsive UI | Works on desktop and mobile browsers |

---

## Preprocessing Pipeline

Matches the training pipeline exactly:

1. Auto-orient — strips EXIF rotation tag
2. Contrast stretch — per-channel p2–p98 percentile normalization
3. CLAHE — adaptive histogram equalization (LAB color space)
4. Sharpen — unsharp mask (σ=1.5, weight=1.4)
5. Fit-within resize — scale to 640×640 with black padding (preserves aspect ratio)

---

## API

### `POST /detect`

**Request body (JSON):**
```json
{ "image": "data:image/jpeg;base64,..." }
```

**Response (JSON):**
```json
{
  "detections": [
    { "class": "500",  "confidence": 0.94, "box": [x1, y1, x2, y2] },
    { "class": "1000", "confidence": 0.87, "box": [x1, y1, x2, y2] }
  ],
  "counts": { "500": 1, "1000": 1 },
  "total": 1500,
  "annotated_image": "data:image/jpeg;base64,..."
}
```

---

## Model Info

| Property | Value |
|---|---|
| Architecture | YOLOv8m |
| Classes | ₱500, ₱1000 (old series) |
| Input size | 640×640 |
| Confidence threshold | 0.15 |
| IoU threshold | 0.5 |
| Export format | ONNX (opset 17) |

---

## Requirements

- Python 3.9+
- Flask ≥ 2.3
- onnxruntime ≥ 1.17
- opencv-python ≥ 4.8
- Pillow ≥ 10.0
- numpy ≥ 1.24
- gunicorn ≥ 21.0.0

GPU acceleration is automatically used if `onnxruntime-gpu` is installed.