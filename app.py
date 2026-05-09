import base64
import io
import json
import os

import cv2
import numpy as np
import onnxruntime as ort
from flask import Flask, jsonify, render_template, request
from PIL import Image, ImageOps

# ─── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")
ONNX_PATH = os.path.join(MODEL_DIR, "yolov8m_php_bills.onnx")
META_PATH = os.path.join(MODEL_DIR, "model_metadata.json")

# ─── Load Metadata ─────────────────────────────────────────────────────────────
with open(META_PATH) as f:
    META = json.load(f)

CLASSES      = META["classes"]                    # ["1000", "500"]
CONF_THRESH  = float(META["confidence_threshold"])  # 0.15
IOU_THRESH   = float(META["iou_threshold"])         # 0.5
INPUT_SIZE   = int(META["input_size"])              # 640

# Denomination values (₱)
BILL_VALUES = {c: int(c) for c in CLASSES}         # {"1000": 1000, "500": 500}

# Per-class BGR colors for OpenCV annotation
# ₱500 → warm gold,  ₱1000 → sky blue
CLASS_COLORS_BGR = {
    "500":  (30,  180, 220),   # gold/amber  (BGR)
    "1000": (220, 120,  40),   # blue        (BGR)
}

# ─── Load ONNX Model ───────────────────────────────────────────────────────────
_providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
session    = ort.InferenceSession(ONNX_PATH, providers=_providers)
INPUT_NAME = session.get_inputs()[0].name
print(f"✅ Model loaded — input: {INPUT_NAME}  |  providers: {session.get_providers()}")


# ─── Preprocessing ─────────────────────────────────────────────────────────────

def _contrast_stretch(arr: np.ndarray, lo: float = 2.0, hi: float = 98.0) -> np.ndarray:
    """Per-channel percentile contrast stretch."""
    out = np.empty_like(arr)
    for c in range(3):
        ch = arr[:, :, c].astype(np.float32)
        p_lo, p_hi = np.percentile(ch, lo), np.percentile(ch, hi)
        if p_hi > p_lo:
            out[:, :, c] = np.clip((ch - p_lo) / (p_hi - p_lo) * 255, 0, 255).astype(np.uint8)
        else:
            out[:, :, c] = arr[:, :, c]
    return out


def _apply_clahe(arr: np.ndarray) -> np.ndarray:
    """CLAHE in LAB space for uniform perceptual enhancement."""
    lab = cv2.cvtColor(arr, cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def _sharpen(arr: np.ndarray) -> np.ndarray:
    """Mild unsharp mask — enhances fine print and serial numbers."""
    blur = cv2.GaussianBlur(arr, (0, 0), sigmaX=1.5)
    return cv2.addWeighted(arr, 1.4, blur, -0.4, 0)


def preprocess(pil_img: Image.Image):
    """
    Full preprocessing pipeline matching training:
      Auto-Orient → Contrast Stretch → CLAHE → Sharpen → Fit-Within 640×640

    Returns:
        tensor        (1, 3, 640, 640) float32 — model input
        params        dict — transform params for mapping boxes back
        original_rgb  np.ndarray — unmodified RGB image for annotation
    """
    pil_img      = ImageOps.exif_transpose(pil_img).convert("RGB")
    original_rgb = np.array(pil_img)          # keep for drawing

    arr = _contrast_stretch(original_rgb)
    arr = _apply_clahe(arr)
    arr = _sharpen(arr)

    h, w   = arr.shape[:2]
    scale  = min(INPUT_SIZE / w, INPUT_SIZE / h)
    nw, nh = int(w * scale), int(h * scale)
    resized = cv2.resize(arr, (nw, nh), interpolation=cv2.INTER_LINEAR)

    pt = (INPUT_SIZE - nh) // 2
    pb = INPUT_SIZE - nh - pt
    pl = (INPUT_SIZE - nw) // 2
    pr = INPUT_SIZE - nw - pl
    padded = cv2.copyMakeBorder(resized, pt, pb, pl, pr,
                                cv2.BORDER_CONSTANT, value=(0, 0, 0))

    params = {"scale": scale, "pad_top": pt, "pad_left": pl,
              "orig_w": w, "orig_h": h}

    tensor = padded.astype(np.float32) / 255.0
    tensor = tensor.transpose(2, 0, 1)[np.newaxis]   # (1, 3, 640, 640)
    return tensor, params, original_rgb


# ─── NMS ───────────────────────────────────────────────────────────────────────

def _nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> list:
    """Standard greedy NMS. boxes: (N, 4) x1y1x2y2; scores: (N,)."""
    if len(boxes) == 0:
        return []
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas  = (x2 - x1).clip(0) * (y2 - y1).clip(0)
    order  = scores.argsort()[::-1]
    keep   = []
    while order.size > 0:
        i = order[0]
        keep.append(int(i))
        if order.size == 1:
            break
        xx1   = np.maximum(x1[i], x1[order[1:]])
        yy1   = np.maximum(y1[i], y1[order[1:]])
        xx2   = np.minimum(x2[i], x2[order[1:]])
        yy2   = np.minimum(y2[i], y2[order[1:]])
        inter = (xx2 - xx1).clip(0) * (yy2 - yy1).clip(0)
        union = areas[i] + areas[order[1:]] - inter + 1e-6
        iou   = inter / union
        order = order[np.where(iou <= iou_threshold)[0] + 1]
    return keep


# ─── Inference ─────────────────────────────────────────────────────────────────

def run_inference(tensor: np.ndarray, params: dict) -> list:
    """
    Run ONNX session and post-process YOLOv8 output.

    YOLOv8 ONNX output: (1, 4+num_classes, 8400)
    Rows 0-3 → cx, cy, w, h  (in 640×640 pixel space)
    Rows 4+  → class scores

    Returns list of dicts:
        {"class": str, "confidence": float, "box": [x1, y1, x2, y2]}
    """
    raw   = session.run(None, {INPUT_NAME: tensor})[0]  # (1, 6, 8400)
    preds = raw[0].T                                     # (8400, 6)

    class_scores = preds[:, 4:]                          # (8400, num_classes)
    conf         = class_scores.max(axis=1)
    class_ids    = class_scores.argmax(axis=1)

    mask = conf >= CONF_THRESH
    if not mask.any():
        return []

    preds_f  = preds[mask]
    conf_f   = conf[mask]
    cids_f   = class_ids[mask]

    # cx,cy,w,h → x1,y1,x2,y2  (model 640×640 space)
    cx, cy = preds_f[:, 0], preds_f[:, 1]
    bw, bh = preds_f[:, 2], preds_f[:, 3]
    boxes_model = np.stack([cx - bw / 2, cy - bh / 2,
                            cx + bw / 2, cy + bh / 2], axis=1)

    scale = params["scale"]
    pl    = params["pad_left"]
    pt    = params["pad_top"]
    ow, oh = params["orig_w"], params["orig_h"]

    detections = []
    for cid in np.unique(cids_f):
        idx  = np.where(cids_f == cid)[0]
        keep = _nms(boxes_model[idx], conf_f[idx], IOU_THRESH)
        for k in keep:
            orig_i = idx[k]
            bm     = boxes_model[orig_i]
            # Map from 640×640 model space → original image coordinates
            ox1 = int(max(0,  (bm[0] - pl) / scale))
            oy1 = int(max(0,  (bm[1] - pt) / scale))
            ox2 = int(min(ow, (bm[2] - pl) / scale))
            oy2 = int(min(oh, (bm[3] - pt) / scale))
            detections.append({
                "class":      CLASSES[int(cid)],
                "confidence": round(float(conf_f[orig_i]), 4),
                "box":        [ox1, oy1, ox2, oy2],
            })

    # Sort by confidence (highest first)
    detections.sort(key=lambda d: d["confidence"], reverse=True)
    return detections


# ─── Annotation ────────────────────────────────────────────────────────────────

def annotate_image(rgb: np.ndarray, detections: list) -> np.ndarray:
    """Draw bounding boxes + labels onto the RGB image. Returns annotated RGB."""
    img   = rgb.copy()
    short = min(img.shape[:2])

    # Dynamic sizing based on image dimensions
    box_thick  = max(2, int(short * 0.004))
    font_scale = max(0.45, short * 0.0012)
    font_thick = max(1, int(short * 0.003))

    for det in detections:
        cls   = det["class"]
        conf  = det["confidence"]
        x1, y1, x2, y2 = det["box"]
        color = CLASS_COLORS_BGR.get(cls, (0, 200, 100))
        label = f"  P{cls}  {conf:.0%}"

        # Bounding box
        cv2.rectangle(img, (x1, y1), (x2, y2), color, box_thick)

        # Label background pill
        (tw, th), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thick)
        pad    = 6
        lx1, ly1 = x1, max(0, y1 - th - baseline - pad * 2)
        lx2, ly2 = x1 + tw + pad, y1
        cv2.rectangle(img, (lx1, ly1), (lx2, ly2), color, -1)
        cv2.putText(img, label,
                    (lx1 + pad // 2, ly2 - baseline - pad // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                    (255, 255, 255), font_thick, cv2.LINE_AA)
    return img

def img_to_base64_jpeg(rgb: np.ndarray) -> str:
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    _, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 92])
    return base64.b64encode(buf).decode()


# ─── Flask App ─────────────────────────────────────────────────────────────────
app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/detect", methods=["POST"])
def detect():
    """
    POST /detect
    Body (JSON): { "image": "data:image/jpeg;base64,..." }
    Response (JSON):
        detections      list of {class, confidence, box}
        counts          {"500": int, "1000": int}
        total           int  (total peso value)
        annotated_image data URI of annotated JPEG
    """
    payload = request.get_json(silent=True)
    if not payload or "image" not in payload:
        return jsonify({"error": "No image provided"}), 400

    try:
        raw_b64 = payload["image"].split(",")[-1]
        img_bytes = base64.b64decode(raw_b64)
        pil_img   = Image.open(io.BytesIO(img_bytes))

        tensor, params, original_rgb = preprocess(pil_img)
        detections  = run_inference(tensor, params)
        annotated   = annotate_image(original_rgb, detections)
        ann_b64     = img_to_base64_jpeg(annotated)

        counts = {"500": 0, "1000": 0}
        for d in detections:
            counts[d["class"]] = counts.get(d["class"], 0) + 1

        total = sum(BILL_VALUES.get(c, 0) * n for c, n in counts.items())

        return jsonify({
            "detections":      detections,
            "counts":          counts,
            "total":           total,
            "annotated_image": f"data:image/jpeg;base64,{ann_b64}",
        })

    except Exception as exc:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(exc)}), 500


# ─── Entry Point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)