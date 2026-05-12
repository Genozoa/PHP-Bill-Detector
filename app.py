import base64
import io
from flask import Flask, jsonify, render_template, request
from PIL import Image

from pipeline import CLASSES, BILL_VALUES, preprocess, run_inference, annotate_image, img_to_base64_jpeg

#Flask App
app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/detect", methods=["POST"])
def detect():
    """
    POST /detect
    Body (JSON): { "image": "data:image/jpeg;base64,...", "confidence": float, "iou": float }
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
        print(f"Processing detection request - image size: {len(payload['image'])} chars")
        raw_b64 = payload["image"].split(",")[-1]
        img_bytes = base64.b64decode(raw_b64)
        pil_img   = Image.open(io.BytesIO(img_bytes))
        print(f"Image loaded: {pil_img.size}")

        # Get dynamic thresholds from client
        conf_threshold = float(payload.get("confidence", 0.15))
        iou_threshold = float(payload.get("iou", 0.5))
        print(f"Using thresholds: conf={conf_threshold}, iou={iou_threshold}")

        tensor, params, original_rgb = preprocess(pil_img)
        print(f"Preprocessing done: tensor shape {tensor.shape}")

        detections = run_inference(tensor, params, conf_threshold, iou_threshold)
        print(f"Inference done: {len(detections)} detections")

        annotated = annotate_image(original_rgb, detections)
        ann_b64 = img_to_base64_jpeg(annotated)
        print(f"Annotation done: annotated image size {len(ann_b64)} bytes")

        counts = {c.split('_')[1]: 0 for c in CLASSES}
        for d in detections:
            counts[d["class"].split('_')[1]] += 1

        total = sum(int(c) * n for c, n in counts.items())
        print(f"Counts: {counts}, Total: {total}")

        return jsonify({
            "detections":      detections,
            "counts":          counts,
            "total":           total,
            "annotated_image": f"data:image/jpeg;base64,{ann_b64}",
        })

    except MemoryError:
        print("MemoryError during processing")
        return jsonify({"error": "Insufficient memory for processing. Try a smaller image."}), 500
    except Exception as exc:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(exc)}), 500


if __name__ == "__main__":
    import os

    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host="0.0.0.0", port=port)
