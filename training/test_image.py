"""
AI-Driven Precision Agrochemical Sprayer - Real Image Prediction (CLI)

Usage:
    python training/test_image.py path/to/image.jpg [--threshold 50.0]

Prints crop, healthy/diseased status, disease name, and confidence using the
REAL trained model. If confidence is below threshold, prints UNKNOWN result.
"""

import os
import sys
import json
import argparse

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import numpy as np
import tensorflow as tf
from PIL import Image

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")

IMG_SIZE = 224
DEFAULT_THRESHOLD = 50.0


def load_model_and_classes():
    model_path = os.path.join(MODEL_DIR, "plant_disease_model.keras")
    model = tf.keras.models.load_model(model_path)
    with open(os.path.join(MODEL_DIR, "class_names.json"), "r", encoding="utf-8") as f:
        class_info = json.load(f)
    return model, class_info


def predict_image(model, class_info, image_path):
    img = Image.open(image_path).convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
    arr = np.asarray(img, dtype=np.float32)[np.newaxis, ...] / 255.0
    probs = model.predict(arr, verbose=0)[0]

    idx = int(np.argmax(probs))
    conf = float(probs[idx]) * 100.0
    info = class_info[str(idx)]

    top3 = sorted(range(len(probs)), key=lambda i: probs[i], reverse=True)[:3]

    return {
        "class_index": idx,
        "folder_name": info["folder_name"],
        "crop": info.get("crop", "Unknown"),
        "display_name": info["display_name"],
        "disease": info.get("disease"),
        "status": info["status"],
        "confidence": conf,
        "top3": [
            {
                "class": class_info[str(t)]["folder_name"],
                "confidence": round(float(probs[t]) * 100, 2),
            }
            for t in top3
        ],
    }


def main():
    # Windows console default (cp1252) cannot encode emoji; make it safe.
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    ap = argparse.ArgumentParser(description="AI-Driven Precision Agrochemical Sprayer image prediction")
    ap.add_argument("image_path", help="Path to plant image")
    ap.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                    help="Confidence threshold (percent)")
    args = ap.parse_args()

    if not os.path.exists(args.image_path):
        print(f"[ERROR] Image not found: {args.image_path}")
        sys.exit(1)

    print("=" * 50)
    print("  AI-Driven Precision Agrochemical Sprayer - AI Plant Health Prediction")
    print("=" * 50)

    model, class_info = load_model_and_classes()
    res = predict_image(model, class_info, args.image_path)

    conf = res["confidence"]
    print(f"Crop Name   : {res['crop']}")
    print(f"Detected    : {res['display_name']}")
    print(f"Confidence  : {conf:.2f}%")

    if conf < args.threshold:
        print("Status      : \u26a0\ufe0f UNKNOWN / LOW CONFIDENCE")
        print("Message     : Unable to confidently identify this plant or disease.")
        print("\nTop-3 candidates:")
        for t in res["top3"]:
            print(f"              {t['class']}: {t['confidence']:.2f}%")
    elif res["status"] == "Healthy":
        print("Status      : \U0001f7e2 HEALTHY")
        print("Disease     : None")
        print("Message     : No disease detected.")
    else:
        print("Status      : \U0001f534 DISEASED")
        print(f"Disease     : {res['disease']}")
        print("Message     : Disease detected. Targeted inspection recommended.")
    print("=" * 50)


if __name__ == "__main__":
    main()