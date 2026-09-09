"""
AI-Driven Precision Agrochemical Sprayer - Flask Backend Server

Multi-stage AI pipeline:

    POST /predict
        validate plant → identify crop → supported-crop check →
        crop-specific disease detection → severity analysis → result

Runs on 0.0.0.0 so a mobile phone camera (and later an ESP32-CAM) on the same
Wi-Fi network can reach it. The backend is camera-source independent; both
mobile and ESP32-CAM POST the same multipart image to /predict.

Endpoints:
    GET  /                   -> service info
    GET  /status              -> pipeline / model status
    GET  /supported_crops     -> list of supported crops (real data only)
    POST /predict             -> upload image, returns REAL multi-stage result
"""

import os
import sys

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
UPLOAD_DIR = os.path.join(PROJECT_ROOT, "uploads")
FRONTEND_DIR = os.path.join(PROJECT_ROOT, "frontend")

from flask import Flask, request, jsonify
from flask_cors import CORS

from backend.pipeline import get_pipeline
from backend.utils import save_upload, ALLOWED_EXTS

app = Flask(__name__, static_folder=FRONTEND_DIR, static_url_path="")
# CORS for mobile devices on the same Wi-Fi. Default: allow any origin.
# Restrict with env SPRAYBOT_CORS_ORIGINS="http://192.168.1.10:5000,http://..."
cors_origins = os.environ.get("SPRAYBOT_CORS_ORIGINS", "*")
CORS(app, resources={r"/*": {
    "origins": [o.strip() for o in cors_origins.split(",") if o.strip()],
    "methods": ["GET", "POST", "OPTIONS"],
    "allow_headers": ["Content-Type", "Authorization"],
}})
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024
app.config["JSON_AS_ASCII"] = False          # real UTF-8 (emoji) in JSON, no \u escapes


@app.after_request
def utf8_headers(resp):
    ct = resp.headers.get("Content-Type", "")
    if ct.startswith("text/") and "charset=" not in ct:
        resp.headers["Content-Type"] = ct + "; charset=utf-8"
    elif ct.startswith("application/json") and "charset=" not in ct:
        resp.headers["Content-Type"] = ct + "; charset=utf-8"
    resp.headers.setdefault("Cache-Control", "no-store, must-revalidate")
    resp.headers.setdefault("X-Content-Type-Options", "nosniff")
    return resp


@app.route("/", methods=["GET"])
def index():
    return app.send_static_file("index.html")


@app.route("/api/config", methods=["GET"])
def api_config():
    """Frontend API base URL config (mobile-safe, never localhost).

    The phone loads this page from the laptop over Wi-Fi, so request.host is
    already http://<laptop-lan-ip>:<port>. That auto-base-URL is returned.
    An explicit override can be forced with env SPRAYBOT_API_BASE_URL.
    """
    scheme = request.headers.get("X-Forwarded-Proto", request.scheme)
    base = os.environ.get("SPRAYBOT_API_BASE_URL") or f"{scheme}://{request.host}"
    return jsonify({
        "base_url": base.rstrip("/"),
        "port": int(os.environ.get("SPRAYBOT_PORT", "5000")),
        "host": os.environ.get("SPRAYBOT_HOST", "0.0.0.0"),
    })


@app.route("/api/info", methods=["GET"])
def info():
    return jsonify({
        "success": True,
        "service": "AI-Driven Precision Agrochemical Sprayer",
        "message": "Multi-stage plant validation, crop identification, "
                   "disease detection and severity analysis API",
        "endpoints": ["/", "/status", "/supported_crops", "/predict"],
    })


@app.route("/status", methods=["GET"])
def status():
    try:
        p = get_pipeline()
        payload = p.status()
        payload["model_loaded"] = True
        return jsonify(payload)
    except FileNotFoundError as e:
        return jsonify({
            "model_loaded": False,
            "error": str(e),
            "message": "Combined model not found. Run training/train_model.py "
                       "(or a training script) first.",
        }), 503
    except Exception as e:
        return jsonify({"model_loaded": False, "error": str(e)}), 500


@app.route("/supported_crops", methods=["GET"])
def supported_crops():
    p = get_pipeline()
    return jsonify({
        "success": True,
        "supported_crops": p.supported,
        "crop_metadata": p.crop_meta,
        "note": "true = real disease model available; false = identifiable "
                "but disease detection not yet available.",
    })


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"success": False, "message": "No image part in request."}), 400

    file = request.files["image"]
    try:
        save_upload(file, UPLOAD_DIR)
    except ValueError as e:
        return jsonify({"success": False, "message": str(e)}), 400

    file.stream.seek(0)
    data = file.stream.read()
    if not data:
        return jsonify({"success": False, "message": "Empty image."}), 400

    try:
        p = get_pipeline()
        result = p.predict_bytes(data)
        return jsonify(result)
    except Exception as e:
        return jsonify({
            "success": False,
            "message": f"Prediction error: {e}",
        }), 500


if __name__ == "__main__":
    host = os.environ.get("SPRAYBOT_HOST", "0.0.0.0")
    port = int(os.environ.get("SPRAYBOT_PORT", "5000"))
    app.run(host=host, port=port, debug=False)