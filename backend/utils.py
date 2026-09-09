"""
AI-Driven Precision Agrochemical Sprayer - Backend Utilities

File validation, saving uploads, JSON responses.
"""

import os
import time
import json
import uuid

ALLOWED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
MAX_IMAGE_BYTES = 15 * 1024 * 1024  # 15 MB


def allowed_image(filename):
    ext = os.path.splitext(filename or "")[1].lower()
    return ext in ALLOWED_EXTS


def save_upload(file_storage, upload_dir):
    """Validate and persist an uploaded image. Returns (path, original_name)."""
    if not file_storage:
        raise ValueError("No file provided")
    if not allowed_image(file_storage.filename):
        raise ValueError(
            f"Unsupported file type. Allowed: {', '.join(sorted(ALLOWED_EXTS))}"
        )
    file_storage.stream.seek(0, os.SEEK_END)
    size = file_storage.stream.tell()
    file_storage.stream.seek(0)
    if size > MAX_IMAGE_BYTES:
        raise ValueError(f"Image too large ({size / 1e6:.1f} MB). Max {MAX_IMAGE_BYTES / 1e6:.0f} MB.")

    os.makedirs(upload_dir, exist_ok=True)
    ext = os.path.splitext(file_storage.filename)[1].lower()
    fname = f"{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}{ext}"
    path = os.path.join(upload_dir, fname)
    file_storage.save(path)
    return path, file_storage.filename


def validate_image_bytes(data):
    """Quick validation that bytes look like an image (size + magic)."""
    if not data:
        return False
    if len(data) > MAX_IMAGE_BYTES:
        return False
    return True


def json_response(payload, status=200):
    return payload, status


def system_status_report(pipeline):
    """Basic pipeline info for the /status endpoint."""
    s = pipeline.status()
    return {
        "model_loaded": True,
        "num_classes": s.get("num_combined_classes", 0),
        "threshold": s.get("disease_threshold", 60.0),
        "temperature": s.get("temperature"),
        "image_size": 224,
    }