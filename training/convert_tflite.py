"""
AI-Driven Precision Agrochemical Sprayer - TensorFlow Lite Conversion + ESP32 Compatibility Check

Converts models/plant_disease_model.keras -> models/plant_disease_model.tflite
and reports the model size and compatibility constraints for the target
ESP32-CAM hardware.

IMPORTANT (honesty requirement):
  The full multi-crop model is generally TOO LARGE to execute on an ESP32-CAM
  (which has ~320 KB RAM usable by the TFLite Micro runtime). This script
  prints the actual size and provides the real guidance: run inference on the
  Flask server, and use the ESP32-CAM only for image capture / streaming.

Usage:
    python training/convert_tflite.py
"""

import os
import json

import tensorflow as tf

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")

KERAS_PATH = os.path.join(MODEL_DIR, "plant_disease_model.keras")
TFLITE_PATH = os.path.join(MODEL_DIR, "plant_disease_model.tflite")


def main():
    if not os.path.exists(KERAS_PATH):
        print(f"[ERROR] Model not found: {KERAS_PATH}")
        return

    print("=" * 60)
    print("AI-Driven Precision Agrochemical Sprayer - TFLite Conversion")
    print("=" * 60)

    model = tf.keras.models.load_model(KERAS_PATH)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    # quantized (float16) is small; full int8 requires calibration dataset
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()

    with open(TFLITE_PATH, "wb") as f:
        f.write(tflite_model)
    size_bytes = os.path.getsize(TFLITE_PATH)
    print(f"[OK] TFLite model saved: {TFLITE_PATH}")
    print(f"[OK] Model size: {size_bytes / (1024 * 1024):.2f} MB")

    keras_bytes = os.path.getsize(KERAS_PATH)
    print(f"[OK] Keras model size: {keras_bytes / (1024 * 1024):.2f} MB")

    # ESP32 compatibility analysis
    esp32_ram_kb = 320  # approx usable SRAM for TFLite Micro
    report = {
        "keras_model": KERAS_PATH,
        "tflite_model": TFLITE_PATH,
        "tflite_size_bytes": size_bytes,
        "tflite_size_mb": round(size_bytes / (1024 * 1024), 2),
        "keras_size_mb": round(keras_bytes / (1024 * 1024), 2),
        "esp32_note": (
            "The ESP32/CAM typically has ~320 KB SRAM usable by "
            "TensorFlow Lite for Microcontrollers. A multi-crop foliar "
            "MobileNetV2-derived model (tens of MB) cannot fit on-device. "
            "Recommended architecture: ESP32-CAM captures and streams JPEG "
            "over Wi-Fi to the Flask server, which runs the REAL inference "
            "and returns the result. On-device inference is not claimed."
        ),
    }
    report_path = os.path.join(MODEL_DIR, "tflite_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"[OK] Compatibility report saved: {report_path}")
    print("\n" + report["esp32_note"])


if __name__ == "__main__":
    main()