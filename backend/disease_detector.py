"""
AI-Driven Precision Agrochemical Sprayer - Stage 3: Crop-Specific Disease Detection

Runs ONLY after plant validation and crop identification succeed AND the crop
is listed as disease-supported in models/supported_crops.json.

Preference order:
  1. Dedicated per-crop model models/disease_models/<crop>_model.keras
     (trained by training/train_disease_models.py).
  2. Fallback: the real 41-class combined model RESTRICTED to the identified
     crop's own classes and renormalised - a genuine, crop-specific read from
     the same real trained model. An image of crop X is never evaluated
     against the disease classes of crop Y.

THRESHOLD RULE:
- Below the disease threshold -> "Unknown" (never forced to Healthy/Diseased).
- The crop is healthy only when the restricted top class is the Healthy class.
"""

import os
import json

import numpy as np
import tensorflow as tf
from PIL import Image

from .config import CONFIG, IMG_SIZE


class DiseaseDetector:
    def __init__(self, combined_model, disease_models_dir=None):
        self.combined = combined_model
        self.dir = disease_models_dir or CONFIG.disease_models_dir
        self.cache = {}

    # ------------------------------------------------------------------
    def _load_crop_model(self, crop):
        if crop in self.cache:
            return self.cache[crop]
        # safe key for file system
        safe = crop.replace("/", "_").replace(" ", "_")
        model_path = os.path.join(self.dir, f"{safe}_model.keras")
        class_path = os.path.join(self.dir, f"{safe}_classes.json")
        if os.path.exists(model_path) and os.path.exists(class_path):
            print(f"[DiseaseDetector] Crop model: {model_path}")
            m = tf.keras.models.load_model(model_path)
            cls = json.load(open(class_path, "r", encoding="utf-8"))
        else:
            m, cls = None, None
        self.cache[crop] = (m, cls)
        return m, cls

    # ------------------------------------------------------------------
    def crop_classes(self, crop):
        """[(crop_class_display, status, is_healthy)] restricted to crop."""
        return self.combined.crop_classes.get(crop, [])

    def crop_supported(self, crop):
        """True when a per-crop model exists OR the combined model covers crop."""
        m, cls = self._load_crop_model(crop)
        if m is not None:
            return True
        return len(self.crop_classes(crop)) > 0

    # ------------------------------------------------------------------
    def detect(self, crop, image_rgb, combined_probs=None):
        """
        Returns dict:
            status             : "Healthy" | "Diseased" | "Unknown"
            disease            : name or None
            confidence         : percentage
            threshold          : configured
            available          : bool
            model_source       : "crop_model" | "combined_restricted" | None
            top_classes        : debug list
        """
        threshold = CONFIG.disease_threshold
        m, cls = self._load_crop_model(crop)

        if m is not None:
            arr = np.asarray(image_rgb.convert("RGB").resize((IMG_SIZE, IMG_SIZE),
                                                             Image.LANCZOS),
                             dtype=np.float32)[np.newaxis, ...] / 255.0
            probs = self.model_probs = self._predict_crop(m, arr)
            labels = cls
            source = "crop_model"
        else:
            class_list = self.crop_classes(crop)
            if not class_list:
                return {
                    "status": "Unknown", "disease": None, "confidence": 0.0,
                    "threshold": threshold, "available": False,
                    "model_source": None, "top_classes": [],
                }
            if combined_probs is None:
                combined_probs = self.combined.predict_probs(image_rgb)
            # Restrict real model output to THIS crop's classes only.
            idxs = [i for i, _ in class_list]
            sub = combined_probs[idxs]
            sub = sub / sub.sum()            # renormalise within the crop
            labels = [info["display_name"] for _, info in class_list]
            is_healthy = [info["status"] == "Healthy" for _, info in class_list]
            diseases = [info.get("disease") for _, info in class_list]
            probs = sub
            source = "combined_restricted"

        top = int(np.argmax(probs))
        conf = float(probs[top]) * 100.0
        display = labels[top]

        if conf < threshold:
            return {
                "status": "Unknown", "disease": None, "confidence": round(conf, 2),
                "threshold": threshold, "available": True,
                "model_source": source,
                "top_classes": self._top_classes(probs, labels),
            }

        if source == "crop_model":
            healthy = ("healthy" in display.lower() or "heal" in display.lower())
            disease = None if healthy else display
            status = "Healthy" if healthy else "Diseased"
        else:
            healthy = is_healthy[top]
            status = "Healthy" if healthy else "Diseased"
            disease = None if healthy else diseases[top]

        return {
            "status": status, "disease": disease, "confidence": round(conf, 2),
            "threshold": threshold, "available": True,
            "model_source": source,
            "top_classes": self._top_classes(probs, labels),
        }

    # ------------------------------------------------------------------
    @staticmethod
    def _predict_crop(m, arr):
        return m.predict(arr, verbose=0)[0]

    @staticmethod
    def _top_classes(probs, labels, n=3):
        order = np.argsort(probs)[::-1][:n]
        return [{"class": labels[int(o)], "confidence": round(float(probs[int(o)]) * 100, 2)}
                for o in order]


_detector = None


def get_detector(combined_model=None):
    global _detector
    if _detector is None:
        from .model_core import get_combined
        _detector = DiseaseDetector(combined_model or get_combined())
    return _detector