"""
AI-Driven Precision Agrochemical Sprayer - Stage 2: Plant / Crop Identification

Identifies WHICH plant or crop is in the image (independent task from disease
detection).

Preference order:
  1. Dedicated model models/plant_identifier/plant_identifier.keras
     (trained by training/train_plant_identifier.py on real labeled data).
  2. Fallback: the real 41-class combined model aggregated per crop.

Never guesses a plant outside the supported identity space - if top confidence
is below the configured threshold the result is "Unknown Plant".
"""

import os
import json

import numpy as np
import tensorflow as tf
from PIL import Image

from .config import CONFIG, IMG_SIZE


class PlantIdentifier:
    def __init__(self, model_path=None, classes_path=None):
        self.model = None
        self.class_names = []
        path = model_path or CONFIG.plant_identifier_path
        cp = classes_path or CONFIG.plant_identifier_classes_path
        if os.path.exists(path) and os.path.exists(cp):
            print(f"[PlantIdentifier] Loading {path}")
            self.model = tf.keras.models.load_model(path)
            with open(cp, "r", encoding="utf-8") as f:
                self.class_names = json.load(f)
        else:
            print("[PlantIdentifier] No dedicated model - using combined-model "
                  "crop aggregation fallback.")

    # ------------------------------------------------------------------
    def _dedicated_identify(self, image_rgb):
        arr = np.asarray(image_rgb.convert("RGB").resize((IMG_SIZE, IMG_SIZE),
                                                         Image.LANCZOS),
                         dtype=np.float32)[np.newaxis, ...] / 255.0
        probs = self.model.predict(arr, verbose=0)[0]
        return probs

    # ------------------------------------------------------------------
    def identify(self, image_rgb, combined_probs=None):
        """
        Returns dict:
            plant        : top plant name (or "Unknown")
            confidence   : top percentage
            threshold    : configured plant identification threshold
            top_plants   : [{plant, confidence}] (debug)
        """
        threshold = CONFIG.identifier_threshold

        if self.model is not None:
            probs = self._dedicated_identify(image_rgb)
            labels = self.class_names if self.class_names else \
                [f"Crop{i}" for i in range(len(probs))]
        else:
            if combined_probs is None:
                from .model_core import get_combined
                combined_probs = get_combined().predict_probs(image_rgb)
            # Aggregate per-crop probabilities (real aggregation of the real model).
            from .model_core import get_combined
            combined = get_combined()
            crop_p = {}
            for i in range(combined.num_classes):
                crop = combined.class_info[str(i)]["crop"]
                crop_p[crop] = crop_p.get(crop, 0.0) + float(combined_probs[i])
            labels = list(crop_p.keys())
            probs = np.array([crop_p[c] for c in labels])

        order = np.argsort(probs)[::-1]
        top_plant = labels[int(order[0])]
        top_conf = float(probs[int(order[0])]) * 100.0

        top_plants = [{"plant": labels[int(o)],
                       "confidence": round(float(probs[int(o)]) * 100.0, 2)}
                      for o in order[:3]]

        if top_conf < threshold:
            return {
                "plant": "Unknown",
                "confidence": round(top_conf, 2),
                "threshold": threshold,
                "top_plants": top_plants,
                "raw_top_plant": top_plant,
            }

        return {
            "plant": top_plant,
            "confidence": round(top_conf, 2),
            "threshold": threshold,
            "top_plants": top_plants,
            "raw_top_plant": top_plant,
        }


_identifier = None


def get_identifier():
    global _identifier
    if _identifier is None:
        _identifier = PlantIdentifier()
    return _identifier