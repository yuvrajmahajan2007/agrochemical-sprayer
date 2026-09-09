"""
AI-Driven Precision Agrochemical Sprayer - Shared model core

Loads the combined crop+disease Keras model ONCE (it is also used as a
fallback source for crop identification and crop-restricted disease
detection) and provides temperature-calibrated probabilities.

Temperature calibration (softmax(logits / T), T default 2.0) reduces the
overconfidence the raw model shows on images that are NOT real plant leaves,
so out-of-distribution images fall below configured thresholds and return
Unknown instead of a forced disease.
"""

import os
import json

import numpy as np
import tensorflow as tf
from PIL import Image

from .config import CONFIG, IMG_SIZE


class CombinedModel:
    """Singleton wrapper around the real 41-class crop+disease model."""

    def __init__(self, model_path=None, class_names_path=None):
        keras_path = model_path or CONFIG.combined_model_path
        if not os.path.exists(keras_path):
            raise FileNotFoundError(f"Combined model not found: {keras_path}")
        print(f"[CombinedModel] Loading {keras_path}")
        self.model = tf.keras.models.load_model(keras_path)

        cp = class_names_path or CONFIG.class_names_path
        with open(cp, "r", encoding="utf-8") as f:
            self.class_info = json.load(f)
        self.num_classes = len(self.class_info)

        # crop -> list of (class_index, info)
        self.crop_classes = {}
        self.crops = []
        for i in range(self.num_classes):
            info = self.class_info[str(i)]
            crop = info.get("crop", "Unknown")
            self.crop_classes.setdefault(crop, []).append((i, info))
            if crop not in self.crops:
                self.crops.append(crop)
        print(f"[CombinedModel] {self.num_classes} classes, "
              f"{len(self.crops)} crops: {', '.join(self.crops)}")

    # ------------------------------------------------------------------
    @staticmethod
    def temperature_scale(probs, temperature):
        """Softmax(log(p)/T) - standard neural-net calibration."""
        if abs(temperature - 1.0) <= 1e-6:
            return probs
        logits = np.log(np.clip(probs, 1e-12, 1.0)) / float(temperature)
        logits = logits - np.max(logits)
        ex = np.exp(logits)
        return ex / np.sum(ex)

    # ------------------------------------------------------------------
    def predict_probs(self, image_rgb, temperature=None):
        """image_rgb: PIL.Image -> temperature-calibrated 41-class probs."""
        if isinstance(image_rgb, Image.Image):
            rgb = image_rgb.convert("RGB")
        else:
            rgb = Image.fromarray(np.asarray(image_rgb)).convert("RGB")
        arr = np.asarray(rgb.resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS),
                         dtype=np.float32)[np.newaxis, ...] / 255.0
        raw = self.model.predict(arr, verbose=0)[0]
        return self.temperature_scale(raw, temperature if temperature is not None
                                      else CONFIG.temperature)


_model = None


def get_combined():
    global _model
    if _model is None:
        _model = CombinedModel()
    return _model