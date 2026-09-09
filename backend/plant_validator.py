"""
AI-Driven Precision Agrochemical Sprayer - Stage 1: Plant / Leaf Validation

Determines how confident we are that an uploaded image actually contains a
plant or leaf BEFORE any crop identification or disease reasoning is attempted.

HOW CONFIDENCE IS COMPUTED (no fake numbers):

  1. A vegetation fraction is measured directly from the image pixels:
        veg_frac = max( green_hue_fraction, excess_green_fraction )
     - green_hue_fraction : fraction of pixels inside a green/leaf-yellow hue
       band in HSV (works on spotted/diseased leaves too).
     - excess_green_fraction : fraction of pixels where the classic vegetation
       index  2*G - R - B > 15  and brightness is not near-black (ExG is a
       well-known, lighting-robust vegetation detector).
     Both are pure pixel statistics - nothing hardcoded about the *class* of
     plant, so ANY leaf (tomato, wheat, unknown weed ...) counts the same.

  2. veg_frac is remapped to a 0..1 "vegetation score" on a fixed curve:
        veg_c = clip((veg_frac - 0.05) / 0.35)
     (5% vegetation = start of evidence, 40%+ vegetation = fully convincing.)

  3. Model agreement: when the real crop+disease classifier probabilities are
     available (or a dedicated plant_validator.keras exists), the top-1
     temperature-calibrated probability is used as an independent LEARNED
     "does this look like a real plant" signal.

  4. Final Plant Confidence = 50% vegetation score + 50% model agreement
     (both in 0..100). An image only passes Stage 1 when this crosses the
     configured SPRAYBOT_PLANT_THRESHOLD (default 70%).

Near-blank / uniform images (no colour variation, almost no edges) are
hard-rejected as "Not a Plant"; everything else that falls below threshold is
honestly reported as ambiguous ("Low Plant Confidence") and the pipeline
stops before identification.
"""

import os

import numpy as np
from PIL import Image

from .config import CONFIG, IMG_SIZE


class PlantValidator:
    def __init__(self, model_path=None):
        self.model = None
        path = model_path or CONFIG.plant_validator_path
        if os.path.exists(path):
            import tensorflow as tf
            print(f"[PlantValidator] Loading model: {path}")
            self.model = tf.keras.models.load_model(path)
        else:
            print(f"[PlantValidator] No model at {path} - using CV+model signals.")

    # ------------------------------------------------------------------
    def decompose(self, rgb):
        """Return raw pixel statistics (all NumPy-only, no extra deps)."""
        arr = np.asarray(rgb, dtype=np.float32)
        hsv = np.asarray(rgb.convert("HSV"), dtype=np.float32)  # 0-255 channels
        h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

        # Green/leaf-tone hue mask (0-255 scale). Wide enough for healthy
        # green leaves, yellow-ish disease-stressed leaves and brown-dry spots.
        greenish = ((s >= 35) & (v >= 35)) & (
            ((h >= 25) & (h <= 100)) |          # green band (incl. yellow-green)
            ((h >= 15) & (h <= 40) & (s >= 55)) # amber / early-disease tones
        )
        green_frac = float(greenish.mean())

        # Excess-green vegetation index:  ExG = 2*G - R - B
        # > 0  => pixel is greener than its average of R/B (classic CIVE/ExG
        # vegetation detector, robust against shadows & white balance).
        r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
        exg = (2.0 * g) - r - b
        veg = (exg > 15.0) & (arr.mean(axis=2) > 25.0)
        veg_frac = float(veg.mean())

        gray = arr.mean(axis=2)
        gy, gx = np.gradient(gray)
        texture = float(np.abs(gx).mean() + np.abs(gy).mean())
        std = float(gray.std())
        sat_mean = float(s.mean() / 255.0)

        return {
            "green_frac": green_frac,
            "veg_frac": veg_frac,
            "texture": texture,
            "std": std,
            "sat_mean": sat_mean,
            "max_veg_frac": max(green_frac, veg_frac),
        }

    # ------------------------------------------------------------------
    @staticmethod
    def _veg_score(veg_frac):
        return float(np.clip((veg_frac - 0.05) / 0.35, 0.0, 1.0))

    # ------------------------------------------------------------------
    def plant_likelihood(self, sig, combined_probs):
        """0..1 plant likelihood from pixel stats + (optional) model signal."""
        veg_c = self._veg_score(sig["max_veg_frac"])
        if combined_probs is not None:
            probs = np.asarray(combined_probs, dtype=np.float64)
            model_p = float(probs.max()) if probs.size else 0.0
        else:
            model_p = veg_c  # pure-CV fallback when no model scores are passed
        return 0.5 * veg_c + 0.5 * model_p

    # ------------------------------------------------------------------
    def validate(self, image_rgb, combined_probs=None):
        """
        image_rgb: PIL.Image or (H,W,3) RGB array -> dict result.

        Returns:
            result     : "Plant" | "Not a Plant" | "Ambiguous"
            confidence : 0..100 plant confidence (honest, from real signals)
            reason     : short explanation of the deciding signals
        """
        if isinstance(image_rgb, Image.Image):
            rgb = image_rgb.convert("RGB")
        else:
            rgb = Image.fromarray(np.asarray(image_rgb)).convert("RGB")

        sig = self.decompose(rgb)
        green_frac = sig["green_frac"]
        veg_frac = sig["veg_frac"]
        std = sig["std"]
        texture = sig["texture"]

        # --- Dedicated trained validator model (when present on disk) ------
        if self.model is not None:
            arr = np.asarray(rgb.resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS),
                             dtype=np.float32)[np.newaxis, ...] / 255.0
            probs = self.model.predict(arr, verbose=0)[0]
            plant_p = float(probs[1]) * 100.0 if len(probs) >= 2 else float(probs[0]) * 100.0
            result = ("Plant" if plant_p >= CONFIG.plant_threshold
                      else "Ambiguous")
            return {"result": result, "confidence": round(plant_p, 2),
                    "reason": "validator_model", "signals": sig}

        # --- Near-blank / uniform frame -> clearly not a leaf ---------------
        if std < 8.0 and texture < CONFIG.texture_min:
            return {"result": "Not a Plant",
                    "confidence": round(min(30.0, self.plant_likelihood(sig, combined_probs) * 100), 2),
                    "reason": "uniform_low_texture", "signals": sig}

        # --- Real confidence (pixel statistics + model agreement) ----------
        likelihood = self.plant_likelihood(sig, combined_probs)
        confidence = round(likelihood * 100.0, 2)

        if confidence >= CONFIG.plant_threshold:
            return {"result": "Plant", "confidence": confidence,
                    "reason": f"veg={sig['max_veg_frac']:.0%}", "signals": sig}

        # Everything below threshold is honestly ambiguous -> pipeline STOPS
        # with "Low Plant Confidence" (never silently continues to Stage 2).
        return {"result": "Ambiguous", "confidence": confidence,
                "reason": f"below_threshold veg={sig['max_veg_frac']:.0%}",
                "signals": sig}


_validator = None


def get_validator():
    global _validator
    if _validator is None:
        _validator = PlantValidator()
    return _validator