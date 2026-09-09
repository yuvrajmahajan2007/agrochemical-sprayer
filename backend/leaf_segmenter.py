"""
AI-Driven Precision Agrochemical Sprayer - Real Leaf / Health Segmentation

Core of the redesigned, crop-agnostic pipeline:

    Plant/Leaf detected  →  segment healthy leaf area  →  segment visibly
    symptomatic / diseased area  →  affected area %  →  severity  →  spray
    decision support.

Everything below comes from REAL model masks (trained U-Net) when the model
is present:

    affected_area_pct =
        symptomatic_pixels / (healthy_pixels + symptomatic_pixels) * 100

Severity is derived ONLY from that visible affected-area percentage - never
from "AI confidence" and never from random numbers:

    0%            HEALTHY
    (0, 10%]      LOW
    (10, 40%]     MEDIUM
    > 40%         HIGH

If no trained segmenter exists yet, a deterministic pixel-segmentation
fallback is used (real, explicable, labelled as fallback) so the demo still
works - the result never becomes a made-up number.

Overlay highlights ONLY the pixels the model actually marked as symptomatic.
"""

import base64
import io
import os

import numpy as np
from PIL import Image

from .config import CONFIG


class LeafSegmenter:
    def __init__(self, model_path=None):
        path = model_path or CONFIG.leaf_segmenter_path
        self.model = None
        self.img_size = int(CONFIG.seg_image_size)
        if os.path.exists(path):
            import tensorflow as tf
            print(f"[LeafSegmenter] Loading segmentation model: {path}")
            self.model = tf.keras.models.load_model(path)
            cfg = path.replace("leaf_segmenter.keras", "infer_config.json")
            if os.path.exists(cfg):
                import json
                d = json.load(open(cfg, encoding="utf-8"))
                self.img_size = int(d.get("image_size", self.img_size))
        else:
            print("[LeafSegmenter] No model on disk - pixel-segmentation fallback.")

    # ------------------------------------------------------------------
    @staticmethod
    def _pixel_mask(rgb):
        """Deterministic fallback labels: 0 bg, 1 healthy, 2 symptomatic."""
        arr = np.asarray(rgb.convert("RGB"), dtype=np.float32)
        hsv = np.asarray(rgb.convert("HSV"), dtype=np.float32)
        h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
        sky = (v > 200) & (s < 40)
        green = (s >= 40) & (h <= 100) & (h >= 25)
        lesion = ((h >= 15) & (h <= 40) & (s >= 60)) | \
                 ((h > 100) & (h <= 170) & (s >= 30))
        dark_leaf = (s < 60) & (v > 40) & (v < 200) & (h <= 115)
        foreground = (green | lesion | dark_leaf) & ~sky
        healthy = green & (s >= 60) & (v >= 55) & ~sky
        symptom = foreground & (
            ((h >= 15) & (h <= 60) & (s >= 40) & (v <= 150)) |
            ((h > 100) & (h <= 170) & (s >= 30)) |
            ((h >= 25) & (h <= 45) & (s >= 20) & (s < 60) & (v > 60) & (v <= 180))
        )
        m = np.zeros(arr.shape[:2], dtype=np.int32)
        m[healthy] = 1
        m[symptom] = 2
        return m

    # ------------------------------------------------------------------
    def _model_mask(self, rgb):
        """Real model output: 3-class mask upsampled to the original size."""
        if self.model is None:
            return None
        orig = np.asarray(rgb.convert("RGB"), dtype=np.uint8)
        small = np.asarray(rgb.resize((self.img_size, self.img_size),
                                      Image.LANCZOS), dtype=np.float32)
        pred = self.model.predict(small[np.newaxis, ...] / 255.0, verbose=0)[0]
        sm = pred[..., 0] + pred[..., 1]  # background excluded later
        argmax = np.argmax(pred, axis=-1).astype(np.uint8)  # (s, s)
        img = Image.fromarray(argmax * 85)   # scale 0/1/2 -> visible greys
        scale = (orig.shape[1] / self.img_size, orig.shape[0] / self.img_size)
        img = img.resize((orig.shape[1], orig.shape[0]), Image.LANCZOS)
        full = np.asarray(img, dtype=np.float32) / 85.0
        mask = np.clip(np.round(full), 0, 2).astype(np.int32)
        return mask, sm

    # ------------------------------------------------------------------
    def analyze(self, image_rgb, with_overlay=False):
        arr = np.asarray(image_rgb.convert("RGB"), dtype=np.uint8)
        if arr.size == 0:
            return self._unavailable("empty_image")

        source = "model"
        got = self._model_mask(image_rgb)
        if got is None:
            mask = self._pixel_mask(image_rgb)
            source = "pixel_fallback"
        else:
            mask, _ = got

        healthy = (mask == 1)
        symptom = (mask == 2)
        leaf = healthy | symptom
        leaf_px = int(leaf.sum())

        if leaf_px < max(900, arr.shape[0] * arr.shape[1] * 0.02):
            return self._unavailable("foreground_too_small")

        affected_px = int(symptom.sum())
        pct = affected_px / leaf_px * 100.0

        if pct <= CONFIG.severity_low_max:
            level = "Low"
        elif pct <= CONFIG.severity_med_max:
            level = "Medium"
        else:
            level = "High"

        decision = self._spray_decision(pct, level)

        overlay = None
        if with_overlay:
            overlay = self._compose_overlay(image_rgb, symptom)

        note = ("Affected area measured from real leaf segmentation "
                f"({source}); demonstration thresholds, calibrate for "
                "production agricultural rules.")

        return {
            "available": True,
            "severity": level,
            "affected_area_pct": round(pct, 1),
            "affected_pixels": affected_px,
            "leaf_pixels": leaf_px,
            "segmentation_source": source,
            "thresholds": {
                "healthy_upto": 0.0,
                "low_max": CONFIG.severity_low_max,
                "med_max": CONFIG.severity_med_max,
            },
            "decision": decision,
            "note": note,
            "overlay_data_uri": overlay,
        }

    # ------------------------------------------------------------------
    @staticmethod
    def _spray_decision(pct, level):
        def msg_for(key):
            return getattr(CONFIG, "decision_" + key,
                           CONFIG.default_decision_messages[key])

        if pct <= 0:
            return {"code": "MONITOR",
                    "label": "MONITOR",
                    "message": msg_for("healthy")}
        if level == "Low":
            return {"code": "MONITOR",
                    "label": "MONITOR",
                    "message": msg_for("low")}
        if level == "Medium":
            return {"code": "INSPECTION_RECOMMENDED",
                    "label": "INSPECTION RECOMMENDED",
                    "message": msg_for("medium")}
        return {"code": "TREATMENT_ASSESSMENT_RECOMMENDED",
                "label": "TREATMENT ASSESSMENT RECOMMENDED",
                "message": msg_for("high")}

    # ------------------------------------------------------------------
    @staticmethod
    def _unavailable(reason):
        return {
            "available": False,
            "severity": None,
            "affected_area_pct": None,
            "thresholds": None,
            "decision": None,
            "note": "Severity analysis unavailable",
            "reason": reason,
            "overlay_data_uri": None,
        }

    MAX_OVERLAY_DIM = 400

    @classmethod
    def _compose_overlay(cls, image_rgb, symptom):
        """Highlight ONLY detected symptomatic pixels (red) on the image."""
        a = np.asarray(image_rgb.convert("RGB"), dtype=np.float32)
        m = symptom[..., None].astype(np.float32)
        blend = a * (1.0 - 0.40 * m)
        blend[..., 0] += (255.0 - a[..., 0]) * 0.40 * m[..., 0]
        out = Image.fromarray(np.clip(blend, 0, 255).astype(np.uint8))
        if max(out.size) > cls.MAX_OVERLAY_DIM:
            out.thumbnail((cls.MAX_OVERLAY_DIM, cls.MAX_OVERLAY_DIM),
                          Image.LANCZOS)
        buf = io.BytesIO()
        out.save(buf, "PNG", optimize=True)
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        return f"data:image/png;base64,{b64}"


_segmenter = None


def get_segmenter():
    global _segmenter
    if _segmenter is None:
        _segmenter = LeafSegmenter()
    return _segmenter