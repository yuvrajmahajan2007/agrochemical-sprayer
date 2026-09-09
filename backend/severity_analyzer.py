"""
AI-Driven Precision Agrochemical Sprayer - Severity Analysis

Estimates the percentage of the visible leaf area that shows disease/necrosis
using explainable image analysis (not random values):

  1. Segment foreground "leaf-ish" pixels (green vegetation tones OR typical
     plant disease tones - yellow / brown / orange lesions).
  2. Within that foreground, segment damaged/necrotic pixels (low-saturation
     yellow-to-brown, distinct from healthy saturated green).
  3. affected_area_pct = (damaged / leaf-ish) * 100

Thresholds (configurable, defaults):
    0-10%    -> Low
    10-40%   -> Medium
    > 40%    -> High

If the segmentation is unstable (too little foreground etc.) the analyzer
returns "Severity analysis unavailable" - it NEVER invents a value.
"""

import base64
import io

import numpy as np
from PIL import Image

from .config import CONFIG


class SeverityAnalyzer:
    LEVELS = ("Low", "Medium", "High")

    def analyze(self, image_rgb, with_overlay=False):
        arr = np.asarray(image_rgb.convert("RGB"), dtype=np.float32)
        if arr.size == 0:
            return self._unavailable("empty_image")

        hsv = np.asarray(image_rgb.convert("HSV"), dtype=np.float32)
        h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

        # Healthy leaf green + disease-affected yellow/brown tones that remain
        # inside a plant, but exclude bright sky (high V, low S).
        sky = (v > 200) & (s < 40)
        leaf_green = (s >= 40) & (h <= 100) & (h >= 25)
        lesion_tone = ((h >= 15) & (h <= 40) & (s >= 60)) | \
                      ((h > 100) & (h <= 170) & (s >= 30))  # browns/olives
        dark_leaf = (s < 60) & (v > 40) & (v < 200) & (h <= 110)

        foreground = (leaf_green | lesion_tone | dark_leaf) & ~sky
        fg_count = int(foreground.sum())

        if fg_count < max(900, arr.shape[0] * arr.shape[1] * 0.02):
            return self._unavailable("foreground_too_small")

        # Healthy tissue = high-saturation green; damaged = yellow/brown/dull.
        healthy = leaf_green & (s >= 70) & (v >= 60)
        damaged = ((h >= 15) & (h <= 60) & (s >= 40) & (v <= 120)) | \
                  ((h >= 25) & (h <= 45) & (s >= 20) & (v > 120))

        total_tissue = float((healthy | damaged).sum())
        if total_tissue < max(400, fg_count * 0.15):
            return self._unavailable("tissue_segmentation_unstable")

        affected = float(damaged.sum())
        pct = affected / total_tissue * 100.0

        if pct <= CONFIG.severity_low_max:
            level = "Low"
        elif pct <= CONFIG.severity_med_max:
            level = "Medium"
        else:
            level = "High"

        overlay = None
        if with_overlay:
            overlay = self._compose_overlay(image_rgb, damaged)

        return {
            "available": True,
            "severity": level,
            "affected_area_pct": round(pct, 1),
            "thresholds": {
                "low_max": CONFIG.severity_low_max,
                "med_max": CONFIG.severity_med_max,
            },
            "note": "Estimated from visible leaf-area segmentation.",
            "overlay_data_uri": overlay,
        }

    @staticmethod
    def _unavailable(reason):
        return {
            "available": False,
            "severity": None,
            "affected_area_pct": None,
            "thresholds": None,
            "note": "Severity analysis unavailable",
            "reason": reason,
            "overlay_data_uri": None,
        }

    # ------------------------------------------------------------------
    # Real segmentation overlay: red highlights over the actual damaged
    # pixels found by the pixel-level analysis above. Never fake/invented.
    # ------------------------------------------------------------------
    MAX_OVERLAY_DIM = 400

    @classmethod
    def _compose_overlay(cls, image_rgb, damaged):
        img = image_rgb.convert("RGB")
        a = np.asarray(img, dtype=np.float32)
        m = damaged[..., None].astype(np.float32)
        blend = a * (1.0 - 0.45 * m)
        blend[..., 0] += (255.0 - a[..., 0]) * 0.45 * m[..., 0]
        out = Image.fromarray(np.clip(blend, 0, 255).astype(np.uint8))
        if max(out.size) > cls.MAX_OVERLAY_DIM:
            out.thumbnail((cls.MAX_OVERLAY_DIM, cls.MAX_OVERLAY_DIM), Image.LANCZOS)
        buf = io.BytesIO()
        out.save(buf, "PNG", optimize=True)
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        return f"data:image/png;base64,{b64}"


_analyzer = None


def get_severity_analyzer():
    global _analyzer
    if _analyzer is None:
        _analyzer = SeverityAnalyzer()
    return _analyzer