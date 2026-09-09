"""
AI-Driven Precision Agrochemical Sprayer - Inference Pipeline

Crop-agnostic analysis of one uploaded image:

📸 Image
        │
        ▼
    🌿 STAGE 1: Plant / Leaf Validation      (plant_validator)
        │  "Not a Plant" → STOP, status "Not a Plant"
        │  below threshold → STOP, status "Low Plant Confidence"
        ▼
    🗺️ STAGE 2: Health Segmentation          (leaf_segmenter)
        │  pixel-wise U-Net mask:
        │    background / healthy leaf / symptomatic (diseased) leaf
        ▼
    📊 Affected Area % = symptom pixels / (healthy + symptom) pixels × 100
        ▼
    🔴 HEALTH STATUS   = 0 %    → Healthy
                       > 0 %    → Diseased
        ▼
    📏 SEVERITY (from the REAL visible affected area, configurable demo
        thresholds):   (0, 10%] Low  |  (10, 40%] Medium  |  >40% High
        ▼
    🚿 SPRAY DECISION SUPPORT: MONITOR / INSPECTION RECOMMENDED /
        TREATMENT ASSESSMENT RECOMMENDED (never an auto-pesticide choice)
        ▼
    🖼️ Overlay = highlight of the exact pixels the model marked symptomatic

The MAIN result is the HEALTH STATUS + real affected-area percentage +
severity + spray decision support. Plant/crop identification is NOT required
and is never used to decide health.

GOLDEN RULES:
  - Severity is computed ONLY from the real segmented visible affected area,
    never from "AI confidence" and never from random numbers.
  - If segmentation cannot reliably find a leaf region the pipeline honestly
    says "Analysis Not Available".
  - No specific pesticide is ever recommended without a valid crop/disease
    source.
"""

from PIL import Image

from .config import CONFIG

ANALYSIS_NOT_AVAILABLE = (
    "Unable to reliably determine the health condition of this plant with the "
    "currently supported AI models."
)


class AgroPipeline:
    def __init__(self):
        from .plant_validator import get_validator
        from .leaf_segmenter import get_segmenter

        self.validator = get_validator()
        self.segmenter = get_segmenter()

    # ------------------------------------------------------------------
    def predict_bytes(self, image_bytes):
        from io import BytesIO
        try:
            rgb = Image.open(BytesIO(image_bytes)).convert("RGB")
        except Exception as e:
            return {"success": False, "message": f"Invalid image: {e}"}
        return self.predict_image(rgb)

    # ------------------------------------------------------------------
    def predict_image(self, rgb):
        stages = []

        # ---------------- STAGE 1: plant/leaf validation -----------------
        v = self.validator.validate(rgb, combined_probs=None)
        stages.append({"stage": "Plant Validation", "result": v["result"],
                       "confidence": v["confidence"], "reason": v["reason"]})

        def stop(status, message, **extra):
            base = {
                "success": True,
                "status": status,
                "plant": None,
                "plant_confidence": None,
                "plant_validation_confidence": v["confidence"],
                "pipeline_stopped": True,
                "disease": None,
                "disease_confidence": None,
                "severity": None,
                "severity_available": False,
                "spray_decision": None,
                "message": message,
                "pipeline": stages,
            }
            base.update(extra)
            return base

        # Hard gate: an image that was not confidently verified as a plant/leaf
        # never reaches health segmentation.
        if v["result"] == "Not a Plant":
            return stop("Not a Plant",
                        "Please scan a clear image of a plant or leaf.")

        if v["confidence"] < CONFIG.plant_threshold:
            return stop("Low Plant Confidence",
                        "Unable to confidently verify that this image contains "
                        "a plant or leaf.")

        # ---------------- STAGE 2: health segmentation ---------------------
        seg = self.segmenter.analyze(rgb, with_overlay=True)
        stages.append({"stage": "Health Segmentation",
                       "result": seg["severity"] if seg["available"] else "Unavailable",
                       "affected_area_pct": seg.get("affected_area_pct"),
                       "segmentation_source": seg.get("segmentation_source"),
                       "note": seg["note"]})

        if not seg["available"]:
            return {
                "success": True,
                "status": "Analysis Not Available",
                "plant": None,
                "plant_confidence": None,
                "plant_validation_confidence": v["confidence"],
                "pipeline_stopped": False,
                "disease": None,
                "disease_confidence": None,
                "health_confidence": None,
                "severity": None,
                "severity_available": False,
                "affected_area_pct": None,
                "severity_thresholds": None,
                "severity_note": None,
                "overlay_data_uri": None,
                "spray_decision": None,
                "reason_blocked": seg.get("reason", "segmentation_unavailable"),
                "message": seg["note"],
                "pipeline": stages,
            }

        pct = seg["affected_area_pct"]
        healthy = pct <= 0.0
        status = "Healthy" if healthy else "Diseased"
        decision = seg["decision"]

        message = decision["message"] if not healthy else decision["message"]

        return {
            "success": True,
            "plant": None,
            "plant_confidence": None,
            "plant_validation_confidence": v["confidence"],
            "pipeline_stopped": False,
            "status": status,
            "disease": None,
            "disease_confidence": None,
            "health_confidence": None,
            "severity": seg["severity"],
            "severity_available": True,
            "affected_area_pct": pct,
            "severity_thresholds": seg["thresholds"],
            "severity_note": seg["note"],
            "overlay_data_uri": seg["overlay_data_uri"],
            "spray_decision": decision,
            "message": message,
            "pipeline": stages,
        }

    # ------------------------------------------------------------------
    def status(self):
        return {
            "service": "AI-Driven Precision Agrochemical Sprayer",
            "pipeline": "Plant/Leaf Detection -> Health Segmentation "
                        "(background / healthy / symptomatic) -> affected "
                        "area % -> severity -> spray decision support",
            "segmenter_model_loaded": self.segmenter.model is not None,
            "segmentation_source": ("model" if self.segmenter.model is not None
                                    else "pixel_fallback"),
            "seg_image_size": self.segmenter.img_size,
            "plant_threshold": CONFIG.plant_threshold,
            "severity_low_max": CONFIG.severity_low_max,
            "severity_med_max": CONFIG.severity_med_max,
            "severity_levels": {
                "HEALTHY": "0% affected area",
                "LOW": f"(0, {CONFIG.severity_low_max:g}%]",
                "MEDIUM": f"({CONFIG.severity_low_max:g}, "
                          f"{CONFIG.severity_med_max:g}%]",
                "HIGH": f"> {CONFIG.severity_med_max:g}%",
            },
            "spray_decision_map": {
                "HEALTHY / LOW": "MONITOR",
                "MEDIUM": "INSPECTION_RECOMMENDED",
                "HIGH": "TREATMENT_ASSESSMENT_RECOMMENDED",
            },
            "thresholds_note": "Demonstration severity thresholds - calibrate "
                               "them for real production agricultural rules.",
        }


_pipeline = None


def get_pipeline():
    global _pipeline
    if _pipeline is None:
        _pipeline = AgroPipeline()
    return _pipeline