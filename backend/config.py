"""
AI-Driven Precision Agrochemical Sprayer - Backend Configuration

Central runtime configuration. Everything is overridable via environment
variables so the same system can be tuned per deployment / per phone demo.
"""

import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")

IMG_SIZE = 224


def _f(name, default):
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return float(default)


def _i(name, default):
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return int(default)


class Config:
    # ---- Paths ------------------------------------------------------------
    model_dir = MODEL_DIR
    supported_crops_path = os.path.join(MODEL_DIR, "supported_crops.json")
    crop_metadata_path = os.path.join(MODEL_DIR, "crop_metadata.json")
    class_names_path = os.path.join(MODEL_DIR, "class_names.json")
    combined_model_path = os.path.join(MODEL_DIR, "plant_disease_model.keras")
    plant_identifier_path = os.path.join(MODEL_DIR, "plant_identifier", "plant_identifier.keras")
    plant_identifier_classes_path = os.path.join(MODEL_DIR, "plant_identifier", "class_names.json")
    plant_validator_path = os.path.join(MODEL_DIR, "plant_validator", "plant_validator.keras")
    disease_models_dir = os.path.join(MODEL_DIR, "disease_models")

    # ---- Confidence gating --------------------------------------------------
    # Stage 1 plant/leaf validation minimum confidence (%).
    # Below this the pipeline STOPS and reports "Low Plant Confidence" - it
    # never proceeds to plant identification or disease detection.
    plant_threshold = _f("SPRAYBOT_PLANT_THRESHOLD", 70.0)
    # Stage 2 crop-identification minimum confidence (%). If the highest crop
    # probability is below this the result is "Unknown Plant" (never a guess).
    identifier_threshold = _f("SPRAYBOT_IDENTIFIER_THRESHOLD", 70.0)
    # Stage 3 disease minimum confidence (%).
    disease_threshold = _f("SPRAYBOT_DISEASE_THRESHOLD", 60.0)
    # Combined-model temperature calibration (< 2.0 = less overconfidence).
    temperature = _f("SPRAYBOT_TEMPERATURE", 2.0)

    # ---- Stage 1 plant/leaf CV validation ------------------------------------
    # Minimum fraction of green/leaf-coloured pixels to consider "plant-like".
    green_min = _f("SPRAYBOT_GREEN_MIN", 0.05)
    # If non-green and texture is very low, the image is almost certainly not a leaf.
    texture_min = _f("SPRAYBOT_TEXTURE_MIN", 1.5)

    # ---- Severity analysis ----------------------------------------------------
    # Affected-area percentage thresholds -> Low / Medium / High.
    severity_low_max = _f("SPRAYBOT_SEVERITY_LOW", 10.0)
    severity_med_max = _f("SPRAYBOT_SEVERITY_MED", 40.0)

    # ---- Leaf/health segmentation ---------------------------------------------
    # Real U-Net that segments background / healthy leaf / symptomatic leaf.
    leaf_segmenter_path = os.path.join(
        MODEL_DIR, "leaf_segmenter", "leaf_segmenter.keras")
    seg_image_size = _i("SPRAYBOT_SEG_IMG_SIZE", 128)

    # ---- Spray decision support messages ----------------------------------------
    # Demonstration wording. Replace when hooking the real agricultural rules.
    default_decision_messages = {
        "healthy": "No disease symptoms detected. No spraying recommendation.",
        "low": "Monitor the plant and recheck later.",
        "medium": "Inspection recommended. Spray decision "
                  "must follow configured agricultural rules.",
        "high": "High visible disease symptoms. Immediate agricultural "
                "inspection / treatment assessment recommended.",
    }


CONFIG = Config()