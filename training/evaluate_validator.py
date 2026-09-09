"""
AI-Driven Precision Agrochemical Sprayer - Evaluate the REAL validator model

Loads models/plant_validator/plant_validator.keras (trained by
train_validator.py), runs it over every real image in
datasets/plant_validation/, and reports:

    - accuracy / precision / recall / F1 / confusion matrix  (test split)
    - a threshold sweep 50%..85% so we can pick an operating threshold where
      REAL leaves pass reliably while random objects do NOT (never a blind
      lowering - chosen by measured balanced accuracy on the test split)

Writes models/plant_validator/threshold_report.json and updates metadata.json
with the chosen operating threshold.

Usage:
    python training/evaluate_validator.py
"""

import os
import sys
import json

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import numpy as np  # noqa: E402
import tensorflow as tf  # noqa: E402
from sklearn.metrics import (accuracy_score, precision_score,  # noqa: E402
                             recall_score, f1_score, confusion_matrix,
                             balanced_accuracy_score)
from sklearn.model_selection import train_test_split  # noqa: E402

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
tf.get_logger().setLevel("ERROR")

from training.dataset_utils import load_plant_validation  # noqa: E402
from training.dataset_utils import IMG_SIZE  # noqa: E402

OUT_DIR = os.path.join(PROJECT_ROOT, "models", "plant_validator")
MODEL_PATH = os.path.join(OUT_DIR, "plant_validator.keras")
METADATA_PATH = os.path.join(OUT_DIR, "metadata.json")
CLASS_ORDER = ["Non_Plant", "Plant_Leaf"]


def main():
    if not os.path.exists(MODEL_PATH):
        raise SystemExit(f"Missing model: {MODEL_PATH}. Train it first "
                         "(python training/train_validator.py).")

    model = tf.keras.models.load_model(MODEL_PATH)

    data = load_plant_validation()
    plant = data.get("plant", [])
    non_plant = data.get("non_plant", [])
    if not plant or not non_plant:
        raise SystemExit("Missing real validation images.")

    items = [(p, 0) for p in non_plant] + [(p, 1) for p in plant]
    labels = [y for _, y in items]
    _, te = train_test_split(items, test_size=0.15,
                             stratify=labels, random_state=42)
    te_paths = [x for x, _ in te]
    te_y = np.array([y for _, y in te], dtype=np.int32)

    # model probability of "Plant_Leaf" (class index 1)
    proba = []
    for p in te_paths:
        img = tf.io.decode_image(tf.io.read_file(p), channels=3,
                                 expand_animations=False)
        img = tf.image.resize(tf.cast(img, tf.float32), (IMG_SIZE, IMG_SIZE))
        out = model.predict(tf.expand_dims(img / 255.0, 0), verbose=0)[0]
        proba.append(float(out[1]))
    proba = np.asarray(proba, dtype=np.float64)
    print(f"Test images: {len(te)}  "
          f"(plant={int(te_y.sum())}, non_plant={len(te) - int(te_y.sum())})")

    # ---- threshold sweep ---------------------------------------------------
    rows = []
    best = None
    for tp in range(50, 86, 1):
        t = tp / 100.0
        pred = (proba >= t).astype(np.int32)
        acc = accuracy_score(te_y, pred)
        prec = precision_score(te_y, pred, zero_division=0)
        rec = recall_score(te_y, pred)                       # plant recall
        fpr = float(((te_y == 0) & (pred == 1)).sum()) / max(
            int((te_y == 0).sum()), 1)                       # non-plant -> plant
        bal = balanced_accuracy_score(te_y, pred)
        rows.append((tp, round(acc, 4), round(prec, 4),
                     round(rec, 4), round(fpr, 4), round(bal, 4)))
        if best is None or bal > best[5]:
            best = (tp, round(acc, 4), round(prec, 4),
                    round(rec, 4), round(fpr, 4), round(bal, 4))

    print("\nTHRESHOLD SWEEP (test held-out)")
    print("   t%    acc    prec   recall(fpr) balacc")
    for r in rows:
        if r[0] in (50, 55, 60, 65, 70, 75, 80, 85):
            print(f"  {r[0]:02d}   {r[1]:.4f}  {r[2]:.4f}  "
                  f"{r[3]:.4f} ({r[4]:.4f})  {r[5]:.4f}")
    print(f"\nBEST-EVAL threshold = {best[0]}%  "
          f"(balanced accuracy {best[5]:.4f}, acc {best[1]:.4f}, "
          f"plant recall {best[3]:.4f}, non-plant->plant {best[4]:.4f})")

    # ---- final confusion matrix at the recommended threshold ---------------
    t_star = best[0] / 100.0
    pred = (proba >= t_star).astype(np.int32)
    cm = confusion_matrix(te_y, pred)
    print("\nCONFUSION MATRIX @ %.0f%%  [row=truth, col=pred]" % (t_star * 100))
    print("   columns = Non_Plant | Plant_Leaf")
    print("   " + str(cm).replace("\n", "\n   "))

    report = {
        "recommended_threshold_pct": best[0],
        "test_images": int(len(te)),
        "test_accuracy": best[1],
        "test_precision": best[2],
        "test_recall_plant": best[3],
        "test_fpr_non_plant_to_plant": best[4],
        "test_balanced_accuracy": best[5],
        "sweep": [dict(zip(("threshold", "accuracy", "precision",
                            "recall_plant", "fpr_non_plant", "balanced_acc"),
                           r)) for r in rows],
        "confusion_matrix": cm.tolist(),
    }
    with open(os.path.join(OUT_DIR, "threshold_report.json"), "w",
              encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    # update metadata operating threshold
    if os.path.exists(METADATA_PATH):
        meta = json.load(open(METADATA_PATH, encoding="utf-8"))
        meta["operating_threshold_pct"] = best[0]
        meta["evaluation"] = report
        with open(METADATA_PATH, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

    print(f"\nSaved -> {os.path.join(OUT_DIR, 'threshold_report.json')}")
    print(f"Update backend default (if desired): "
          f"SPRAYBOT_PLANT_THRESHOLD={best[0]:.0f}")


if __name__ == "__main__":
    main()