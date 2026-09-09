"""
AI-Driven Precision Agrochemical Sprayer - Evaluate Trained Models

Reports real train/val/test numbers only from artifacts already produced:

  models/evaluation_report.json        (combined 41-class model, held-out test)
  models/plant_identifier/metadata.json
  models/plant_validator/metadata.json
  models/disease_models/<crop>_metadata.json
  models/plant_disease_model.keras     (live re-evaluation on the persisted
                                        held-out split when requested)

Usage:
    python training/evaluate_models.py                # summarize artifacts
    python training/evaluate_models.py --re-evaluate  # re-measure combined model
"""

import os
import sys
import json
import argparse

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

MODELS = os.path.join(PROJECT_ROOT, "models")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--re-evaluate", action="store_true",
                    help="re-run the combined model on models/test_split.json")
    args = ap.parse_args()

    print("=== AI-Driven Precision Agrochemical Sprayer - Model Registry ===\n")
    report = {}

    # 1. Combined model artifact (if evaluation was persisted)
    comb = os.path.join(MODELS, "evaluation_report.json")
    if os.path.exists(comb):
        with open(comb) as f:
            r = json.load(f)
        report["combined_41class"] = {"file": comb}
        for k in ("test_accuracy", "train_accuracy", "val_accuracy", "test_loss"):
            if k in r:
                report["combined_41class"][k] = r[k]
                print(f"Combined (41-class): {k} = {r[k]}")

    # 2. Dedicated pipeline models
    entries = [
        ("plant_identifier", os.path.join(MODELS, "plant_identifier", "metadata.json")),
        ("plant_validator", os.path.join(MODELS, "plant_validator", "metadata.json")),
    ]
    for name, p in entries:
        if os.path.exists(p):
            with open(p) as f:
                m = json.load(f)
            report[name] = m
            print(f"{name}: test_accuracy = {m.get('test_accuracy')}"
                  f" ({m.get('test_images', '?')} test images)")

    # 3. Disease models
    dis_dir = os.path.join(MODELS, "disease_models")
    if os.path.isdir(dis_dir):
        report["disease_models"] = {}
        for f in sorted(os.listdir(dis_dir)):
            if f.endswith("_metadata.json"):
                with open(os.path.join(dis_dir, f)) as fh:
                    m = json.load(fh)
                report["disease_models"][f[:-13]] = m
                print(f"disease {m.get('crop')}: test_accuracy = {m.get('test_accuracy')}")

    # 4. Optional live re-evaluation of the combined model
    if args.re_evaluate:
        print("\nRe-evaluating combined model on persisted held-out split ...")
        import numpy as np
        import tensorflow as tf
        from PIL import Image
        from backend.config import CONFIG
        from backend.model_core import CombinedModel

        split_path = os.path.join(MODELS, "test_split.json")
        if not os.path.exists(split_path):
            raise SystemExit("models/test_split.json missing.")
        with open(split_path) as f:
            split = json.load(f)

        cm = CombinedModel()
        correct = total = 0
        cmap = {}
        for i in range(cm.num_classes):
            cmap[cm.class_info[str(i)]["folder_name"]] = i
        for item in split:
            path, label = item.get("path"), item.get("label")
            if not path or not os.path.exists(path):
                continue
            img = Image.open(path).convert("RGB")
            probs = cm.predict_probs(img)
            pred = int(np.argmax(probs))
            total += 1
            if pred == label:
                correct += 1
        acc = correct / total if total else 0.0
        out = {
            "test_images": total,
            "correct_predictions": correct,
            "test_accuracy": round(acc, 4),
        }
        with open(comb, "w") as f:
            json.dump({**(json.load(open(comb)) if os.path.exists(comb) else {}), **out}, f, indent=2)
        print(f"Re-evaluated: accuracy = {acc:.4f} on {total} held-out images.")
        report["combined_41class_re_eval"] = out

    with open(os.path.join(MODELS, "model_registry.json"), "w") as f:
        json.dump(report, f, indent=2)
    print("\nRegistry written to models/model_registry.json")


if __name__ == "__main__":
    main()