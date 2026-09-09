"""
AI-Driven Precision Agrochemical Sprayer - Model Evaluation

Evaluates the trained model ONLY on the held-out test images that were NEVER
used during training (they come from datasets/split/test, produced by
train_model.py and persisted in models/test_split.json).

Outputs:
    models/evaluation_report.json   (accuracy, per-class report, confusion matrix)
    models/confusion_matrix.png     (saved figure)

Usage:
    python training/evaluate_model.py
"""

import os
import sys
import json
from collections import OrderedDict

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import tensorflow as tf
from PIL import Image
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
TEST_ROOT = os.path.join(PROJECT_ROOT, "datasets", "split", "test")

IMG_SIZE = 224


def load_model_and_classes():
    model_path = os.path.join(MODEL_DIR, "plant_disease_model.keras")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}. Train it first.")
    model = tf.keras.models.load_model(model_path)

    with open(os.path.join(MODEL_DIR, "class_names.json"), "r", encoding="utf-8") as f:
        class_info = json.load(f)
    n = len(class_info)
    class_names = [class_info[str(i)]["folder_name"] for i in range(n)]
    return model, class_names, class_info


def collect_test_pairs(test_root, class_names):
    """Return (class_idx, path) for every image in test_root/<class_folder>/."""
    pairs = []
    for idx, folder in enumerate(class_names):
        d = os.path.join(test_root, folder)
        if not os.path.isdir(d):
            continue
        for f in os.listdir(d):
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                pairs.append((idx, os.path.join(d, f)))
    return pairs


def preprocess(path):
    img = Image.open(path).convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return arr


def main():
    print("=" * 60)
    print("AI-Driven Precision Agrochemical Sprayer - Model Evaluation (held-out test set only)")
    print("=" * 60)

    model, class_names, class_info = load_model_and_classes()
    print(f"[OK] Model loaded. {len(class_names)} classes")

    pairs = collect_test_pairs(TEST_ROOT, class_names)
    print(f"[OK] Test images: {len(pairs)}")

    # Batch predict
    X, y_true = [], []
    for idx, path in pairs:
        X.append(preprocess(path))
        y_true.append(idx)
    X = np.stack(X)
    y_true = np.array(y_true)

    probs = model.predict(X, verbose=1)
    y_pred = np.argmax(probs, axis=1)

    acc = accuracy_score(y_true, y_pred)
    print(f"\n[OK] TEST ACCURACY: {acc * 100:.2f}%")

    # Per-class accuracy
    per_class = []
    for i, name in enumerate(class_names):
        mask = y_true == i
        if mask.sum() == 0:
            per_class.append({
                "index": i, "folder_name": name,
                "display_name": class_info[str(i)]["display_name"],
                "correct": 0, "total": 0, "accuracy": 0.0,
            })
            continue
        correct = int((y_pred[mask] == i).sum())
        per_class.append({
            "index": i, "folder_name": name,
            "display_name": class_info[str(i)]["display_name"],
            "correct": correct, "total": int(mask.sum()),
            "accuracy": round(correct / mask.sum() * 100, 2),
        })

    # Classification report
    report = classification_report(
        y_true, y_pred, labels=list(range(len(class_names))),
        target_names=class_names, output_dict=True, zero_division=0)

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))

    # Save figure
    display_names = [class_info[str(i)]["display_name"] for i in range(len(class_names))]
    plt.figure(figsize=(max(14, len(class_names) * 0.45), max(12, len(class_names) * 0.4)))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title(f"AI-Driven Precision Agrochemical Sprayer Confusion Matrix (Test Accuracy {acc * 100:.2f}%)")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, display_names, rotation=90, fontsize=7)
    plt.yticks(tick_marks, display_names, fontsize=7)
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    cm_path = os.path.join(MODEL_DIR, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=100)
    print(f"[OK] Confusion matrix saved: {cm_path}")

    report_simplified = OrderedDict()
    for name in class_names:
        r = report.get(name, {})
        report_simplified[name] = {
            "precision": round(float(r.get("precision", 0)), 4),
            "recall": round(float(r.get("recall", 0)), 4),
            "f1-score": round(float(r.get("f1-score", 0)), 4),
            "support": int(r.get("support", 0)),
        }
    report_simplified["accuracy"] = round(float(report.get("accuracy", 0)), 4)
    macro = report.get("macro avg", {})
    weighted = report.get("weighted avg", {})
    report_simplified["macro_avg"] = {
        "precision": round(float(macro.get("precision", 0)), 4),
        "recall": round(float(macro.get("recall", 0)), 4),
        "f1-score": round(float(macro.get("f1-score", 0)), 4),
    }
    report_simplified["weighted_avg"] = {
        "precision": round(float(weighted.get("precision", 0)), 4),
        "recall": round(float(weighted.get("recall", 0)), 4),
        "f1-score": round(float(weighted.get("f1-score", 0)), 4),
    }

    summary = {
        "model": os.path.join(MODEL_DIR, "plant_disease_model.keras"),
        "test_set": TEST_ROOT,
        "total_test_images": len(pairs),
        "correct_predictions": int((y_true == y_pred).sum()),
        "overall_accuracy_percent": round(acc * 100, 2),
        "per_class": per_class,
        "classification_report": report_simplified,
        "confusion_matrix": cm.tolist(),
        "eval_date": __import__("time").strftime("%Y-%m-%d %H:%M:%S"),
    }

    report_path = os.path.join(MODEL_DIR, "evaluation_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"[OK] Evaluation report saved: {report_path}")

    print("\n" + "=" * 60)
    print(f"  TEST ACCURACY : {acc * 100:.2f}%  ({int((y_true == y_pred).sum())}/{len(pairs)})")
    print(f"  Macro F1      : {report_simplified['macro_avg']['f1-score']}")
    print(f"  Weighted F1   : {report_simplified['weighted_avg']['f1-score']}")
    print("=" * 60)

    print("\nPer-class accuracy (lowest 8):")
    for pc in sorted(per_class, key=lambda x: x["accuracy"])[:8]:
        print(f"  {pc['display_name']:45s} {pc['accuracy']:6.2f}%  ({pc['correct']}/{pc['total']})")


if __name__ == "__main__":
    main()