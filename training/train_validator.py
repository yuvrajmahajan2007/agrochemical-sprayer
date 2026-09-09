"""
AI-Driven Precision Agrochemical Sprayer - Train a REAL Plant/Leaf validator

Stage-1 binary model:  class 0 = Non_Plant, class 1 = Plant_Leaf
Architecture: MobileNetV2 (ImageNet) transfer-learning + dropout + softmax.

Data (REAL images only, never synthetic):
    datasets/plant_validation/plant/      Plant_Leaf  (leaves/diseased leaves,
                                                     multi-crop, mobile photos)
    datasets/plant_validation/non_plant/  Non_Plant   (real photographs: food,
                                                     humans, buildings, cars,
                                                     electronics, stones, paper,
                                                     furniture, empty scenes)

Training outputs of every run (printed AND saved in metadata.json):
    - number of images per class
    - train / validation / test split sizes
    - training accuracy and validation accuracy (per epoch, final best)
    - test accuracy (held-out)
    - precision / recall / F1 (test, at the deployed threshold)

A separate threshold sweep (training/evaluate_validator.py) picks the operating
threshold and reports the confusion matrix.

Usage:
    python training/train_validator.py --epochs 15 --batch-size 32
"""

import os
import sys
import json
import argparse
from collections import Counter

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import numpy as np  # noqa: E402
import tensorflow as tf  # noqa: E402
from tensorflow import keras  # noqa: E402
from tensorflow.keras import layers  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402
from sklearn.metrics import (accuracy_score, precision_score,  # noqa: E402
                             recall_score, f1_score,
                             confusion_matrix, classification_report)

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
tf.get_logger().setLevel("ERROR")

from training.dataset_utils import load_plant_validation  # noqa: E402
from training.dataset_utils import IMG_SIZE  # noqa: E402

OUT_DIR = os.path.join(PROJECT_ROOT, "models", "plant_validator")
MODEL_PATH = os.path.join(OUT_DIR, "plant_validator.keras")
METADATA_PATH = os.path.join(OUT_DIR, "metadata.json")
CLASS_NAMES_PATH = os.path.join(OUT_DIR, "class_names.json")
CLASS_ORDER = ["Non_Plant", "Plant_Leaf"]
# Default operating threshold (%). Fine-tuned by evaluate_validator.py sweep;
# this value is what the ONLINE /predict stage compares the model prob against.
DEFAULT_THRESHOLD = 65.0


# --------------------------------------------------------------------------
# Biologically-realistic augmentation. Rotation, zoom/shift (via crop/pad),
# brightness/contrast variation and a slight blur - never changes the class
# or makes a leaf look like a non-leaf.
# --------------------------------------------------------------------------
def build_pipeline(items, n_classes, batch, augment):
    def generator():
        for p, y in items:
            yield p, y

    ds = tf.data.Dataset.from_generator(
        generator,
        output_types=(tf.string, tf.int32),
        output_shapes=(tf.TensorShape([]), tf.TensorShape([])))
    ds = ds.shuffle(max(1000, len(items) // 4),
                    reshuffle_each_iteration=True)

    def load(p, y):
        img = tf.io.decode_image(tf.io.read_file(p), channels=3,
                                 expand_animations=False)
        img = tf.cast(img, tf.float32) / 255.0
        if augment:
            # zoom + small translations: pad to 248 then centre-crop back
            img = tf.image.resize_with_crop_or_pad(img, IMG_SIZE + 24,
                                                   IMG_SIZE + 24)
            img = tf.image.random_crop(img, (IMG_SIZE, IMG_SIZE, 3))
            # rotation robustness (0/90/180/270 UHD flips - leaf angle-free)
            k = tf.random.uniform((), 0, 4, dtype=tf.int32)
            img = tf.image.rot90(img, k=k)
            img = tf.image.random_flip_left_right(img)
            # lighting diversity
            img = tf.image.random_brightness(img, 0.20)
            img = tf.image.random_contrast(img, 0.80, 1.20)
            # slight blur (mild down/up resize) - keeps leaf recognisable
            if tf.random.uniform(()) < 0.20:
                img = tf.image.resize(img, (IMG_SIZE // 2, IMG_SIZE // 2))
        img = tf.image.resize(img, (IMG_SIZE, IMG_SIZE))
        return img, tf.one_hot(y, n_classes)

    return (ds.map(load, num_parallel_calls=tf.data.AUTOTUNE)
              .batch(batch).prefetch(tf.data.AUTOTUNE))


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--no-augment", action="store_true")
    args = ap.parse_args()

    data = load_plant_validation()
    plant = data.get("plant", [])
    non_plant = data.get("non_plant", [])
    if not plant or not non_plant:
        raise SystemExit(
            "Cannot train a REAL validator: need real images in BOTH "
            "datasets/plant_validation/plant/ and "
            "datasets/plant_validation/non_plant/ (run "
            "build_validator_dataset.py first). Refusing to fake labels.")

    items = [(p, 0) for p in non_plant] + [(p, 1) for p in plant]
    labels = [y for _, y in items]

    # stratified splits: 70% train / 15% val / 15% test (never touched by fit)
    tr, te = train_test_split(items, test_size=0.15,
                              stratify=labels, random_state=42)
    tr, va = train_test_split(tr, test_size=0.15 / 0.85,
                              stratify=[y for _, y in tr], random_state=42)

    n_plant, n_non = len(plant), len(non_plant)
    print("=" * 62)
    print("VALIDATOR DATASET (real images only)")
    print(f"  Plant_Leaf : {n_plant}")
    print(f"  Non_Plant  : {n_non}")
    print(f"  Train {len(tr)}  Val {len(va)}  Test {len(te)}")
    print("=" * 62)

    base = keras.applications.MobileNetV2(
        input_shape=(IMG_SIZE, IMG_SIZE, 3), include_top=False,
        weights="imagenet", pooling="avg")
    base.trainable = False
    model = keras.Sequential([
        base,
        layers.Dropout(0.35),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.30),
        layers.Dense(2, activation="softmax"),
    ])
    model.compile(optimizer=keras.optimizers.Adam(1e-3),
                  loss="categorical_crossentropy", metrics=["accuracy"])

    os.makedirs(OUT_DIR, exist_ok=True)
    cbs = [
        keras.callbacks.EarlyStopping(patience=4, restore_best_weights=True,
                                      monitor="val_loss"),
        keras.callbacks.ModelCheckpoint(MODEL_PATH, save_best_only=True,
                                        monitor="val_accuracy",
                                        mode="max"),
        keras.callbacks.ReduceLROnPlateau(patience=2, factor=0.5, min_lr=1e-6),
    ]
    # balance classes by oversampling the (much rarer) real negatives in the
    # TRAIN split only - test/val stay untouched so metrics are honest.
    n_neg = sum(1 for _, y in tr if y == 0)
    repeat = max(1, int(round((len(tr) - n_neg) / max(n_neg, 1))))
    tr_bal = tr + [(p, 0) for p, y in tr if y == 0] * (repeat - 1)
    print(f"Balance: repeating non-plant train items x{repeat} "
          f"-> train items {len(tr_bal)}")

    train_ds = build_pipeline(tr_bal, 2, args.batch_size,
                              augment=not args.no_augment)
    val_ds = build_pipeline(va, 2, args.batch_size, augment=False)
    test_ds = build_pipeline(te, 2, args.batch_size, augment=False)

    history = model.fit(
        train_ds, validation_data=val_ds, epochs=args.epochs,
        callbacks=cbs, verbose=1)

    test_loss, test_acc = model.evaluate(test_ds, verbose=1)

    # ---- held-out metrics at the operating threshold ----------------------
    te_paths = [x for x, _ in te]
    te_y = np.array([y for _, y in te], dtype=np.int32)

    def predict_proba(paths):
        proba = []
        for p in paths:
            img = tf.io.decode_image(tf.io.read_file(p), channels=3,
                                     expand_animations=False)
            img = tf.image.resize(tf.cast(img, tf.float32), (IMG_SIZE, IMG_SIZE))
            out = model.predict(tf.expand_dims(img / 255.0, 0), verbose=0)[0]
            proba.append(out[1])
        return np.asarray(proba, dtype=np.float64)

    te_proba = predict_proba(te_paths)
    te_pred = (te_proba >= DEFAULT_THRESHOLD / 100.0).astype(np.int32)
    acc = accuracy_score(te_y, te_pred)
    prec = precision_score(te_y, te_pred, zero_division=0)
    rec = recall_score(te_y, te_pred)
    f1 = f1_score(te_y, te_pred, zero_division=0)
    cm = confusion_matrix(te_y, te_pred)

    print("=" * 62)
    print("TEST  (held-out, threshold = %.0f%%)" % DEFAULT_THRESHOLD)
    print(f"  Accuracy : {acc:.4f}")
    print(f"  Precision: {prec:.4f}   Recall: {rec:.4f}   F1: {f1:.4f}")
    print("  Confusion matrix [row=truth, col=pred]:"
          "\n    columns = Non_Plant | Plant_Leaf")
    print("    " + str(cm).replace("\n", "\n    "))
    print(classification_report(te_y, te_pred,
                                target_names=CLASS_ORDER, zero_division=0))
    print("=" * 62)

    # ---- train/val accuracy (for the printed report) ----------------------
    train_acc = float(history.history["accuracy"][-1])
    val_acc = float(history.history["val_accuracy"][-1])

    write_json(METADATA_PATH, {
        "project": "AI-Driven Precision Agrochemical Sprayer",
        "architecture": "MobileNetV2 transfer learning (real binary model)",
        "classes": CLASS_ORDER,
        "class_0": CLASS_ORDER[0],
        "class_1": CLASS_ORDER[1],
        "n_plant_leaf_images": n_plant,
        "n_non_plant_images": n_non,
        "split": {"train": len(tr), "val": len(va), "test": len(te)},
        "train_accuracy": round(train_acc, 4),
        "val_accuracy": round(val_acc, 4),
        "test_accuracy": round(acc, 4),
        "test_loss": round(float(test_loss), 4),
        "test_precision": round(prec, 4),
        "test_recall": round(rec, 4),
        "test_f1": round(f1, 4),
        "confusion_matrix": cm.tolist(),
        "operating_threshold_pct": DEFAULT_THRESHOLD,
        "augmentation": "crop/pad zoom+shift, rot90, flip, brightness±20%, "
                        "contrast 0.8-1.2, mild blur 20%",
    })
    with open(CLASS_NAMES_PATH, "w", encoding="utf-8") as f:
        json.dump(CLASS_ORDER, f)

    print(f"\nSaved REAL model -> {MODEL_PATH}")
    print(f"Metadata       -> {METADATA_PATH}")
    print(f"\nTRAIN acc={train_acc:.4f}  VAL acc={val_acc:.4f}  "
          f"TEST acc={acc:.4f}  TEST F1={f1:.4f}")


def write_json(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


if __name__ == "__main__":
    main()