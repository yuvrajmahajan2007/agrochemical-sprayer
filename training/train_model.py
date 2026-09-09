"""
AI-Driven Precision Agrochemical Sprayer - REAL AI Model Trainer (Transfer Learning: MobileNetV2)

Trains a single multi-class classifier over ALL standardized classes
(crop + disease combined), using a train/validation/test split that is
created once and reused by evaluate_model.py.

Class labels are read from datasets/processed/dataset_manifest.json so the
mapping between class index -> (crop, disease, display_name, status) is exact.

Outputs (in models/):
    plant_disease_model.keras
    class_names.json
    dataset_metadata.json
    training_history.json
    test_split.json            (the held-out test file list reused by evaluation)

Usage:
    python training/train_model.py [--epochs N] [--batch-size N]
"""

import os
import sys
import json
import time
import random
import argparse
import shutil

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import (
    Dense, GlobalAveragePooling2D, Dropout, BatchNormalization, Input
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "datasets", "processed")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
MANIFEST_PATH = os.path.join(PROCESSED_DIR, "dataset_manifest.json")

IMG_SIZE = 224
BATCH_SIZE = 24
HEAD_EPOCHS = 4
FINE_TUNE_EPOCHS = 20
FINE_TUNE_LAYERS = 24
SEED = 42
TEST_FRACTION = 0.15
VALIDATION_FRACTION = 0.15


def parse_args():
    ap = argparse.ArgumentParser(description="AI-Driven Precision Agrochemical Sprayer model trainer")
    ap.add_argument("--epochs", type=int, default=FINE_TUNE_EPOCHS,
                    help="max fine-tune epochs")
    ap.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    ap.add_argument("--img-size", type=int, default=IMG_SIZE)
    return ap.parse_args()


def load_manifest():
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def build_class_list(manifest):
    """Return list of dicts (folder, crop, key, display, status) in canonical order."""
    classes = []
    for c in manifest["classes"]:
        folder = f"{c['crop']}_{c['class_key']}".replace(" ", "_")
        classes.append({
            "folder": folder,
            "crop": c["crop"],
            "class_key": c["class_key"],
            "display_name": c["display_name"],
            "status": c["status"],
            "source": c["source"],
        })
    # deterministic order, dedupe
    seen = set()
    out = []
    for c in classes:
        if c["folder"] in seen:
            continue
        seen.add(c["folder"])
        out.append(c)
    return out


def collect_all_images(processed_dir, class_list):
    """
    Returns list of (class_idx, path) for every image in processed dir,
    ordered by class_list folder names.
    Assumes processed/<crop>/<class_key>/*.
    """
    samples = []
    crops = os.listdir(processed_dir)
    for i, cls in enumerate(class_list):
        crop_dir = os.path.join(processed_dir, cls["crop"])
        class_dir = os.path.join(crop_dir, cls["class_key"])
        if not os.path.isdir(class_dir):
            continue
        for f in os.listdir(class_dir):
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                samples.append((i, os.path.join(class_dir, f)))
    return samples


def make_split(class_list, samples):
    """
    Splits samples into train / val / test, stratified by class, at the
    CLASS LEVEL (dir-by-dir), seeded. The test list is persisted so
    evaluate_model.py uses the exact same held-out set.
    """
    by_class = {i: [] for i in range(len(class_list))}
    for idx, path in samples:
        by_class[idx].append(path)

    train_paths, test_paths = [], []
    val_paths = []
    used_test = []
    for i, paths in by_class.items():
        if len(paths) < 2:
            continue
        # Stratified split
        tr, te = train_test_split(paths, test_size=TEST_FRACTION,
                                  random_state=SEED + i, shuffle=True)
        # Further split validation from training
        if len(tr) >= 4:
            tr2, va = train_test_split(tr, test_size=VALIDATION_FRACTION,
                                       random_state=SEED + i, shuffle=True)
        else:
            tr2, va = tr, []
        train_paths.extend((i, p) for p in tr2)
        val_paths.extend((i, p) for p in va)
        test_paths.extend((i, p) for p in te)
        used_test.append((list(te), i))

    return train_paths, val_paths, test_paths, used_test


def materialize_split_files(split_dir, class_list, train_paths, val_paths, test_paths):
    """Copy selected images into split_dir/<train|val|test>/<class_folder>/*."""
    for subset in ("train", "val", "test"):
        d = os.path.join(split_dir, subset)
        os.makedirs(d, exist_ok=True)
    for subset, paths in (("train", train_paths), ("val", val_paths), ("test", test_paths)):
        for idx, p in paths:
            class_folder = class_list[idx]["folder"]
            dst_dir = os.path.join(split_dir, subset, class_folder)
            os.makedirs(dst_dir, exist_ok=True)
            name = os.path.basename(p)
            dst = os.path.join(dst_dir, f"{idx:03d}_{name}")
            shutil.copyfile(p, dst)


def build_generators(split_dir, batch_size, img_size):
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255.0,
        rotation_range=25,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.15,
        zoom_range=0.25,
        brightness_range=[0.8, 1.2],
        horizontal_flip=True,
        fill_mode="nearest",
    )
    val_datagen = ImageDataGenerator(rescale=1.0 / 255.0)

    train_gen = train_datagen.flow_from_directory(
        os.path.join(split_dir, "train"),
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=True,
        seed=SEED,
    )
    val_gen = val_datagen.flow_from_directory(
        os.path.join(split_dir, "val"),
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=False,
        seed=SEED,
    )
    return train_gen, val_gen, train_gen.class_indices


def build_model(num_classes, img_size):
    base = MobileNetV2(weights="imagenet", include_top=False,
                       input_shape=(img_size, img_size, 3))
    base.trainable = False
    x = base.output
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dense(256, activation="relu")(x)
    x = Dropout(0.5)(x)
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.3)(x)
    out = Dense(num_classes, activation="softmax", name="disease_output")(x)
    model = Model(inputs=base.input, outputs=out)
    return model, base


def compute_class_weights(train_gen, train_paths, class_list):
    """Weight inversely proportional to class frequency (class-imbalance handling)."""
    counts = np.zeros(len(class_list), dtype=np.float64)
    for idx, _ in train_paths:
        counts[idx] += 1
    total = counts.sum()
    if total == 0:
        return None
    n_classes = (counts > 0).sum()
    weights = {}
    for i in range(len(class_list)):
        if counts[i] > 0:
            weights[i] = total / (n_classes * counts[i])
    return weights


def callbacks(model_dir, model_name, history_path):
    return [
        EarlyStopping(monitor="val_accuracy", patience=5,
                      restore_best_weights=True, verbose=1),
        ModelCheckpoint(os.path.join(model_dir, model_name),
                        monitor="val_accuracy", save_best_only=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2,
                          min_lr=1e-7, verbose=1),
    ]


def train_phase(model, train_gen, val_gen, epochs, lr, cbs, class_weights, phase):
    model.compile(optimizer=Adam(learning_rate=lr),
                  loss="categorical_crossentropy", metrics=["accuracy"])
    print(f"[INFO] Phase {phase}: lr={lr}, epochs<={epochs}")
    history = model.fit(
        train_gen,
        epochs=epochs,
        validation_data=val_gen,
        callbacks=cbs,
        class_weight=class_weights,
        verbose=2,
    )
    return history


def save_artifacts(model, class_list, train_gen, histories, manifest, used_test,
                   split_dir, model_dir, img_size):
    model.save(os.path.join(model_dir, "plant_disease_model.keras"))
    print(f"[OK] Model saved: {os.path.join(model_dir, 'plant_disease_model.keras')}")

    # class_names.json: index -> rich info
    class_info = {}
    indices = train_gen.class_indices  # folder -> idx
    for folder, idx in indices.items():
        # find matching class list entry
        entry = next((c for c in class_list if c["folder"] == folder), None)
        if entry is None:
            continue
        if entry["status"] == "Healthy":
            display = f"{entry['crop']} Healthy"
        elif entry["crop"].lower() in entry["display_name"].lower():
            display = entry["display_name"]
        else:
            display = f"{entry['crop']} {entry['display_name']}"
        class_info[str(idx)] = {
            "folder_name": folder,
            "crop": entry["crop"],
            "display_name": display,
            "disease": entry["display_name"] if entry["status"] != "Healthy" else None,
            "status": entry["status"],
            "source": entry["source"],
        }
    with open(os.path.join(model_dir, "class_names.json"), "w", encoding="utf-8") as f:
        json.dump(class_info, f, indent=2, ensure_ascii=False)
    print(f"[OK] class_names.json ({len(class_info)} classes)")

    # dataset_metadata.json
    crops = {}
    for cls in class_list:
        crops[cls["crop"]] = crops.get(cls["crop"], 0) + 1

    metadata = {
        "project": "AI-Driven Precision Agrochemical Sprayer",
        "architecture": "Single multi-class MobileNetV2 (transfer learning)",
        "image_size": img_size,
        "num_classes": len(class_list),
        "crops": sorted(crops),
        "num_crops": len(crops),
        "crop_class_counts": crops,
        "classes": class_list,
        "datasets_used": manifest["datasets"],
        "test_split_file": "test_split.json",
    }
    with open(os.path.join(model_dir, "dataset_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    # persist test split
    test_payload = {"test_paths": []}
    for test_list, i in used_test:
        for p in test_list:
            test_payload["test_paths"].append({"class_idx": i, "path": p})
    with open(os.path.join(model_dir, "test_split.json"), "w", encoding="utf-8") as f:
        json.dump(test_payload, f, indent=2)

    # history
    merged = {}
    for h in histories:
        for metric, values in h["history"].items():
            merged.setdefault(metric, []).extend(float(v) for v in values)
    history_data = {
        "by_phase": [{"phase": h["phase"], "history": h["history"]} for h in histories],
        "accuracy": merged.get("accuracy", []),
        "val_accuracy": merged.get("val_accuracy", []),
        "loss": merged.get("loss", []),
        "val_loss": merged.get("val_loss", []),
    }
    with open(os.path.join(model_dir, "training_history.json"), "w") as f:
        json.dump(history_data, f, indent=2)

    # summary print
    print("\n" + "=" * 60)
    if merged.get("val_accuracy"):
        print(f"  BEST VALIDATION ACCURACY: {max(merged['val_accuracy']) * 100:.2f}%")
    if merged.get("accuracy"):
        print(f"  FINAL TRAINING ACCURACY:  {merged['accuracy'][-1] * 100:.2f}%")
    print("=" * 60)


def main():
    args = parse_args()

    print("=" * 60)
    print("AI-Driven Precision Agrochemical Sprayer - REAL AI Model Training")
    print("=" * 60)

    manifest = load_manifest()
    class_list = build_class_list(manifest)
    print(f"[OK] {len(class_list)} classes from manifest")

    samples = collect_all_images(PROCESSED_DIR, class_list)
    total_images = len(samples)
    print(f"[OK] {total_images} total images in processed dataset")

    train_paths, val_paths, test_paths, used_test = make_split(class_list, samples)
    print(f"[OK] Split: train={len(train_paths)} val={len(val_paths)} test={len(test_paths)}")

    os.makedirs(MODEL_DIR, exist_ok=True)

    split_dir = os.path.join(PROJECT_ROOT, "datasets", "split")
    if os.path.isdir(split_dir):
        shutil.rmtree(split_dir)
    materialize_split_files(split_dir, class_list, train_paths, val_paths, test_paths)
    print(f"[OK] Split materialized at {split_dir}")

    train_gen, val_gen, _ = build_generators(split_dir, args.batch_size, args.img_size)
    print(f"[OK] train classes: {len(train_gen.class_indices)}")

    model, base = build_model(len(train_gen.class_indices), args.img_size)
    model.summary()

    class_weights = compute_class_weights(train_gen, train_paths, class_list)
    if class_weights:
        print(f"[OK] Class weights applied for {sum(1 for v in class_weights.values() if v > 1)} minority classes")

    cb = callbacks(MODEL_DIR, "plant_disease_model.keras",
                   os.path.join(MODEL_DIR, "training_history.json"))

    start = time.time()
    t0 = time.time()
    hist_a = train_phase(model, train_gen, val_gen, HEAD_EPOCHS, 1e-3,
                         cb, class_weights, "A (frozen base)")
    print(f"[INFO] Phase A completed in {time.time() - t0:.0f}s")

    for layer in base.layers[-FINE_TUNE_LAYERS:]:
        layer.trainable = True
    trainable = sum(tf.keras.backend.count_params(w) for w in model.trainable_weights)
    print(f"[INFO] Phase B trainable parameters: {trainable:,}")

    t0 = time.time()
    hist_b = train_phase(model, train_gen, val_gen, args.epochs, 1e-4,
                         cb, class_weights, "B (fine-tune)")
    print(f"[INFO] Phase B completed in {time.time() - t0:.0f}s")

    print(f"\n[OK] Total training time: {time.time() - start:.1f}s")

    save_artifacts(model, class_list, train_gen,
                   [{"phase": "A_frozen_base", "history": dict(hist_a.history)},
                    {"phase": "B_fine_tune", "history": dict(hist_b.history)}],
                   manifest, used_test, split_dir, MODEL_DIR, args.img_size)

    print("\n[DONE] Training complete!")


if __name__ == "__main__":
    main()