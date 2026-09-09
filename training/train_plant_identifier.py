"""
AI-Driven Precision Agrochemical Sprayer - Train Plant Identifier

Stage-2 model: identifies WHICH plant/crop is in the image.

Data: datasets/plant_identification/<Crop>/<Class>/...
Fallback: existing real datasets/processed/<Crop>/ grouped by crop.

Uses Materialized split folders + ImageDataGenerator (the same proven
training stack used by train_model.py) and real MobileNetV2 transfer
learning with frozen-head + fine-tune phases.

Usage:
    python training/train_plant_identifier.py --epochs 14 --batch-size 24
"""

import os
import sys
import json
import time
import shutil
import argparse

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
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from training.dataset_utils import load_plant_identification, write_json

OUT_DIR = os.path.join(PROJECT_ROOT, "models", "plant_identifier")
MODEL_PATH = os.path.join(OUT_DIR, "plant_identifier.keras")
CLASSES_PATH = os.path.join(OUT_DIR, "class_names.json")
METADATA_PATH = os.path.join(OUT_DIR, "metadata.json")
SPLIT_DIR = os.path.join(PROJECT_ROOT, "datasets", "split_identifier")
IMG_SIZE = 224
SEED = 42
HEAD_EPOCHS = 4
FINE_TUNE_LAYERS = 24


def make_split(data):
    """Crop-level stratified split -> (train,val,test) lists of (crop,path)."""
    train, val, test = [], [], []
    for crop, paths in data.items():
        tr, te = train_test_split(paths, test_size=0.15, random_state=SEED, shuffle=True)
        tr2, va = train_test_split(tr, test_size=0.15, random_state=SEED, shuffle=True)
        train += [(crop, p) for p in tr2]
        val += [(crop, p) for p in va]
        test += [(crop, p) for p in te]
    return train, val, test


def materialize(data, train, val, test):
    if os.path.isdir(SPLIT_DIR):
        shutil.rmtree(SPLIT_DIR)
    for subset, items in (("train", train), ("val", val), ("test", test)):
        os.makedirs(os.path.join(SPLIT_DIR, subset), exist_ok=True)
    for subset, items in (("train", train), ("val", val), ("test", test)):
        for crop, p in items:
            dst_dir = os.path.join(SPLIT_DIR, subset, crop)
            os.makedirs(dst_dir, exist_ok=True)
            shutil.copyfile(p, os.path.join(dst_dir, os.path.basename(p)))


def build_generators(batch):
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255.0, rotation_range=25, width_shift_range=0.2,
        height_shift_range=0.2, shear_range=0.15, zoom_range=0.25,
        brightness_range=[0.8, 1.2], horizontal_flip=True, fill_mode="nearest",
    )
    val_datagen = ImageDataGenerator(rescale=1.0 / 255.0)
    train_gen = train_datagen.flow_from_directory(
        os.path.join(SPLIT_DIR, "train"), target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=batch, class_mode="categorical", shuffle=True, seed=SEED)
    val_gen = val_datagen.flow_from_directory(
        os.path.join(SPLIT_DIR, "val"), target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=batch, class_mode="categorical", shuffle=False, seed=SEED)
    return train_gen, val_gen


def build_model(n, img_size):
    base = MobileNetV2(weights="imagenet", include_top=False,
                       input_shape=(img_size, img_size, 3))
    base.trainable = False
    x = base.output
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dense(256, activation="relu")(x)
    x = Dropout(0.5)(x)
    x = Dense(n, activation="softmax", name="plant_output")(x)
    return Model(inputs=base.input, outputs=x), base


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=14)
    ap.add_argument("--batch-size", type=int, default=24)
    ap.add_argument("--no-fallback", action="store_true")
    ap.add_argument("--max-per-crop", type=int, default=0)
    args = ap.parse_args()

    data = load_plant_identification(processed_fallback=not args.no_fallback)
    if not data:
        raise SystemExit("No plant-identification data found.")

    if args.max_per_crop > 0:
        data = {c: p[:args.max_per_crop] for c, p in data.items()}

    crops = sorted(data)
    print(f"Crops: {len(crops)} - total images: {sum(len(v) for v in data.values())}")
    for c in crops:
        print(f"  {c}: {len(data[c])}")

    train, val, test = make_split(data)
    materialize(data, train, val, test)
    print(f"Split materialized: train={len(train)} val={len(val)} test={len(test)}")

    train_gen, val_gen = build_generators(args.batch_size)
    n = len(train_gen.class_indices)
    print(f"Train classes: {n}")

    model, base = build_model(n, IMG_SIZE)
    cbs = [
        EarlyStopping(monitor="val_accuracy", patience=5, restore_best_weights=True, verbose=1),
        ModelCheckpoint(MODEL_PATH, monitor="val_accuracy", save_best_only=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-7, verbose=1),
    ]

    start = time.time()
    model.compile(optimizer=Adam(1e-3), loss="categorical_crossentropy",
                  metrics=["accuracy"])
    print("[Phase A] frozen base")
    model.fit(train_gen, validation_data=val_gen, epochs=HEAD_EPOCHS,
              callbacks=cbs, verbose=1)

    for layer in base.layers[-FINE_TUNE_LAYERS:]:
        layer.trainable = True
    model.compile(optimizer=Adam(1e-4), loss="categorical_crossentropy",
                  metrics=["accuracy"])
    print(f"[Phase B] fine-tune up to {args.epochs} epochs")
    model.fit(train_gen, validation_data=val_gen, epochs=args.epochs,
              callbacks=cbs, verbose=1)

    model.save(MODEL_PATH)
    write_json(CLASSES_PATH, crops)

    test_datagen = ImageDataGenerator(rescale=1.0 / 255.0)
    test_gen = test_datagen.flow_from_directory(
        os.path.join(SPLIT_DIR, "test"), target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=args.batch_size, class_mode="categorical", shuffle=False)
    test_acc = float(model.evaluate(test_gen, verbose=1)[1]) if len(test_gen) > 0 else None
    print(f"[OK] Held-out test accuracy: {test_acc:.4f}" if test_acc is not None else "[WARN] no test images")

    write_json(METADATA_PATH, {
        "project": "AI-Driven Precision Agrochemical Sprayer",
        "architecture": "MobileNetV2 transfer learning (frozen + fine-tune)",
        "num_crops": n,
        "crops": crops,
        "train_images": len(train),
        "val_images": len(val),
        "test_images": len(test),
        "test_accuracy": round(test_acc, 4) if test_acc is not None else None,
        "training_seconds": round(time.time() - start, 1),
        "note": "Held-out test images excluded from training/validation.",
    })
    print(f"[DONE] Saved {MODEL_PATH} / {CLASSES_PATH} "
          f"({time.time()-start:.0f}s)")


if __name__ == "__main__":
    main()