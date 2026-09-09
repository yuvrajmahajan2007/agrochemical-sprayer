"""
AI-Driven Precision Agrochemical Sprayer - Train Crop-Specific Disease Models

Stage-3 models: one classifier per crop (Healthy + crop-specific diseases).

Data: datasets/disease/<crop>/<DiseaseClass>/..., else real
datasets/processed/<crop>/<Class>/...

Because per-crop models are OPTIONAL (the pipeline falls back to the real
41-class combined model restricted to the identified crop), this script
defaults to a report + one crop. Use --crops to train several.

Usage:
    python training/train_disease_models.py --crops Tomato Rice --epochs 15
"""

import os
import sys
import json
import argparse

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
tf.get_logger().setLevel("ERROR")

from training.dataset_utils import load_disease_sets, write_json, IMG_SIZE

OUT_DIR = os.path.join(PROJECT_ROOT, "models", "disease_models")


def build_pipeline(paths, labels, n, batch, augment):
    def gen():
        for p, y in zip(paths, labels):
            yield p, y
    ds = tf.data.Dataset.from_generator(gen, output_types=(tf.string, tf.int32))

    def load(p, y):
        img = tf.io.decode_image(tf.io.read_file(p), channels=3, expand_animations=False)
        img = tf.image.resize(img, (IMG_SIZE, IMG_SIZE))
        img = tf.cast(img, tf.float32) / 255.0
        return img, tf.one_hot(y, n)

    def aug(img, y):
        if augment:
            img = tf.image.random_flip_left_right(img)
            img = tf.image.random_brightness(img, 0.12)
            img = tf.image.random_contrast(img, 0.9, 1.1)
        return img, y

    return (ds.map(load, num_parallel_calls=tf.data.AUTOTUNE)
              .map(aug, num_parallel_calls=tf.data.AUTOTUNE)
              .batch(batch).prefetch(tf.data.AUTOTUNE))


def train_crop(crop, classes, epochs, batch, augment):
    class_names = sorted(classes)
    if len(class_names) < 2:
        print(f"  SKIP {crop}: only {len(class_names)} class present")
        return None
    paths, labels = [], []
    for i, cname in enumerate(class_names):
        imgs = classes[cname]
        print(f"    {cname}: {len(imgs)}")
        paths += imgs
        labels += [i] * len(imgs)

    tr, te = train_test_split(list(zip(paths, labels)), test_size=0.15,
                              stratify=labels, random_state=42)
    tr, va = train_test_split(tr, test_size=0.15,
                              stratify=[y for _, y in tr], random_state=42)
    n = len(class_names)

    base = keras.applications.MobileNetV2(
        input_shape=(IMG_SIZE, IMG_SIZE, 3), include_top=False,
        weights="imagenet", pooling="avg")
    base.trainable = False
    model = keras.Sequential([
        base, layers.Dropout(0.3),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(n, activation="softmax"),
    ])
    model.compile(optimizer=keras.optimizers.Adam(1e-3),
                  loss="categorical_crossentropy", metrics=["accuracy"])

    safe = crop.replace("/", "_").replace(" ", "_")
    model_path = os.path.join(OUT_DIR, f"{safe}_model.keras")
    class_path = os.path.join(OUT_DIR, f"{safe}_classes.json")
    cbs = [
        keras.callbacks.EarlyStopping(patience=4, restore_best_weights=True, monitor="val_loss"),
        keras.callbacks.ModelCheckpoint(model_path, save_best_only=True, monitor="val_accuracy"),
        keras.callbacks.ReduceLROnPlateau(patience=2, factor=0.5, min_lr=1e-6),
    ]
    train_ds = build_pipeline([p for p, _ in tr], [y for _, y in tr], n, batch, augment)
    val_ds = build_pipeline([p for p, _ in va], [y for _, y in va], n, batch, False)
    test_ds = build_pipeline([p for p, _ in te], [y for _, y in te], n, batch, False)

    model.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=cbs, verbose=1)
    loss, acc = model.evaluate(test_ds, verbose=1)
    write_json(class_path, class_names)
    write_json(os.path.join(OUT_DIR, f"{safe}_metadata.json"), {
        "project": "AI-Driven Precision Agrochemical Sprayer",
        "crop": crop,
        "classes": class_names,
        "test_accuracy": round(float(acc), 4),
        "test_loss": round(float(loss), 4),
    })
    print(f"    TEST {crop}: {acc:.4f}")
    return model_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--crops", nargs="*", default=None,
                    help="crop names; default = all available")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--no-augment", action="store_true")
    ap.add_argument("--dry-run", action="store_true",
                    help="only report what could be trained")
    args = ap.parse_args()

    sets = load_disease_sets(crops=args.crops)
    if not sets:
        raise SystemExit("No per-crop disease data found.")
    print("Available crops and classes:")
    for c, classes in sets.items():
        print(f"  {c}: {', '.join(sorted(classes))}")

    if args.dry_run:
        return
    os.makedirs(OUT_DIR, exist_ok=True)
    for crop, classes in sets.items():
        train_crop(crop, classes, args.epochs, args.batch_size,
                   not args.no_augment)


if __name__ == "__main__":
    main()