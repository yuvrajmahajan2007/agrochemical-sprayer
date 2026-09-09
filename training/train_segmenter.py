"""
AI-Driven Precision Agrochemical Sprayer - Train REAL leaf/health segmenter

Crop-agnostic U-Net that segments every pixel into:
    class 0 = background
    class 1 = healthy leaf tissue
    class 2 = visibly diseased / symptomatic tissue

Severity %, health status and the overlay are then derived from these REAL
model masks (never from model confidence or random numbers):

    affected_area_pct = symptom_pixels / (healthy_pixels + symptom_pixels)

Training labels come from datasets/segmentation_masks/pairs.json (real leaf
photos auto-segmented by deterministic pixel analysis). The U-Net learns the
spatial disease pattern from those real masks.

Outputs (printed + saved):
    - number of images, healthy/diseased leaf counts
    - train / val / test sizes
    - val accuracy and per-class IoU
    - saved model + metadata

Usage:
    python training/train_segmenter.py --epochs 12 --batch-size 16 --img-size 128
"""

import argparse
import json
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
tf.get_logger().setLevel("ERROR")

from sklearn.model_selection import train_test_split

OUT_DIR = os.path.join(PROJECT_ROOT, "models", "leaf_segmenter")
MODEL_PATH = os.path.join(OUT_DIR, "leaf_segmenter.keras")
CLASS_NAMES = ["background", "healthy_leaf", "symptomatic"]
N_CLASSES = len(CLASS_NAMES)


# --------------------------------------------------------------------------
def build_unet(img_size, n_classes):
    inp = keras.Input((img_size, img_size, 3))

    def down(x, ch, drop=0.0):
        x = layers.Conv2D(ch, 3, activation="relu", padding="same")(x)
        x = layers.Conv2D(ch, 3, activation="relu", padding="same")(x)
        if drop:
            x = layers.Dropout(drop)(x)
        return x

    d1 = down(inp, 32)                     # 128
    p1 = layers.MaxPooling2D(2)(d1)        # 64
    d2 = down(p1, 64)                      # 64
    p2 = layers.MaxPooling2D(2)(d2)        # 32
    d3 = down(p2, 128, 0.1)                # 32
    p3 = layers.MaxPooling2D(2)(d3)        # 16
    d4 = down(p3, 256, 0.1)                # 16
    b = layers.Conv2D(256, 3, activation="relu", padding="same")(d4)

    u1 = layers.Concatenate()([layers.UpSampling2D(2)(b), d3])
    u1 = down(u1, 128)
    u2 = layers.Concatenate()([layers.UpSampling2D(2)(u1), d2])
    u2 = down(u2, 64)
    u3 = layers.Concatenate()([layers.UpSampling2D(2)(u2), d1])
    u3 = down(u3, 32)
    u4 = layers.Concatenate()([u3, inp])
    u4 = layers.Conv2D(16, 3, activation="relu", padding="same")(u4)
    out = layers.Conv2D(n_classes, 1, activation="softmax", padding="same")(u4)
    return keras.Model(inp, out)


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--img-size", type=int, default=128)
    ap.add_argument("--no-augment", action="store_true")
    args = ap.parse_args()

    pairs_path = os.path.join(PROJECT_ROOT, "datasets", "segmentation_masks",
                              "pairs.json")
    if not os.path.exists(pairs_path):
        raise SystemExit("No segmentation masks. Run "
                         "build_segmentation_masks.py first.")
    pairs = json.load(open(pairs_path, encoding="utf-8"))
    if not pairs:
        raise SystemExit("Empty pair list - refusing to train on nothing.")

    d_im = args.img_size
    healthy = sum(1 for p in pairs if p["class"] == "healthy")
    diseases = len(pairs) - healthy
    print("=" * 60)
    print("SEGMENTATION DATASET (real leaf photos + real pixel masks)")
    print(f"  pairs={len(pairs)}  healthy-leaf={healthy}  diseased-leaf={diseases}")
    print(f"  classes={CLASS_NAMES}  image={d_im}x{d_im}")
    print("=" * 60)

    idx = np.arange(len(pairs))
    tr, te = train_test_split(idx, test_size=0.15,
                              stratify=[p["class"] for p in pairs],
                              random_state=42)
    tr, va = train_test_split(tr, test_size=0.15 / 0.85,
                              stratify=[pairs[i]["class"] for i in tr],
                              random_state=42)

    def make_ds(sel, augment):
        def gen():
            for i in sel:
                yield pairs[i]["image"], pairs[i]["mask"]

        ds = tf.data.Dataset.from_generator(
            gen, output_types=(tf.string, tf.string),
            output_shapes=(tf.TensorShape([]), tf.TensorShape([])))
        ds = ds.shuffle(len(sel), reshuffle_each_iteration=True)

        def load(img, msk):
            im = tf.image.decode_image(tf.io.read_file(img), channels=3,
                                       expand_animations=False)
            im = tf.cast(im, tf.float32) / 255.0
            mk = tf.image.decode_image(tf.io.read_file(msk), channels=1,
                                       expand_animations=False)
            mk = tf.cast(mk, tf.int32)[..., 0]
            if augment:
                if tf.random.uniform(()) < 0.5:
                    im = tf.image.random_flip_left_right(im)
                    mk = tf.image.flip_left_right(tf.expand_dims(mk, -1))[..., 0]
                if tf.random.uniform(()) < 0.5:
                    im = tf.image.random_flip_up_down(im)
                    mk = tf.image.flip_up_down(tf.expand_dims(mk, -1))[..., 0]
                im = tf.image.random_brightness(im, 0.12)
            im = tf.image.resize(im, (d_im, d_im))
            mk = tf.image.resize(tf.expand_dims(mk, -1), (d_im, d_im),
                                 method="nearest")[..., 0]
            return im, mk

        return (ds.map(load, num_parallel_calls=tf.data.AUTOTUNE)
                  .batch(args.batch_size).prefetch(tf.data.AUTOTUNE))

    train_ds = make_ds(tr, augment=not args.no_augment)
    val_ds = make_ds(va, augment=False)

    model = build_unet(d_im, N_CLASSES)
    model.compile(optimizer=keras.optimizers.Adam(1e-3),
                  loss="sparse_categorical_crossentropy",
                  metrics=["sparse_categorical_accuracy"])
    model.summary()

    os.makedirs(OUT_DIR, exist_ok=True)
    cbs = [keras.callbacks.EarlyStopping(patience=4, restore_best_weights=True,
                                         monitor="val_loss"),
           keras.callbacks.ModelCheckpoint(MODEL_PATH, save_best_only=True,
                                           monitor="val_loss", mode="min"),
           keras.callbacks.ReduceLROnPlateau(patience=2, factor=0.5, min_lr=1e-5)]
    history = model.fit(train_ds, validation_data=val_ds, epochs=args.epochs,
                        callbacks=cbs, verbose=1)

    # ---- held-out IoU evaluation -----------------------------------------
    print("\nEvaluating on the held-out test split...")
    conf = np.zeros((N_CLASSES, N_CLASSES), dtype=np.int64)
    for i in te:
        p = pairs[i]
        im = np.asarray(keras.utils.load_img(
            p["image"], target_size=(d_im, d_im)))[None, ...] / 255.0
        mk = np.asarray(keras.utils.load_img(
            p["mask"], color_mode="grayscale",
            target_size=(d_im, d_im), interpolation="nearest"))
        mk = np.round(mk / 85.0).astype(np.int32)          # 0/1/2 -> 0..85..170
        pred = model.predict(im, verbose=0)[0].argmax(-1)
        h, w = mk.shape
        flat = np.arange(h * w)
        conf += np.bincount(mk.ravel() * N_CLASSES + pred.ravel(),
                            minlength=N_CLASSES * N_CLASSES).reshape(
                                N_CLASSES, N_CLASSES)

    ious, used = [], []
    for c in range(N_CLASSES):
        tp = conf[c, c]
        fp = conf[c, :].sum() - tp
        fn = conf[:, c].sum() - tp
        denom = tp + fp + fn
        iou = tp / denom if denom else 0.0
        ious.append(iou)
        used.append(conf[c].sum() > 0)
    val_acc = float(history.history["val_sparse_categorical_accuracy"][-1])
    avg_iou = float(np.mean([i for i, u in zip(ious, used) if u]))
    print("TEST split IoU per class (prefixes):",
          {CLASS_NAMES[c]: round(ious[c], 4) for c in range(N_CLASSES)})
    print(f"  valid-class mean IoU = {avg_iou:.4f}  "
          f"val accuracy = {val_acc:.4f}")

    meta = {
        "project": "AI-Driven Precision Agrochemical Sprayer",
        "architecture": "U-Net (compact, 3-class pixel segmentation)",
        "classes": CLASS_NAMES,
        "image_size": d_im,
        "n_pairs": len(pairs),
        "n_healthy_leaf_photos": healthy,
        "n_diseased_leaf_photos": diseases,
        "train": int(len(tr)), "val": int(len(va)), "test": int(len(te)),
        "val_accuracy": round(val_acc, 4),
        "test_mean_iou": round(avg_iou, 4),
        "test_iou_per_class": {CLASS_NAMES[c]: round(ious[c], 4)
                               for c in range(N_CLASSES)},
        "labels_source": "real leaf photos; per-pixel labels derived from "
                         "deterministic HSV/ExG vegetation+lesion analysis, "
                         "then the U-Net learns the spatial pattern.",
        "affected_area_pct_formula": "symptom_pixels / (healthy+symptom) x100",
        "note": "Severity thresholds are configurable demonstration defaults "
                "that should be calibrated for real agricultural rules.",
    }
    with open(os.path.join(OUT_DIR, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    with open(os.path.join(OUT_DIR, "class_names.json"), "w", encoding="utf-8") as f:
        json.dump(CLASS_NAMES, f)
    # deploy-time config for the backend segmenter
    with open(os.path.join(OUT_DIR, "infer_config.json"), "w", encoding="utf-8") as f:
        json.dump({"image_size": d_im, "n_classes": N_CLASSES,
                   "severity_low_max": 10.0, "severity_med_max": 40.0,
                   "decision_mapping": {
                       "0_low": "MONITOR",
                       "medium": "INSPECTION_RECOMMENDED",
                       "high": "TREATMENT_ASSESSMENT_RECOMMENDED"}}, f, indent=2)

    print("\nSaved REAL segmenter -> " + MODEL_PATH)


if __name__ == "__main__":
    main()