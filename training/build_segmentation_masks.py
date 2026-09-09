"""
AI-Driven Precision Agrochemical Sprayer - Build REAL leaf-segmentation masks

Stage for the redesigned, crop-agnostic pipeline:

    Leaf/Plant Detection -> Healthy-leaf segmentation -> Symptomatic-area
    segmentation -> affected area % -> severity -> spray decision support

This script generates per-pixel 3-class training labels on REAL leaf photos
that are already on disk (datasets/processed/<Crop>/<Class>):

    class 0 = background (soil, sky, hand, pot ...)
    class 1 = healthy leaf tissue (saturated green)
    class 2 = visibly diseased / symptomatic tissue
              (yellow-brown necrosis, dull lesions, blotches)

The labels are derived from real pixel evidence (HSV/ExG vegetation + lesion
tones) - deterministic, explainable, no invented labels. The U-Net then
LEARNS the spatial pattern of the disease so inference on a new photo is a
real model output, not a pixel heuristic.

Healthy classes are any folder named "healthy"; every other class folder is
treated as a visibly diseased leaf.

Output layout:
    datasets/segmentation_masks/
        pairs.json                [ {"image": path, "mask": path, "class": "healthy|diseased"} ]
        masks/<name>.png          single-channel label image (0/1/2)

Usage:
    python training/build_segmentation_masks.py --n 1500 --seed 42
"""

import argparse
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from PIL import Image

from training.dataset_utils import DATASETS_ROOT, scan_folder_images

PROCESSED = os.path.join(DATASETS_ROOT, "processed")
OUT_ROOT = os.path.join(DATASETS_ROOT, "segmentation_masks")
MASK_DIR = os.path.join(OUT_ROOT, "masks")


def is_healthy_class(class_folder):
    name = class_folder.strip().lower()
    return name == "healthy" or name.endswith(" healthy") or "healthy" in name.split()


def make_mask(rgb):
    """Real pixel labels: 0 background, 1 healthy leaf, 2 symptomatic leaf."""
    arr = np.asarray(rgb.convert("RGB"), dtype=np.float32)
    if arr.size == 0:
        return None
    hsv = np.asarray(rgb.convert("HSV"), dtype=np.float32)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

    sky = (v > 200) & (s < 40)                       # bright low-sat background
    green = (s >= 40) & (h <= 100) & (h >= 25)       # green leaf tones
    lesion = ((h >= 15) & (h <= 40) & (s >= 60)) | \
             ((h > 100) & (h <= 170) & (s >= 30))     # brown / olive lesions
    dark_leaf = (s < 60) & (v > 40) & (v < 200) & (h <= 115)

    foreground = (green | lesion | dark_leaf) & ~sky
    if foreground.sum() < max(200, arr.shape[0] * arr.shape[1] * 0.01):
        return None                                   # no confident leaf area

    # Healthy: saturated green tissue under decent light.
    healthy = green & (s >= 60) & (v >= 55) & ~sky
    # Symptomatic: confident yellow/brown/necrotic tones on leaf foreground
    # (kept conservative so background or shading does not inflate the area).
    symptom = foreground & (
        ((h >= 15) & (h <= 60) & (s >= 40) & (v <= 150)) |      # yellow-brown nec
        ((h > 100) & (h <= 170) & (s >= 30)) |                   # olive / brown
        ((h >= 25) & (h <= 45) & (s >= 20) & (s < 60) & (v > 60) & (v <= 180))
    )

    mask = np.zeros(arr.shape[:2], dtype=np.uint8)
    mask[healthy] = 1
    mask[symptom] = 2
    mask[~foreground] = 0
    return mask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1500,
                    help="approx images per health category (healthy/diseased)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if not os.path.isdir(PROCESSED):
        raise SystemExit(f"no datasets/processed at {PROCESSED}")

    healthy_paths, diseased_paths = [], []
    for crop in sorted(os.listdir(PROCESSED)):
        cdir = os.path.join(PROCESSED, crop)
        if not os.path.isdir(cdir):
            continue
        for cls in sorted(os.listdir(cdir)):
            cldir = os.path.join(cdir, cls)
            if not os.path.isdir(cldir):
                continue
            imgs = scan_folder_images(cldir)
            if is_healthy_class(cls):
                healthy_paths += imgs
            else:
                diseased_paths += imgs

    print(f"Healthy leaf photos : {len(healthy_paths)}")
    print(f"Diseased leaf photos: {len(diseased_paths)}")

    random.seed(args.seed)
    random.shuffle(healthy_paths)
    random.shuffle(diseased_paths)
    healthy = healthy_paths[: args.n]
    diseased = diseased_paths[: args.n]
    selected = healthy + diseased

    os.makedirs(MASK_DIR, exist_ok=True)
    pairs, done = [], 0
    for path in selected:
        try:
            rgb = Image.open(path).convert("RGB")
        except Exception:
            continue
        mask = make_mask(rgb)
        if mask is None:
            continue
        name = os.path.splitext(os.path.basename(path))[0] + f"__{done:05d}.png"
        mpath = os.path.join(MASK_DIR, name)
        Image.fromarray(mask).save(mpath)
        klass = "diseased" if path in set(diseased) else "healthy"
        pairs.append({"image": path, "mask": mpath, "class": klass})
        done += 1

    with open(os.path.join(OUT_ROOT, "pairs.json"), "w", encoding="utf-8") as f:
        json.dump(pairs, f, indent=1)

    n_healthy = sum(1 for p in pairs if p["class"] == "healthy")
    n_diseases = sum(1 for p in pairs if p["class"] == "diseased")
    print(f"Mask dataset written : {len(pairs)} masks "
          f"(healthy={n_healthy}, diseased leaf={n_diseases})")
    print(f"  -> {OUT_ROOT}")


if __name__ == "__main__":
    main()