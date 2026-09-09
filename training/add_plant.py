"""
AI-Driven Precision Agrochemical Sprayer - Add a New Plant / Crop

Makes the plant-identification catalog EXTENSIBLE without touching any code:

    1. Drop real, diverse images for a new crop into a folder:
           datasets/raw/<CropName>/          (class subfolders optional)
       or pass any folder of images directly.

    2. Run:
           python training/add_plant.py "<CropName>" [<source folder>]

       add_plant.py:
           - validates + de-duplicates every image (no corrupt file survives)
           - copies them into  datasets/plant_identification/<CropName>/
             (this folder is auto-discovered by load_plant_identification,
             and by train_plant_identifier on the next retrain)
           - registers the crop in models/crop_metadata.json and
             models/supported_crops.json (identifiable, no disease model yet)

    3. Retrain the Stage-2 identifier to learn the new crop:
           run_train_identifier.cmd
       (or: python training/train_plant_identifier.py --epochs 14)

    Evidence over guesses: a crop is only identifiable after a retrain with
    its REAL images - the system never fabricates classes from empty folders.

Usage:
    python training/add_plant.py "Wheat"
    python training/add_plant.py "Wheat" "C:/my_phone_photos/wheat_leaves"
    python training/add_plant.py "Tomato" "datasets/raw/PlantVillage" --with-disease
"""

import os
import sys
import json
import hashlib
import shutil
import argparse
from datetime import datetime

from PIL import Image, UnidentifiedImageError

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASETS_ROOT = os.path.join(PROJECT_ROOT, "datasets")
PLANT_ID_ROOT = os.path.join(DATASETS_ROOT, "plant_identification")
RAW_ROOT = os.path.join(DATASETS_ROOT, "raw")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
CROP_META_PATH = os.path.join(MODEL_DIR, "crop_metadata.json")
SUPPORTED_PATH = os.path.join(MODEL_DIR, "supported_crops.json")

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp", ".heic"}


def _load_json(path, fallback):
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return fallback


def _save_json(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def scan_images(folder):
    found = []
    for root, _dirs, files in os.walk(folder):
        for f in sorted(files):
            if os.path.splitext(f)[1].lower() in IMAGE_EXTS:
                found.append(os.path.join(root, f))
    return found


def validate_image(path):
    try:
        with Image.open(path) as im:
            im.verify()
        with Image.open(path) as im:
            im.load()
            w, h = im.size
        return (w, h) if w >= 16 and h >= 16 else None
    except (UnidentifiedImageError, OSError, ValueError, SyntaxError):
        return None


def file_hash(path, chunk=1024 * 1024):
    h = hashlib.md5()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description="Add a new plant/crop to the "
                                             "identifier catalog.")
    ap.add_argument("crop", help='Crop display name, e.g. "Wheat"')
    ap.add_argument("source", nargs="?", default=None,
                    help="Folder with the crop's images (else datasets/raw/<Crop>)")
    ap.add_argument("--diseases", default="",
                    help="Comma-separated disease class names if a disease model "
                         "will be trained (default: none -> identifiable only)")
    args = ap.parse_args()

    crop = args.crop.strip().title()
    if not crop:
        raise SystemExit("No crop name given.")

    source = args.source or os.path.join(RAW_ROOT, crop)
    if not os.path.isdir(source):
        raise SystemExit(f"Source folder not found: {source}")

    images = scan_images(source)
    if not images:
        raise SystemExit(f"No images under {source}")

    # Validate + dedupe
    valid, seen, dup, corrupt = [], set(), 0, 0
    for p in images:
        if validate_image(p):
            h = file_hash(p)
            if h in seen:
                dup += 1
                continue
            seen.add(h)
            valid.append(p)
        else:
            corrupt += 1
    if not valid:
        raise SystemExit("No valid images found - nothing added.")

    # Copy into plant_identification/<Crop>/<Class-or-images>/ (dedup by hash)
    dst_root = os.path.join(PLANT_ID_ROOT, crop)
    copied = 0
    for p in valid:
        rel = os.path.relpath(p, source)
        parts = [x for x in rel.split(os.sep)][:-1]
        sub = parts[0] if len(parts) == 1 else "_".join(parts)
        sub = sub if sub else "images"
        d = os.path.join(dst_root, sub)
        os.makedirs(d, exist_ok=True)
        dst = os.path.join(d, f"{file_hash(p)[:10]}_{os.path.basename(p)}")
        if not os.path.exists(dst):
            shutil.copyfile(p, dst)
        copied += 1

    # Register in crop metadata + supported crops (identifiable; disease model
    # only when the user confirms one will be trained).
    meta = _load_json(CROP_META_PATH, {})
    diseases = [d.strip() for d in args.diseases.split(",") if d.strip()]
    entry = meta.get(crop, {})
    entry["diseases"] = diseases
    entry.setdefault("healthy", diseases == [])
    entry["identity_images"] = copied
    entry["extended_at"] = datetime.now().strftime("%Y-%m-%d %H:%M")
    meta[crop] = entry
    _save_json(CROP_META_PATH, meta)

    supported = _load_json(SUPPORTED_PATH, {})
    supported[crop] = bool(diseases)
    _save_json(SUPPORTED_PATH, supported)

    print(f"[OK] Added crop '{crop}':")
    print(f"      images copied to datasets/plant_identification/{crop}/ : {copied}")
    print(f"      (corrupt dropped: {corrupt}, duplicates dropped: {dup})")
    print(f"      registered in models/crop_metadata.json + supported_crops.json")
    if diseases:
        print(f"      disease model flagged with classes: {diseases}")
    else:
        print("      status: identifiable (Unknown/Unsupported until a disease model is trained)")
    print()
    print("Next step - retrain the Stage-2 identifier to learn this crop:")
    print('      run_train_identifier.cmd')
    print('   or: python training/train_plant_identifier.py --epochs 14')


if __name__ == "__main__":
    main()