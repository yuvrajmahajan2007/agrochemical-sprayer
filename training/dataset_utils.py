"""
AI-Driven Precision Agrochemical Sprayer - Training dataset utilities

Scans dataset folders used by the multi-stage pipeline, validates the folder
structure (class folder -> images), removes corrupted images, standardizes
labels and reports what is really available.

Expected layout:

    datasets/plant_validation/     plant/  non_plant/        (optional)
    datasets/plant_identification/ <Crop>/  <Class>/...      (optional)
    datasets/disease/<crop>/       <Class>/...               (optional)

Fallback sources already on disk (PlantVillage-derived):
    datasets/processed/<Crop>/<Class>/...

No fake/mock data is ever generated.
"""

import os
import json
import shutil

import numpy as np
from PIL import Image

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASETS_ROOT = os.path.join(PROJECT_ROOT, "datasets")

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
IMG_SIZE = 224


def _is_image(name):
    return os.path.splitext(name)[1].lower() in IMAGE_EXTS


def scan_folder_images(folder):
    """List valid, loadable images under a folder (class folders allowed)."""
    found = []
    for root, _dirs, files in os.walk(folder):
        for f in sorted(files):
            if not _is_image(f):
                continue
            path = os.path.join(root, f)
            if _validate_image(path):
                found.append(path)
    return found


def _validate_image(path):
    """Cheap integrity check - returns True when the image can be opened."""
    try:
        img = Image.open(path)
        img.verify()
        with Image.open(path) as im:
            im.load()
        return True
    except Exception:
        try:
            os.remove(path)  # drop corrupted image so it cannot poison training
        except OSError:
            pass
        return False


def remove_corrupted(folder):
    removed = 0
    for root, _dirs, files in os.walk(folder):
        for f in files:
            if not _is_image(f):
                continue
            path = os.path.join(root, f)
            if not _validate_image(path):
                removed += 1
    return removed


def load_plant_identification(processed_fallback=True,
                              use_processed=True):
    """
    Build the plant-identification dataset:
    { "Crop": [image_paths...] }

    Prefers datasets/plant_identification/<Crop>/<Class>/.
    Falls back to grouping the existing processed (real, PlantVillage/Rice)
    dataset by crop so the identifier can be trained without new downloads.
    """
    result = {}
    id_root = os.path.join(DATASETS_ROOT, "plant_identification")
    if os.path.isdir(id_root):
        for crop in os.listdir(id_root):
            crop_dir = os.path.join(id_root, crop)
            if not os.path.isdir(crop_dir):
                continue
            imgs = scan_folder_images(crop_dir)
            if imgs:
                result[crop] = sorted(set(result.get(crop, [])) | set(imgs))

    if processed_fallback and use_processed:
        proc = os.path.join(DATASETS_ROOT, "processed")
        if os.path.isdir(proc):
            for crop in os.listdir(proc):
                crop_dir = os.path.join(proc, crop)
                if not os.path.isdir(crop_dir):
                    continue
                imgs = scan_folder_images(crop_dir)
                if imgs:
                    result[crop] = sorted(set(result.get(crop, [])) | set(imgs))
    return result


def load_plant_validation():
    """{ "plant": [paths], "non_plant": [paths] } - only from real folders."""
    root = os.path.join(DATASETS_ROOT, "plant_validation")
    out = {}
    if not os.path.isdir(root):
        return out
    for cls in ("plant", "non_plant"):
        d = os.path.join(root, cls)
        if os.path.isdir(d):
            imgs = scan_folder_images(d)
            if imgs:
                out[cls] = imgs
    return out


def load_disease_sets(crops=None):
    """
    Per-crop disease datasets:
    { crop: { "ClassName": [paths...] } }
    Sources: datasets/disease/<crop>/ first, else datasets/processed/<crop>/.
    """
    out = {}
    wanted = set(crops or [])
    for crop in sorted(os.listdir(os.path.join(DATASETS_ROOT, "processed"))):
        if wanted and crop not in wanted:
            continue
        d = os.path.join(DATASETS_ROOT, "processed", crop)
        if not os.path.isdir(d):
            continue
        classes = {}
        for cls in os.listdir(d):
            cd = os.path.join(d, cls)
            if not os.path.isdir(cd):
                continue
            imgs = scan_folder_images(cd)
            if imgs:
                classes[cls] = imgs
        if classes:
            out[crop] = classes

    dis_root = os.path.join(DATASETS_ROOT, "disease")
    if os.path.isdir(dis_root):
        for crop in os.listdir(dis_root):
            cd = os.path.join(dis_root, crop)
            if not os.path.isdir(cd):
                continue
            if wanted and crop not in wanted:
                continue
            classes = {}
            for cls in os.listdir(cd):
                ccd = os.path.join(cd, cls)
                if not os.path.isdir(ccd):
                    continue
                imgs = scan_folder_images(ccd)
                if imgs:
                    classes[cls] = imgs
            if classes:
                out[crop] = classes
    return out


def write_json(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def dataset_report():
    """Human-readable report of what is actually on disk."""
    report = {}
    idents = load_plant_identification()
    report["plant_identification_crops"] = {
        c: len(imgs) for c, imgs in idents.items()}
    report["plant_validation"] = {
        c: len(imgs) for c, imgs in load_plant_validation().items()}
    report["disease"] = {
        c: {cls: len(imgs) for cls, imgs in classes.items()}
        for c, classes in load_disease_sets().items()}
    return report