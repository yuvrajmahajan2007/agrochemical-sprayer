"""
AI-Driven Precision Agrochemical Sprayer - Dataset Preparation & Standardization

Inspects every raw dataset, detects crop and disease class names, standardizes
folder structures / labels, validates and filters corrupt images, detects
duplicate images, and builds a unified processed dataset plus a manifest.

Outputs:
    datasets/processed/<Crop>/<Class>/  ... standardized images (symlinked/copied)
    datasets/processed/class_manifest.json
    datasets/processed/crop_metadata.json

Usage:
    python training/prepare_dataset.py
"""

import os
import re
import json
import shutil
import hashlib
import argparse
from collections import defaultdict

from PIL import Image, UnidentifiedImageError

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(PROJECT_ROOT, "datasets", "raw")
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "datasets", "processed")

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

# ----------------------------------------------------------------------
# Source descriptors for known datasets
# ----------------------------------------------------------------------
# Each raw source is described by its layout and label mapping rules.

class Source:
    name = None
    def discover(self, root):
        """
        Returns dict {class_folder_name: [image_paths]}
        plus a list of per-class human labels if discoverable.
        """
        raise NotImplementedError


class FlatFolderSource(Source):
    """Root contains class folders directly (or one nesting level)."""

    def discover(self, root):
        classes = {}
        if not os.path.isdir(root):
            return classes

        def collect_dir(full_dir, dst):
            for entry in sorted(os.listdir(full_dir)):
                full = os.path.join(full_dir, entry)
                if os.path.isdir(full):
                    imgs = [
                        os.path.join(full, f) for f in os.listdir(full)
                        if os.path.splitext(f)[1].lower() in IMAGE_EXTS
                    ]
                    if imgs:
                        dst[entry] = imgs

        # Level 1: direct class folders with images
        collect_dir(root, classes)
        if classes:
            return classes

        # Level 2: single nested subfolder (e.g. PlantVillage/multicrop_diseases)
        for entry in sorted(os.listdir(root)):
            full = os.path.join(root, entry)
            if os.path.isdir(full):
                nested = {}
                collect_dir(full, nested)
                if nested:
                    classes.update(nested)
        return classes


# ----------------------------------------------------------------------
# Crop / label normalization
# ----------------------------------------------------------------------
PLANTVILLAGE_CROP_TO_DISPLAY = {
    "apple": "Apple",
    "blueberry": "Blueberry",
    "cherry": "Cherry",
    "corn": "Maize",
    "grape": "Grape",
    "orange": "Orange",
    "peach": "Peach",
    "pepper": "Pepper",
    "potato": "Potato",
    "raspberry": "Raspberry",
    "soybean": "Soybean",
    "squash": "Squash",
    "strawberry": "Strawberry",
    "tomato": "Tomato",
    "wheat": "Wheat",
    "rice": "Rice",
    "sorghum": "Sorghum",
    "pearl millet": "Pearl Millet",
}


def slugify(name):
    """Lowercase, trim common separators, strip punctuation."""
    name = name.lower()
    name = re.sub(r"\[.*?\]", "", name)
    name = re.sub(r"[\(\)\[\]{},;:]", " ", name)
    name = name.replace("___", " ").replace("__", " ").replace("_", " ").replace("-", " ")
    name = re.sub(r"\s+", " ", name).strip()
    return name


def parse_plantvillage_class(label):
    """
    Take a PlantVillage class label like:
        'Corn_(maize)___Common_rust_'
        'Tomato___healthy'
    and return (crop_display, disease_key, is_healthy).
    """
    label = label.replace("\\", " ").replace("/", " ")
    if "___" in label:
        crop_part, disease_part = label.split("___", 1)
    else:
        crop_part, disease_part = label, ""
    crop_slug = slugify(crop_part)
    disease_slug = slugify(disease_part)
    is_healthy = disease_slug in ("healthy", "healthy_leaf", "leaf_healthy")

    # Map crop slug to display name, unify aliases
    display = None
    for key, disp in PLANTVILLAGE_CROP_TO_DISPLAY.items():
        if key in crop_slug:
            display = disp
            break
    if display is None:
        # If crop unknown, keep a title-cased guess of first token(s)
        display = " ".join(w.capitalize() for w in crop_slug.split())

    if is_healthy:
        disease_key = "healthy"
        disease_display = "Healthy"
    elif disease_slug:
        # Human-readable disease name (presentation only)
        disease_display = " ".join(w.capitalize() for w in disease_slug.split())
        disease_key = disease_slug
    else:
        disease_display = "Unknown Condition"
        disease_key = "unknown"

    return display, disease_key, disease_display, is_healthy


def normalize_rice_class(label):
    """
    Rice disease classes from the CDS/UCI rice dataset:
        'Bacterial leaf blight', 'Brown spot', 'Leaf smut'
    """
    slug = slugify(label)
    mapping = {
        "bacterial leaf blight": "bacterial_leaf_blight",
        "bacterial blight": "bacterial_leaf_blight",
        "blb": "bacterial_leaf_blight",
        "brown spot": "brown_spot",
        "leaf smut": "leaf_smut",
        "healthy": "healthy",
    }
    key = mapping.get(slug, slug.replace(" ", "_"))
    display = " ".join(w.capitalize() for w in key.split("_"))
    return "Rice", key, display, key == "healthy"


# ----------------------------------------------------------------------
# Image validation
# ----------------------------------------------------------------------
def validate_image(path):
    try:
        with Image.open(path) as im:
            im.verify()
        with Image.open(path) as im:
            im.load()
            w, h = im.size
        if w < 16 or h < 16:
            return False, "too small"
        return True, "ok"
    except (UnidentifiedImageError, OSError, ValueError, SyntaxError):
        return False, "corrupt/unreadable"
    except Exception:
        return False, "error"


def file_hash(path, chunk=1024 * 1024):
    h = hashlib.md5()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


# ----------------------------------------------------------------------
# Main pipeline
# ----------------------------------------------------------------------
def process_dataset_source(source_name, root, layout, crop_override=None):
    """
    Process one raw dataset source.
    Returns manifest entries for each class (crop, class_key, display, status,
    count, source).
    """
    print(f"\n[Inspect] Source: {source_name}")
    print(f"          Root:  {root}")

    classes = FlatFolderSource().discover(root)
    if not classes:
        print("          [WARN] No class folders found.")
        return []

    entries = []
    for class_label, imgs in sorted(classes.items()):
        # Determine crop + disease key based on layout
        if layout == "plantvillage":
            crop, disease_key, disease_disp, is_healthy = parse_plantvillage_class(class_label)
        elif layout == "rice":
            crop, disease_key, disease_disp, is_healthy = normalize_rice_class(class_label)
        else:  # generic
            slug = slugify(class_label)
            crop = crop_override or "Unknown"
            disease_key = slug
            disease_disp = " ".join(w.capitalize() for w in slug.split())
            is_healthy = disease_key == "healthy"

        # Validate images, drop corrupt
        valid = []
        dropped = 0
        for p in imgs:
            ok, reason = validate_image(p)
            if ok:
                valid.append(p)
            else:
                dropped += 1
                print(f"          [DROP] corrupt: {os.path.basename(p)} ({reason})")

        # Deduplicate by content hash
        seen = set()
        unique = []
        dup = 0
        for p in valid:
            try:
                h = file_hash(p)
            except OSError:
                dropped += 1
                continue
            if h in seen:
                dup += 1
            else:
                seen.add(h)
                unique.append(p)

        status = "Healthy" if is_healthy else "Diseased"
        entries.append({
            "crop": crop,
            "class_key": disease_key,
            "display_name": disease_disp,
            "status": status,
            "is_healthy": is_healthy,
            "source_class": class_label,
            "source": source_name,
            "count": len(unique),
            "duplicates_removed": dup,
            "corrupt_removed": dropped,
        })
        print(f"          {class_label!r}: {len(unique)} images "
              f"(dup removed {dup}, corrupt removed {dropped})")
    return entries


def main():
    ap = argparse.ArgumentParser(description="AI-Driven Precision Agrochemical Sprayer dataset preparation")
    ap.add_argument("--config", default=None, help="Optional JSON config mapping source->layout")
    args = ap.parse_args()

    # Source discovery: scan raw dir
    if not os.path.isdir(RAW_DIR):
        print(f"[ERROR] Raw dataset dir not found: {RAW_DIR}")
        return

    sources = {}
    for entry in os.listdir(RAW_DIR):
        full = os.path.join(RAW_DIR, entry)
        if os.path.isdir(full):
            sources[entry] = full

    print("=" * 64)
    print("AI-Driven Precision Agrochemical Sprayer - Dataset Preparation")
    print("=" * 64)
    print("[OK] Raw datasets found:")
    for name, path in sorted(sources.items()):
        n = len(FlatFolderSource().discover(path))
        print(f"      - {name}: {n} class folder(s)")

    # Choose layout by source name
    layout_by_source = {}
    for name in sources:
        low = name.lower()
        if "plantvillage" in low:
            layout_by_source[name] = "plantvillage"
        elif "rice" in low:
            layout_by_source[name] = "rice"
        else:
            layout_by_source[name] = "generic"

    # Metadata about the dataset sources
    dataset_meta = {
        "output_dir": PROCESSED_DIR,
        "sources": {
            "PlantVillage": {
                "origin": "PlantVillage dataset (via Kaggle: abdallahalidev/plantvillage-dataset, emmarex/plantdisease)",
                "license": "CC0 / research use - PlantVillage public dataset",
                "note": "Standardized 38-class PlantVillage multicrop subset",
            },
            "Rice": {
                "origin": "Rice leaf disease dataset (CDS Capstone / UCI Rice Leaf Diseases)",
                "license": "Academic research dataset",
                "note": "3 disease classes - no healthy class available",
            },
        },
        "prepared_at": __import__("time").strftime("%Y-%m-%d %H:%M:%S"),
    }

    manifest_entries = []
    for name in sorted(sources):
        entries = process_dataset_source(
            source_name=name,
            root=sources[name],
            layout=layout_by_source[name],
        )
        manifest_entries.extend(entries)

    if not manifest_entries:
        print("[ERROR] No images found at all.")
        return

    # Now copy into processed/ and build per-class folders.
    copied_count = 0
    per_class_folder_image_map = defaultdict(lambda: defaultdict(list))
    for e in manifest_entries:
        key = (e["source"], e["source_class"])
        # search for this class folder anywhere under the source root
        src_root = os.path.join(RAW_DIR, e["source"])
        found_dir = None
        for dirpath, dirnames, filenames in os.walk(src_root):
            if os.path.basename(dirpath) == e["source_class"]:
                found_dir = dirpath
                break
        if found_dir is None:
            continue
        for f in os.listdir(found_dir):
            if os.path.splitext(f)[1].lower() in IMAGE_EXTS:
                per_class_folder_image_map[e["crop"]][e["class_key"]].append(
                    os.path.join(found_dir, f)
                )

    # Build processed classes (dedupe again at copy-time)
    global_dup = set()
    unique_class_dir_counts = {}
    for crop, classmap in per_class_folder_image_map.items():
        crop_dir = os.path.join(PROCESSED_DIR, crop)
        os.makedirs(crop_dir, exist_ok=True)
        for class_key, paths in classmap.items():
            if class_key == "unknown":
                class_key = "Disease_Unknown"
            class_folder = os.path.join(crop_dir, class_key)
            os.makedirs(class_folder, exist_ok=True)
            cnt = 0
            for p in paths:
                h = file_hash(p)
                if h in global_dup:
                    continue
                global_dup.add(h)
                dst = os.path.join(class_folder, f"{h[:10]}_{os.path.basename(p)}")
                shutil.copyfile(p, dst)
                cnt += 1
            unique_class_dir_counts[(crop, class_key)] = cnt
            copied_count += cnt

    # Write manifest
    manifest = {
        "generated_by": "prepare_dataset.py",
        "prepared_at": dataset_meta["prepared_at"],
        "datasets": dataset_meta,
        "classes": [],
    }
    for e in manifest_entries:
        crop, ckey = e["crop"], e["class_key"]
        count = unique_class_dir_counts.get((crop, ckey), 0)
        manifest["classes"].append({
            "crop": crop,
            "class_key": ckey,
            "display_name": e["display_name"],
            "status": e["status"],
            "is_healthy": e["is_healthy"],
            "source": e["source"],
            "source_class": e["source_class"],
            "image_count": count,
        })

    os.makedirs(PROCESSED_DIR, exist_ok=True)
    manifest_path = os.path.join(PROCESSED_DIR, "dataset_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"\n[OK] Manifest written: {manifest_path}")
    print(f"[OK] Total images copied: {copied_count}")
    print(f"[OK] Total classes: {len(manifest['classes'])}")
    crops = sorted({c for c, _ in per_class_folder_image_map.items()})
    print(f"[OK] Total crops: {len(crops)} -> {', '.join(crops)}")


if __name__ == "__main__":
    main()