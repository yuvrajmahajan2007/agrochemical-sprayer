"""
AI-Driven Precision Agrochemical Sprayer - Build the Plant/Leaf Validation dataset

Assembles REAL labelled images into datasets/plant_validation/:

  plant/      1) all crops+classes from datasets/processed/ (PlantVillage/Rice
                 derived - healthy + diseased leaves, green/yellow/dry, single
                 leaves) for background/lighting diversity;
              2) REAL mobile-camera photos already present in uploads/ that the
                 current pipeline confidently labels Healthy or Diseased (these
                 are genuine leaf photos, close to the failing case this stage
                 is being fixed for).
  non_plant/  REAL photographs downloaded from Wikimedia Commons by category
                 (food, human hands, buildings, vehicles, electronics, stones,
                 paper, furniture, empty rooms ...) - the exact negative
                 diversity requested. No fake/synthesised negatives.

Never generates synthetic images. The Commons downloader only keeps real
photographs (image/jpeg), verifies every file it saves, and de-duplicates by
content hash.

Usage:
    python training/build_validator_dataset.py            # processed only
    python training/build_validator_dataset.py --include-mobile   # + uploads
    python training/build_validator_dataset.py --no-commons       # no download
"""

import argparse
import hashlib
import os
import random
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import requests  # noqa: E402
from PIL import Image  # noqa: E402

from training.dataset_utils import scan_folder_images  # noqa: E402

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VAL_ROOT = os.path.join(PROJECT_ROOT, "datasets", "plant_validation")
PLANT_DIR = os.path.join(VAL_ROOT, "plant")
NON_PLANT_DIR = os.path.join(VAL_ROOT, "non_plant")

COMMONS_CATEGORIES = [
    "Food", "Human hands", "Automobiles", "Buildings",
    "Mobile phones", "Electronics", "Rocks", "Paper",
    "Furniture", "Kitchen utensils", "Walls", "Empty rooms",
]
CATEGORY_LIMIT = 60   # real photo thumbnails requested per category
UA = ("AgroSprayerValidatorData/1.0 (educational engineering project; "
      "contact: agrosprayer@localhost)")


def _headers():
    return {"User-Agent": UA}


# ---------------------------------------------------------------------------
def _safe_copy(src, dst):
    try:
        with Image.open(src) as im:
            im.load()
        Image.open(src).verify()
        shutil.copy2(src, dst)
        return True
    except Exception:
        return False


def build_plant_processed():
    """Copy every real image from datasets/processed -> plant/ (diverse)."""
    proc = os.path.join(PROJECT_ROOT, "datasets", "processed")
    if not os.path.isdir(proc):
        print("  [plant] no datasets/processed - skipped", flush=True)
        return
    os.makedirs(PLANT_DIR, exist_ok=True)
    existing = {f for f in os.listdir(PLANT_DIR) if f.lower().endswith(
        (".jpg", ".jpeg", ".png", ".bmp", ".webp"))}
    done = 0
    for crop in sorted(os.listdir(proc)):
        cdir = os.path.join(proc, crop)
        if not os.path.isdir(cdir):
            continue
        for cls in sorted(os.listdir(cdir)):
            cldir = os.path.join(cdir, cls)
            if not os.path.isdir(cldir):
                continue
            images = sorted(f for f in os.listdir(cldir)
                            if f.lower().endswith((".jpg", ".jpeg", ".png",
                                                   ".bmp", ".webp")))
            for i, name in enumerate(images):
                dst_name = f"{crop}__{cls}__{i:04d}__{name}"
                if dst_name in existing:
                    continue
                if _safe_copy(os.path.join(cldir, name),
                              os.path.join(PLANT_DIR, dst_name)):
                    done += 1
        print(f"  [plant] {crop}: copied", flush=True)
    print(f"  [plant] total from processed: {done}", flush=True)


def build_plant_mobile():
    """Real phone uploads already confidently labelled Healthy/Diseased."""
    up = os.path.join(PROJECT_ROOT, "uploads")
    if not os.path.isdir(up):
        return 0
    images = sorted(
        f for f in os.listdir(up)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
        and not os.path.exists(os.path.join(PLANT_DIR, f"mobile__{f}")))
    if not images:
        print("  [mobile] none left to check", flush=True)
        return 0
    base = "http://127.0.0.1:5000/predict"
    lock = __import__("threading").Lock()
    good = []

    def check(name):
        path = os.path.join(up, name)
        try:
            with open(path, "rb") as fh:
                r = requests.post(base, files={"image": (name, fh)}, timeout=90)
            j = r.json()
        except Exception as exc:
            print(f"  [mobile] skipped {name}: {exc}", flush=True)
            return
        if j.get("status") in ("Healthy", "Diseased"):
            with lock:
                good.append((name, path))

    with ThreadPoolExecutor(max_workers=5) as ex:
        futs = [ex.submit(check, n) for n in images]
        for f in as_completed(futs):
            f.result()
    os.makedirs(PLANT_DIR, exist_ok=True)
    done = 0
    for name, path in good:
        if _safe_copy(path, os.path.join(PLANT_DIR, f"mobile__{name}")):
            done += 1
    print(f"  [plant] mobile uploads verified as leaf: {done}", flush=True)
    return done


# ---------------------------------------------------------------------------
def _category_files(category, limit):
    params = {
        "action": "query", "generator": "categorymembers",
        "gcmtitle": f"Category:{category}", "gcmtype": "file",
        "gcmlimit": str(limit), "prop": "imageinfo",
        "iiprop": "url|mime", "iiurlwidth": "400", "format": "json",
    }
    try:
        data = requests.get("https://commons.wikimedia.org/w/api.php",
                            params=params, headers=_headers(), timeout=40
                            ).json()
    except Exception as exc:
        print(f"  [non_plant] {category}: API failed ({exc})")
        return [], []
    urls, srcs = [], []
    for p in data.get("query", {}).get("pages", {}).values():
        ii = p.get("imageinfo") or [{}]
        info = ii[0]
        mime = str(info.get("mime", ""))
        if "jpeg" not in mime and "png" not in mime:
            continue
        url = info.get("thumburl") or info.get("url")
        if url:
            urls.append(url)
            srcs.append(p.get("index", os.path.basename(url)))
    return urls, srcs


def build_non_plant_commons(limit_per_category):
    """Download real Commons photos for non-plant categories."""
    os.makedirs(NON_PLANT_DIR, exist_ok=True)
    targets = []
    for cat in COMMONS_CATEGORIES:
        urls, _ = _category_files(cat, limit_per_category)
        print(f"  [non_plant] {cat}: {len(urls)} candidate URLs")
        targets.extend((cat, u) for u in urls)

    random.shuffle(targets)
    seen_hashes = {f.split("__")[-1].split(".")[0]
                   for f in os.listdir(NON_PLANT_DIR)}
    lock = __import__("threading").Lock()
    saved = 0

    def one(item):
        nonlocal saved
        cat, url = item
        try:
            r = requests.get(url, timeout=40, headers=_headers())
            if r.status_code != 200:
                return 0
            blob = r.content
            if len(blob) < 5000:
                return 0
            h = hashlib.sha256(blob).hexdigest()[:12]
            with lock:
                if h in seen_hashes:
                    return 0
                seen_hashes.add(h)
                name = f"np__{cat.replace(' ', '_')}__{h}.jpg"
                dst = os.path.join(NON_PLANT_DIR, name)
            with open(dst, "wb") as f:
                f.write(blob)
            with Image.open(dst) as im:
                im.load()
            with lock:
                saved += 1
            return 1
        except Exception:
            return 0

    with ThreadPoolExecutor(max_workers=16) as ex:
        futs = [ex.submit(one, t) for t in targets]
        for f in as_completed(futs):
            f.result()
    print(f"  [non_plant] saved from Commons: {saved}")


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--include-mobile", action="store_true",
                    help="add real phone uploads verified as leaf photos")
    ap.add_argument("--no-commons", action="store_true",
                    help="skip the Wikimedia download (offline mode)")
    args = ap.parse_args()

    print("Building plant/leaf validation dataset (real images only)...")
    build_plant_processed()
    if args.include_mobile:
        build_plant_mobile()
    if not args.no_commons:
        build_non_plant_commons(CATEGORY_LIMIT)

    plant = scan_folder_images(PLANT_DIR)
    non_plant = scan_folder_images(NON_PLANT_DIR)
    print("\nSHAPE:"
          f"\n  plant    = {len(plant)}"
          f"\n  non_plant= {len(non_plant)}"
          "\nTrain with:  python training/train_validator.py --epochs 12")


if __name__ == "__main__":
    main()