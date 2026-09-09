# Models

The trained **model binaries** (`.keras`, `.h5`, `.tflite`, ...) are **not**
committed to GitHub — they are large (each is ~11–30 MB) and are either
trained from the real dataset in this repo or reproduced by running the
included training scripts.

This `models/README.md` is tracked so a fresh clone knows exactly what to
place where.

---

## Required model files

The backend loads models from the folders below at startup. If a model file
is missing, the backend logs a warning and either:

- uses a deterministic **pixel-segmentation fallback** for the leaf-health
  analysis (so the demo still works), or
- reports `Analysis Not Available` / `Low Plant Confidence` honestly — it
  **never** guesses a result.

| Model | Path (relative to repo root) | Purpose | Size |
|---|---|---|---|
| Plant/Leaf validator | `models/plant_validator/plant_validator.keras` | Stage 1: real Plant vs Non-Plant gate | ~11 MB |
| Leaf-health U-Net segmenter | `models/leaf_segmenter/leaf_segmenter.keras` | Stage 2: background / healthy / symptomatic segmentation (severity % from its masks) | ~29 MB |
| *Legacy* combined classifier | `models/plant_disease_model.keras` | 41-class crop+disease classifier (legacy path) | ~24 MB |
| *Optional* plant identifier | `models/plant_identifier/plant_identifier.keras` | Legacy crop identification (no longer required by the new pipeline) | ~23 MB |

Small JSON metadata files (e.g. `class_names.json`, `supported_crops.json`,
`crop_metadata.json`) **are** committed — they are tiny and describe the
model structure without being binaries.

---

## How to get each model

### Option A - Train them yourself (recommended, fully reproducible)

1. Setup the datasets (see `datasets/README.md`).
2. Train the Plant vs Non-Plant validator:

   ```powershell
   .\run_train_validator.cmd
   ```

   (builds the real plant/non-plant dataset, trains MobileNetV2, evaluates,
   and chooses the operating threshold).

3. Train the leaf-health U-Net segmenter:

   ```powershell
   .\run_segmenter.cmd
   ```

   (builds real pixel masks from real leaf photos, trains the U-Net, prints
   held-out IoU, and saves to `models/leaf_segmenter/`).

4. *(Legacy, optional)* Train the combined 41-class classifier:

   ```powershell
   .\run_training.cmd
   ```

Models are written to `models/` automatically by each trainer, alongside
their `metadata.json` / `class_names.json`.

### Option B - Download a pre-trained build

Releases of this project may attach the trained `.keras` files to a
[GitHub Release](https://github.com/) page. If a release is available:

1. Download the `.keras` files.
2. Place them into the folders above:

```
models/
  plant_validator/plant_validator.keras
  leaf_segmenter/leaf_segmenter.keras
```

If you prefer to distribute/share the trained weights yourself (e.g. via a
release, Google Drive, or a private URL), place the weights in the exact
paths listed in the table above and the backend will pick them up on restart.

---

## Requirements for training

- Python 3.10+ (see `README.md`).
- `tensorflow` + the rest of `requirements.txt` installed.
- The datasets prepared locally (see `datasets/README.md`).
- Reasonable CPU (recommend 8+ GB RAM; training is CPU-only). GPU optional.
