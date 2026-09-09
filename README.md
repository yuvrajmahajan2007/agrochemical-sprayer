# AI-Driven Precision Agrochemical Sprayer

**Engineering College Project** - a real, crop-agnostic AI system for plant health and
disease detection in agricultural crops. Built with a **crop-agnostic, health-first
pipeline** that works on ANY leaf photo without needing to identify the plant.

> **Key design principle:** the health result is driven by visible, real
> segmentation of the leaf area - never by guessed/random percentages and never by
> "AI confidence". If the model cannot analyse reliably, it honestly reports
> `Analysis Not Available` instead of fabricating a result.

The system analyzes plant/leaf images and returns one clear, **health-first** answer:

- **🟢 PLANT IS HEALTHY** or **🔴 DISEASE DETECTED**
- **`Analysis Not Available`** when the model cannot reliably analyse the image (never guesses)
- Plus `Not a Plant` / `Low Plant Confidence` when the image never passes validation
- **Specific, labelled confidence values** - Plant Validation / Plant
  Identification / Health Analysis / Disease Detection - never a generic
  "Confidence"
- **Severity** (Low / Medium / High) from real leaf-area segmentation
- **HEALTH STATUS** is the main answer; **severity is the affected leaf area**
  calculated from real pixel segmentation, **never disease confidence**:

  | Severity | Affected leaf area |
  |---|---|
  | 🟢 Low | 0 - 10% |
  | 🟡 Medium | > 10% - 40% |
  | 🔴 High | > 40% |

  Thresholds are configurable via `SPRAYBOT_SEVERITY_LOW` / `SPRAYBOT_SEVERITY_MED`.
  If the segmentation is unstable the analyzer honestly reports
  *"Severity: analysis unavailable"* (`severity_available: false`) instead of
  inventing a number.
- **Real AI Detection Overlay**: when segmentation succeeds, the response also
  carries `overlay_data_uri` - the uploaded leaf with the actual affected region
  highlighted in red from the same segmentation mask (never a fake overlay).

Crop identification is used **internally** to select the correct disease model; the
user is only shown the crop name as secondary information when it is confidently known.

The `/predict` API is used by both the **mobile phone camera** (current test rig)
and the future **ESP32-CAM rover** (final prototype), so the AI backend is fully
camera-independent.

---

## Features

- **Crop-agnostic, health-first pipeline** - no plant identification required;
  any leaf photo goes through the same steps.
- **Real multi-stage AI analysis**
  - **Stage 1 - Plant / Leaf validation** using a real trained
    Plant-vs-Non-Plant MobileNetV2 (rejects hands, buildings, desks, photos ...).
  - **Stage 2 - Health segmentation** with a real pixel-wise **U-Net** that
    separates background / healthy leaf / symptomatic tissue.
- **AI severity analysis** from the real measured affected leaf area
  (`affected_area_pct = symptomatic / (healthy + symptomatic)`), mapped to
  **Low / Medium / High** with configurable thresholds.
- **Spray decision support** - MONITOR / INSPECTION RECOMMENDED /
  TREATMENT ASSESSMENT RECOMMENDED (never auto-prescribes a pesticide).
- **Real detection overlay** - the uploaded leaf with only the model-marked
  symptomatic region highlighted in red (never a fake overlay).
- **Honest "no guess" behaviour** - `Analysis Not Available`, `Low Plant
  Confidence` and `Not a Plant` are reported truthfully when the models cannot
  make a reliable determination.
- **Mobile camera testing** - mobile-friendly web app with rear-camera capture
  (`capture="environment"`), auto-detected backend URL over Wi-Fi, and
  multi-language UI (English / हिंदी / मराठी).
- **Reusable training pipeline** - every script rebuilds real datasets and re-trains
  real models (validator, segmenter, classifier); nothing is faked.

---

## Honest Capability Report (read first)

This is a REAL AI system built on **real labeled images only**. No mock
predictions, no randomly assigned healthy/diseased results, no
"spray / don't spray" advice invented beyond what the model actually detects.

**Unknown/unsupported inputs are never forced into a disease class.**

### Real datasets included (on disk)

| Crop | Images | Disease classes | Source |
|---|---|---|---|
| Tomato | 2000 | 9 diseases + Healthy | PlantVillage |
| Grape | 800 | 3 diseases + Healthy | PlantVillage |
| Maize / Corn | 800 | 3 diseases + Healthy | PlantVillage |
| Apple | 800 | 3 diseases + Healthy | PlantVillage |
| Potato | 552 | 2 diseases + Healthy | PlantVillage |
| Cherry | 400 | Powdery Mildew + Healthy | PlantVillage |
| Peach | 400 | Bacterial Spot + Healthy | PlantVillage |
| Pepper (Bell) | 400 | Bacterial Spot + Healthy | PlantVillage |
| Strawberry | 400 | Leaf Scorch + Healthy | PlantVillage |
| Orange | 200 | Citrus Greening (HLB) | PlantVillage |
| Blueberry | 200 | Healthy only | PlantVillage |
| Raspberry | 200 | Healthy only | PlantVillage |
| Soybean | 200 | Healthy only | PlantVillage |
| Squash | 200 | Powdery Mildew | PlantVillage |
| Rice | 119 | 3 diseases | Rice Leaf Disease (CDS / UCI-derived) |
| **Total** | **7,671** | 41 classes / 15 crops | |

Disease-relevant crops (disease model available): Apple, Cherry, Grape, Maize,
Orange, Peach, Pepper, Potato, Rice, Squash, Strawberry, Tomato.
Healthy-only crops (no disease model exists, so health analysis honestly
reports *Analysis Not Available* for disease): Blueberry, Raspberry, Soybean.
Full per-class manifest:
`datasets/processed/dataset_manifest.json` and
`models/crop_metadata.json`.

### Priority crops NOT included (need your manual download)

Wheat, Sorghum, Pearl Millet, Cotton, Sugarcane, Onion, Garlic and Chilli have
**no real labeled images on disk**. The pipeline honestly reports them as
*Analysis Not Available* - it never fakes a result for them.

To add a crop with real data, see [Adding a new crop](#-adding-a-new-crop).

---

## System Architecture - health-first pipeline (crop-agnostic)

The redesign removes the requirement to identify the crop. Every leaf photo
goes through the SAME steps:

```
Input image
   |
   v
 Stage 1  Plant / Leaf Validation       -> Plant / Not a Plant / Low confidence
   (real trained Plant-vs-Non-Plant MobileNetV2; soft gate = CV + model)
   |   below 70% -> STOP: "Low Plant Confidence" (never guesses health)
   v
 Stage 2  Health Segmentation           (real U-Net, pixel-wise)
   (models/leaf_segmenter/leaf_segmenter.keras, 3 classes trained on real
    leaf photos via deterministic HSV/lesion pixel labels:
        class 0 = background, class 1 = healthy leaf, class 2 = symptomatic)
   |   if no confident leaf region -> "Analysis Not Available" (honest)
   v
 Affected Area % = symptomatic pixels / (healthy + symptomatic) pixels x 100
   v
 Health Status   = 0% -> Healthy ;  > 0% -> Diseased
   v
 Severity (from the REAL visible affected area only)
   (0, 10%] Low  |  (10, 40%] Medium  |  >40% High     <- configurable demo defaults)
   v
 Spray Decision Support (no auto-pesticide choice)
   Healthy/Low  -> MONITOR
   Medium       -> INSPECTION RECOMMENDED
   High         -> TREATMENT ASSESSMENT RECOMMENDED
   v
 Overlay = highlight of ONLY the pixels the model actually marked symptomatic
```

Golden rules:

- The**health classification depends on identifying symptoms, never on the crop**.
- The Affected-Area % comes from the real segmentation output - it is not an
  AI-conviction number and never a random value.
- If the segmentation cannot find a stable leaf region it is reported as
  "Analysis Not Available", never guessed.
- NO pesticide is ever recommended without a valid crop/disease source.

Confidence gating:

| Gate | Default | Effect |
|---|---|---|
| Plant/Leaf validation | 70% | below it -> STOP: `Low Plant Confidence` (segmentation never runs) |
| Severity LOW (demo)   | 10% | `0-10%` = Low        |
| Severity MEDIUM (demo)| 40% | `10-40%` = Medium, `>40%` = High |

The severity thresholds are **configurable demonstration defaults** - they
must be calibrated against real agricultural rules before production use
(see Environment configuration). Backend modules live in `backend/`:

- `config.py` - thresholds, paths, env config
- `plant_validator.py` - Stage 1 (real Plant vs Non-Plant model; CV fallback)
- `leaf_segmenter.py` - Stage 2 (real U-Net segmentation; deterministic pixel
  fallback if the model is not on disk; computes affected %, severity and the
  real overlay) - replaces the old `plant_identifier.py` /
  `disease_detector.py` / `severity_analyzer.py` crop-specific flow
- `pipeline.py` - orchestrator that builds the `pipeline[]` trace

---

## Project structure

```
.
|-- backend/                   (Flask app + pipeline stages)
|   |-- app.py                 (Flask server: /predict, /status, /api/config, ...)
|   |-- pipeline.py            (orchestrator: validator -> segmenter -> severity -> decision)
|   |-- config.py              (thresholds, paths, env config)
|   |-- plant_validator.py     (Stage 1: real Plant vs Non-Plant; CV fallback)
|   |-- leaf_segmenter.py      (Stage 2: U-Net segmentation -> affected % / severity / overlay)
|   |-- model_core.py, plant_identifier.py, disease_detector.py, severity_analyzer.py (legacy)
|   |-- utils.py
|-- frontend/                  (mobile web app served by Flask - HTML/CSS/JS)
|-- training/                  (all training/data-prep/eval scripts)
|   |-- prepare_dataset.py, dataset_utils.py
|   |-- build_validator_dataset.py, train_validator.py, evaluate_validator.py
|   |-- build_segmentation_masks.py, train_segmenter.py
|   |-- train_model.py, evaluate_models.py, train_plant_identifier.py, ...
|-- datasets/                  (NOT committed - see datasets/README.md)
|   |-- README.md
|-- models/                    (model BINARIES not committed - see models/README.md)
|   |-- README.md
|-- test_images/               (small sample leaf images for quick testing)
|-- requirements.txt
|-- run_backend.cmd  |  run_train_validator.cmd  |  run_segmenter.cmd  |  run_training.cmd
|-- .gitignore  |  .env.example  |  README.md
```
Datasets and uploads (runtime) are intentionally ignored; see `datasets/README.md`
and `models/README.md` for how to obtain data and trained weights.

---

## Quick start (laptop)

**Requirements**
- **Python 3.10 or 3.11** (project built and tested on 3.11).
- ~4 GB free disk, 8+ GB RAM recommended (TensorFlow training is CPU-only).
- A phone and laptop on the **same Wi-Fi** to test with the mobile camera.

**Option A - just run it (uses the deterministic fallback or your own models)**

```powershell
# 1. virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1

# 2. dependencies
pip install -r requirements.txt

# 3. (optional) copy config template
copy .env.example .env          # tweak values if you like; defaults work

# 4. start the backend + mobile web app on 0.0.0.0:5000
.\run_backend.cmd               # or: python backend\app.py
```

Then open `http://127.0.0.1:5000/` (laptop) or `http://<laptop-ip>:5000/`
(phone on the same Wi-Fi). Without trained models on disk, the backend uses a
deterministic pixel-segmentation fallback (`segmentation_source: pixel_fallback`)
so the demo still returns real, explainable results. See `models/README.md` for
how to supply or train the real U-Net segmenter and Plant-validator.

**Option B - train the REAL models from scratch (recommended for full capability)**

```powershell
# 1. virtual environment + deps (as above)
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt

# 2. prepare the dataset (see datasets/README.md for sources / download)
python training/prepare_dataset.py

# 3. train the REAL Stage-1 Plant vs Non-Plant validator
.\run_train_validator.cmd

# 4. train the REAL leaf-health U-Net segmenter (new crop-agnostic pipeline)
.\run_segmenter.cmd

# 5. evaluate held-out numbers
python training/evaluate_models.py --re-evaluate

# 6. start the backend
.\run_backend.cmd
```

---

## Dataset setup

The training image datasets are **not** committed (they are ~800 MB of real
leaf photos). Full instructions, sources (public **PlantVillage** dataset via
Kaggle) and the expected local folder layout are in
**[`datasets/README.md`](datasets/README.md)**.

Short version:

1. Download the PlantVillage dataset from Kaggle.
2. Place archives under `datasets/raw/`.
3. Run `python training/prepare_dataset.py` to standardize into
   `datasets/processed/<Crop>/<Class>/`.
4. (Segmenter) run `python training/build_segmentation_masks.py --n 1500`.
5. (Validator, needs internet) run
   `python training/build_validator_dataset.py --include-mobile`.

---

## Model setup

The trained **model binaries** are not committed either (each ~11-30 MB). Full
details (which model files, where they go, how to train or download them) are
in **[`models/README.md`](models/README.md)**.

Short version - train them yourself:

```powershell
.\run_train_validator.cmd    # Stage 1 Plant vs Non-Plant validator -> models/plant_validator/
.\run_segmenter.cmd          # Stage 2 leaf-health U-Net segmenter  -> models/leaf_segmenter/
```

If the model files are absent at startup, the backend uses a deterministic
pixel-segmentation fallback (`segmentation_source: pixel_fallback`) so the demo
still returns real, explainable results - it never fabricates a health/severity
answer.

---

## Stage-1 Plant vs Non-Plant validator (real trained model)

`models/plant_validator/plant_validator.keras` is a **real two-class model**
(MobileNetV2 ImageNet transfer learning) trained on real photographs only:

- `Plant_Leaf` (class 1): 7,735 real images (15 crops / healthy + diseased
  leaves from `datasets/processed/` plus real mobile-camera leaf uploads
  verified as `Healthy`/`Diseased`).
- `Non_Plant` (class 0): 245 real photographs downloaded from Wikimedia
  Commons (food, human hands, buildings, vehicles, phones, electronics,
  rocks, paper, furniture, walls, books, bottles, clothing, glassware ...).

Training split 70/15/15 with realistic augmentation (crop/zoom/shift,
90°-rotations, flips, ±20% brightness, 0.8-1.2 contrast, mild blur) and
balanced oversampling of the rarer negatives. Results on the held-out test
split (1,197 images): **train acc 99.79% · val acc 99.58% · test acc 99.92%**
(precision 0.9991, recall 1.0, F1 0.9996; the one test error was a
Non-Plant classified as Plant). The threshold sweep in
`training/evaluate_validator.py` showed equal FPR (2.7%) from 50-68%, so the
deployed gate was **kept at 70%** (`SPRAYBOT_PLANT_THRESHOLD`) - no blind
lowering. The live `/predict` Stage-1 confidence is now the real softmax
`Plant_Leaf` probability from this model.

Retrain/rebuild everything:

```powershell
# 1. rebuild the real dataset  (needs internet for the Commons negatives;
#    uploads are verified through the running backend at 127.0.0.1:5000)
python training/build_validator_dataset.py --include-mobile

# 2. train the REAL model  (refuses to run without BOTH real folders)
python training/train_validator.py --epochs 15 --batch-size 32

# 3. threshold sweep + confusion matrix + report
python training/evaluate_validator.py

# all three in one go:
.\run_train_validator.cmd
```

---

## Health segmentation - real U-Net (new crop-agnostic pipeline)

`models/leaf_segmenter/leaf_segmenter.keras` is a **real pixel-wise U-Net**
(compact 3-class, 128x128 I/O) that segments **background / healthy leaf /
symptomatic leaf**. Training labels are REAL: `datasets/segmentation_masks/`
are generated from the real leaf photos in `datasets/processed/` by
deterministic HSV/excess-green lesion analysis
(`training/build_segmentation_masks.py`, ~1,500 healthy + ~1,500 diseased
leaf photos, "healthy" class folders vs every other class folder). The U-Net
then LEARNS the spatial pattern of disease from those masks, so at inference
the affected area is a real model output - not a pixel heuristic and never a
random number.

At inference, `leaf_segmenter.py` up-samples the model's softmax mask to the
original photo size and computes:

    affected_area_pct = symptomatic pixels / (healthy + symptomatic) pixels x 100

then `Healthy` (0%), `Low`/`Medium`/`High` severity from the configurable
thresholds, the spray-decision support message, and a PNG overlay that
highlights ONLY the model-marked symptomatic pixels. If no trained model is
on disk, the deterministic pixel segmentation runs as an explicitly-labelled
fallback (`segmentation_source: "pixel_fallback"`) so the demo still works.

Rebuild everything:

```powershell
# 1. generate the real pixel masks + pairs.json
python training/build_segmentation_masks.py --n 1500

# 2. train the REAL U-Net (prints held-out IoU per class)
python training/train_segmenter.py --epochs 12 --batch-size 16 --img-size 128

# (both in one go)
.\run_segmenter.cmd
```

---

## Spray decision support (how the pipeline behaves)

| Status (from real segmented affected area) | Decision shown |
|---|---|
| `Healthy` (0% affected) | MONITOR - no spraying recommendation |
| `Diseased` + severity Low (0-10%) | MONITOR - recheck later |
| `Diseased` + severity Medium (10-40%) | INSPECTION RECOMMENDED - spray decision follows configured rules |
| `Diseased` + severity High (>40%) | TREATMENT ASSESSMENT RECOMMENDED |
| `Analysis Not Available` | no reliable model support for this image - no claim, no spray |
| `Low Plant Confidence` / `Not a Plant` | image is not confidently a plant leaf - no claim, no spray |

The system never auto-recommends a specific pesticide without a valid
crop/disease source.

---

## API reference

### `POST /predict`
Multipart form field `image` = plant photo.

Example response (real diseased leaf - new crop-agnostic pipeline):

```json
{
  "success": true,
  "status": "Diseased",
  "plant": null,
  "plant_confidence": null,
  "plant_validation_confidence": 99.98,
  "pipeline_stopped": false,
  "disease": null,
  "disease_confidence": null,
  "health_confidence": null,
  "severity": "High",
  "severity_available": true,
  "affected_area_pct": 96.9,
  "severity_thresholds": {"low_max": 10.0, "med_max": 40.0},
  "severity_note": "Estimated from visible leaf-area segmentation.",
  "overlay_data_uri": "data:image/png;base64,...",
  "message": "Disease detected. Targeted inspection or spraying may be required.",
  "pipeline": [
    {"stage": "Plant Validation", "result": "Plant", "confidence": 99.98, "reason": "validator_model"},
    {"stage": "Health Segmentation", "result": "High", "affected_area_pct": 96.9, "segmentation_source": "model", "note": "..."}
  ],
  "spray_decision": {"code": "TREATMENT_ASSESSMENT_RECOMMENDED", "label": "TREATMENT ASSESSMENT RECOMMENDED", "message": "High visible disease symptoms. Immediate agricultural inspection / treatment assessment recommended."}
}
```

`affected_area_pct` is the **real measured affected leaf area** from the
segmentation mask (symptomatic / (healthy + symptomatic) pixels) and
`severity` is derived ONLY from it - never from confidence values and never
random. `overlay_data_uri` highlights ONLY the pixels the model marked
symptomatic, side-by-side with the original in the UI. The `spray_decision`
blocks map the severity to MONITOR / INSPECTION RECOMMENDED /
TREATMENT ASSESSMENT RECOMMENDED support messages (no auto-pesticide choice).

Distinct statuses returned (each truthful): `Healthy`, `Diseased`,
`Not a Plant`, `Low Plant Confidence`, `Analysis Not Available`.
The response always includes `plant_validation_confidence` and a
`pipeline_stopped` flag so the UI can show exactly where processing stopped;
`Low Plant Confidence` and `Not a Plant` never reach health segmentation, and
`Analysis Not Available` is returned whenever the segmentation cannot reliably
find a leaf region - the system never guesses `Healthy` or `Diseased`.
Plant/crop identification is no longer a required step; no crop name is
requested or displayed.

### `GET /status`
Returns service name, the health-segmentation model status
(`segmenter_model_loaded`, `segmentation_source`, `seg_image_size`), the
plant/leaf validation threshold, the severity Low/Medium/High demo thresholds,
and the spray-decision mapping.

### `GET /supported_crops`
Legacy - the new pipeline is crop-agnostic. Kept for API compatibility.

### `GET /api/info`

Project name, version, and pipeline stage list.

---

## Training details (honest, reproducible)

- **Backbone:** MobileNetV2 (ImageNet pre-trained), frozen-head phase (lr 1e-3)
  then fine-tuning the last ~24 layers (lr 1e-4).
- **Input:** 224x224 RGB, rescaled to [0,1].
- **Augmentation:** rotation, shift, shear, zoom, brightness, horizontal flip.
- **Imbalance handling:** per-class inverse-frequency weights.
- **Callbacks:** EarlyStopping, ModelCheckpoint (best val), ReduceLROnPlateau.
- **Holdout:** stratified 70/15/15; the test portion is persisted to
  `models/test_split.json` so every evaluation is on untouched images.

### Held-out test metrics (combined 41-class model)

| Metric | Value |
|---|---|
| Test images (held out) | 1,151 |
| **Test accuracy** | **94.18%** |
| Macro F1 | 0.9352 |
| Weighted F1 | 0.9411 |

Source: `models/evaluation_report.json`. Re-run anytime with
`python training/evaluate_models.py --re-evaluate`.
Stage-2 identifier accuracy is written to
`models/plant_identifier/metadata.json` (test_accuracy) after training.

### Adding a new plant / crop (extensible without code changes)

The identification catalog is folder-driven: drop real, diverse images
(different angles, lighting, backgrounds, partial/multiple leaves) for a new
crop and the next retrain learns it. No classifier is ever retrained with
empty folders:

```bat
python training\add_plant.py "Wheat"
python training\add_plant.py "Wheat" "C:\my_phone_photos\wheat_leaves"
```
then retrain the Stage-2 identifier:
```bat
run_train_identifier.cmd
```

`add_plant.py` validates + de-duplicates the images, copies them into
`datasets\plant_identification\<Crop>\` (auto-discovered by training), and
registers the crop in `models\crop_metadata.json` / `models\supported_crops.json`
(identifiable; health analysis stays `Analysis Not Available` until a real
disease model is trained for it).

### Real-world validation notes

The system is deliberately strict on anything that is not a clean,
plant-dominant photo: low-confidence validation results stop the pipeline
(`Low Plant Confidence`), and a segmentation that cannot reliably find a leaf
region honestly returns `Analysis Not Available` instead of guessing. The user
is *never* shown a Healthy/Diseased result unless the real health model
supports it. Genuine close-up leaf photos pass and flow through to health
segmentation. Test against phone camera images taken in different
light/backgrounds before relying on the results for spraying.

---

## Environment configuration

| Variable | Default | Meaning |
|---|---|---|
| `SPRAYBOT_PLANT_THRESHOLD` | 70 | Stage-1 plant/leaf validation gate (below -> `Low Plant Confidence`, pipeline stops) |
| `SPRAYBOT_SEVERITY_LOW` | 10 | affected-area % that separates Severity Low/Medium (demo default) |
| `SPRAYBOT_SEVERITY_MED` | 40 | affected-area % that separates Severity Medium/High (demo default) |
| `SPRAYBOT_SEG_IMG_SIZE` | 128 | health-segmentation input resolution (U-Net) |
| `SPRAYBOT_HOST` / `SPRAYBOT_PORT` | 0.0.0.0 / 5000 | Flask bind |

The severity thresholds are **demonstration defaults** for the demo. Real
spraying decisions require calibrating these against agronomic thresholds for
the actual target crop before production use.

---

## Mobile phone -> laptop over Wi-Fi

1. `ipconfig` on the laptop; note the Wi-Fi IPv4 (e.g. `192.168.1.105`).
2. Start Flask (`run_backend.cmd`) - already bound to `0.0.0.0`.
3. Put the phone on the same Wi-Fi.
4. Open `http://192.168.1.105:5000` on the phone. The frontend now
   **auto-detects the backend URL**: it uses the same host that served the page
   (`/api/config`, request-host based) and never falls back to `127.0.0.1`.
   Settings -> Backend API URL is only needed for a manual override.
5. Settings -> Test Connection should report the combined model loaded.

The scanner uses `<input type="file" accept="image/*" capture="environment">`
which opens the rear camera without `getUserMedia` HTTPS requirements.

Connection-specific environment variables (backend):

| Variable | Default | What it does |
|---|---|---|
| `SPRAYBOT_HOST` | `0.0.0.0` | Bind address (must stay `0.0.0.0` for phone access) |
| `SPRAYBOT_PORT` | `5000` | Backend port |
| `SPRAYBOT_API_BASE_URL` | *(auto from request host)* | Hard override of the base URL served by `/api/config` |
| `SPRAYBOT_CORS_ORIGINS` | `*` | Comma-separated allowed origins for mobile cross-origin calls |

If the phone reports `Unable to connect to the AI analysis server.` the
request failed on the network path itself - check the laptop firewall inbound
rule for port 5000, that both devices are on the same network, and that
`SPRAYBOT_HOST=0.0.0.0`.

---

## Adding a new crop (real data only)

1. Place real labeled images:
   `datasets/processed/<Crop>/<Healthy_or_disease_name>/*.jpg`
   (at least ~100 images per class).
2. Re-run `python training/prepare_dataset.py` to rebuild the manifest.
3. Retrain the combined model: `python training/train_model.py`.
4. If the crop has diseases, also place data under
   `datasets/disease/<Crop>/` and train `train_disease_models.py`; else add the
   crop to the "healthy-only" list in `models/crop_metadata.json` generation.
5. Re-generate `models/supported_crops.json` and restart the backend.

The pipeline will then identify and (if applicable) detect disease for the new
crop. Until that real training exists, the app reports *Analysis Not Available*,
never a guessed Healthy/Diseased result.

---

## ESP32-CAM integration (future)

- The ESP32-CAM POSTs the same multipart image to `POST /predict`.
- **No on-device inference is claimed:** the 41-class model is ~10 MB while the
  ESP32-CAM has ~320 KB usable SRAM for TF Lite Micro. `convert_tflite.py`
  writes `models/tflite_report.json` proving the size mismatch.
  **Recommended:** ESP32-CAM captures & streams; Flask does the real inference;
  the result drives the rover spray decision.

---

## Requirements

Python 3.11, TensorFlow 2.x CPU/GPU, Flask, Pillow, scikit-learn, matplotlib,
numpy. See `requirements.txt`.

---

## License / data attribution

- PlantVillage images: PlantVillage public research dataset (via
  `abdallahalidev/plantvillage-dataset` & `emmarex/plantdisease` on Kaggle).
- Rice images: Rice Leaf Disease dataset (CDS Capstone / UCI Rice Leaf
  Diseases), academic use.
- All original dataset info is preserved in
  `datasets/processed/dataset_manifest.json`.