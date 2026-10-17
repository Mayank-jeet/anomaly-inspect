# anomaly-inspect — Unsupervised Industrial Defect Detection

anomaly-inspect finds and localizes manufacturing defects in photos of industrial parts, trained on **defect-free images only** — a realistic constraint, since defective examples are rare and expensive to collect on a real production line. Upload a photo of one of 15 part categories (bottles, cables, screws, textiles, etc.) and it returns a defective/normal verdict plus a heatmap showing *where*.

Two methods are built and compared from scratch: a convolutional-autoencoder baseline and a **PatchCore-style** method (frozen ResNet-18 + FAISS memory banks), trained and evaluated on [MVTec AD](https://www.mvtec.com/company/research/datasets/mvtec-ad), exported to ONNX, and served through a FastAPI + React demo.

🔗 **Live Application:** https://anomaly-inspect.vercel.app
🔗 **API:** https://anomaly-inspect-5m7u.onrender.com

> The backend runs on a free Render instance and loads model artifacts on demand — the first prediction for each category is slower (downloading its ~26 MB memory bank) and the service may take a few seconds to wake up after inactivity. See [Notes on the live demo](#notes-on-the-live-demo).

---

## Results at a glance

| Method | Mean image AUROC | Mean pixel AUROC |
|---|---|---|
| Autoencoder baseline | 0.669 | 0.505 |
| **PatchCore-style** | **0.977** | **0.973** |

PatchCore beats the autoencoder on **every one of the 15 categories**, several reaching a perfect 1.000. Full per-category tables, ablations, and error analysis are below.

---

## Features

- Per-category unsupervised models, trained on normal images only — no labeled defects used in training
- Two methods compared on identical preprocessing, splits and seeds: autoencoder baseline vs. PatchCore-style
- Frozen, ImageNet-pretrained ResNet-18 feature extractor (`layer2`+`layer3`, no fine-tuning)
- Greedy coreset subsampling to keep memory banks small without losing much accuracy
- FAISS nearest-neighbor search for real-time patch scoring
- Per-category calibration (verdict threshold + heatmap normalization) from validation data only
- Out-of-distribution guard: images that don't match the training setup are reported as "Unsupported image" instead of getting a misleading defective verdict (see [Out-of-distribution guard](#out-of-distribution-guard))
- 3 ablation families (backbone depth, coreset ratio, blur sigma) with accuracy, bank size, *and* latency reported together
- Per-defect-type error analysis with a heatmap gallery
- ONNX export with numerical verification against the original PyTorch model
- Lazy LRU memory-bank cache so the API fits in 512 MB of RAM without loading all 15 categories at once
- Model artifacts delivered via GitHub Release and downloaded on first use, keeping the git repo small
- React dashboard: category picker, drag-and-drop or sample-image upload, heatmap overlay with opacity control

---

## Tech Stack

### Modeling
- Python 3.11, PyTorch, torchvision (ResNet-18)
- FAISS (`faiss-cpu`) for nearest-neighbor search
- NumPy, SciPy, scikit-learn (AUROC)
- ONNX, ONNX Runtime
- Jupyter notebooks for exploration and error analysis

### Serving
- FastAPI, Uvicorn
- ONNX Runtime (CPU) — **no PyTorch at inference time**, kept the serving footprint small
- React (Vite)

### Data
- [MVTec AD](https://www.mvtec.com/company/research/datasets/mvtec-ad) — 15 categories, ~5,300 images

### Deployment
- Render (FastAPI backend)
- Vercel (React frontend)
- GitHub Releases (model artifacts: ONNX model + memory banks, ~405 MB total, not stored in git)

---

## Project Architecture

```text
anomaly-inspect/
│
├── backend/
│   │
│   ├── data/
│   │   ├── mvtec.py              # MVTecDataset: train/val/test splits
│   │   └── transforms.py          # preprocessing shared by train, eval, and serving
│   │
│   ├── models/
│   │   ├── autoencoder.py         # ConvAE baseline
│   │   ├── feature_extractor.py   # frozen ResNet-18, layer2+layer3
│   │   ├── coreset.py             # greedy k-center subsampling
│   │   └── patchcore.py           # FAISS scoring, score maps
│   │
│   ├── utils/
│   │   ├── metrics.py             # image/pixel AUROC
│   │   ├── postprocess.py         # upsample, blur, normalize (NumPy/SciPy only)
│   │   └── common.py              # seeding, config loading, paths
│   │
│   ├── api/                       # FastAPI serving app (no torch dependency)
│   │   ├── main.py                # routes, lifespan (ONNX + stats load), CORS
│   │   ├── inference.py           # preprocess → ONNX → FAISS → verdict
│   │   ├── bank_cache.py          # lazy LRU cache, downloads banks on demand
│   │   └── schemas.py
│   │
│   ├── notebooks/
│   │   ├── 01_explore_mvtec.ipynb
│   │   ├── 02_shapes_and_hooks.ipynb
│   │   └── 03_heatmap_gallery.ipynb   # error analysis, failure cases
│   │
│   ├── configs/default.yaml       # every hyperparameter in one place
│   ├── train_ae.py                # Stage B: autoencoder training
│   ├── build_memory_bank.py       # Stage C: PatchCore memory banks
│   ├── calibrate.py               # thresholds, heatmap range, OOD cutoff (validation only)
│   ├── evaluate.py                # Stage D: AUROC tables (test split)
│   ├── run_ablations.py           # Stage E: ablation study
│   ├── export_onnx.py             # Stage F: ONNX export + verification
│   ├── benchmark.py               # latency and size report
│   ├── make_samples.py            # small demo images per category
│   ├── check_ood.py               # sanity check of the out-of-distribution rule
│   ├── explore_ood_stats.py       # compares candidate out-of-distribution statistics
│   ├── tests/                     # pytest: transforms, dataset, coreset, metrics, API
│   └── samples/                   # demo images served by /samples
│
└── web/                            # React frontend
    └── src/
        ├── App.jsx
        └── api.js                  # all backend communication
```

| File | Role |
|---|---|
| `data/mvtec.py` | Seeded 90/10 train/val split of normal images; test split has both normal and defective, with pixel masks |
| `models/feature_extractor.py` | ResNet-18 `layer2`+`layer3` → upsample → concat → 3×3 avg-pool → 384-dim patch features |
| `models/coreset.py` | Greedy k-center selection (GPU-accelerated) to shrink the patch set before storing it |
| `models/patchcore.py` | FAISS `IndexFlatL2` nearest-neighbor search → score map → image score |
| `calibrate.py` | Per-category threshold (`mean + 3·std` of normal validation scores), heatmap min/max, and the out-of-distribution cutoff — reads validation data only |
| `check_ood.py` | Read-only sanity check: how many MVTec test images the out-of-distribution rule would reject, and the score of any image you pass in |
| `api/bank_cache.py` | Downloads and caches memory banks on first request per category; evicts least-recently-used when full |

---

## Method

```text
image (224×224)
      │
      ▼
ResNet-18 (frozen, ImageNet-pretrained)
      │
      ├── layer2 ──► [128, 28, 28] ─────┐
      │                                  ├─► concat ─► 3×3 avg-pool ─► 384-dim patches (28×28 grid)
      └── layer3 ──► [256, 14, 14] ──► upsample ┘
                                                        │
                      ┌─────────────────────────────────┴─────────────────────────────┐
                      │ TRAIN (normal images only)         │ TEST                      │
                      ▼                                     ▼                           │
           all patches ─► greedy coreset ─► memory bank      each patch ─► FAISS k=1 NN search
                              (.npy, 10%)                          against the memory bank
                                                                            │
                                                                            ▼
                                                            784 distances ─► 28×28 map
                                                                    ─► upsample to 224×224
                                                                    ─► Gaussian blur
                                                                            │
                                                              ┌─────────────┴──────────────┐
                                                              ▼                             ▼
                                                   image score = map.max()        heatmap (display)
                                                              │
                                                     compare to calibrated
                                                       per-category threshold
                                                              │
                                                              ▼
                                                     defective / normal verdict
```

The autoencoder baseline follows a simpler path: encode → decode → per-pixel squared reconstruction error → same max-based scoring.

Both methods train **one model per category** (15 total), since a screw and a bottle have nothing in common to generalize across.

---

## Out-of-distribution guard

PatchCore is a one-class novelty detector: "normal" means "looks like the training images" — same camera, background, lighting and scale. A photo from a different setup (say, a random toothbrush picture from the web) moves nearly **every** patch away from the memory bank, so the image score lands far above the threshold and the heatmap saturates uniformly. Without a guard, the demo would confidently label such an image "defective".

So `/predict` checks the image first:

```text
min patch distance  >  per-category OOD cutoff   ─►  "Unsupported image" (no verdict, no heatmap)
otherwise                                        ─►  normal defective / normal verdict + heatmap
```

**Statistic: the minimum patch distance** — how close the *best-matching* patch is to the memory bank. An image from the training setup always contains some patches that match the bank (background, context), even when the defect is global (`metal_nut` "flip", `transistor` "misplaced"). An image from another camera, background or object has none, so even its minimum is far away.

I compared four statistics (`explore_ood_stats.py`). Values are multiples of each statistic's maximum on normal validation images:

| Statistic | Worst MVTec test image | Two external toothbrush photos | Separation |
|---|---|---|---|
| median | 1.94 | 1.91 / 2.63 | ~1.0× (none) |
| p25 | 1.62 | 2.08 / 2.36 | 1.3× |
| p10 | 1.38 | 2.58 / 2.79 | 1.9× |
| **min** | **1.42** | **3.54 / 4.20** | **2.5×** |

The median — my first choice — could not separate globally-defective MVTec images from unrelated photos; the minimum separated them best.

**Cutoff:** the larger of `mean + 5·std` and `2.0 × max` of the normal validation minimums, per category (`calibrate.py`). `check_ood.py` is the sanity script that reports how many MVTec test images the rule would reject.

**Honest caveats:**
- The 2.0× margin was set between the worst in-setup MVTec test image and the external photos, so it was informed by the test split and only two outside images — a design choice, not a purely validation-derived value.
- It rejects *clearly* different images, not subtle shifts, and passing the guard does not guarantee a correct verdict.

---

## Results

### Autoencoder baseline vs. PatchCore (test set)

<details>
<summary><strong>Per-category results</strong> (click to expand)</summary>

| Category | AE image AUROC | AE pixel AUROC | PatchCore image AUROC | PatchCore pixel AUROC |
|---|---|---|---|---|
| bottle | 0.863 | 0.410 | 1.000 | 0.978 |
| cable | 0.463 | 0.760 | 0.977 | 0.977 |
| capsule | 0.221 | 0.542 | 0.947 | 0.984 |
| carpet | 0.857 | 0.598 | 0.979 | 0.987 |
| grid | 0.862 | 0.783 | 0.916 | 0.969 |
| hazelnut | 0.984 | 0.300 | 1.000 | 0.982 |
| leather | 0.975 | 0.614 | 1.000 | 0.989 |
| metal_nut | 0.396 | 0.289 | 0.993 | 0.971 |
| pill | 0.360 | 0.318 | 0.941 | 0.968 |
| screw | 0.352 | 0.433 | 0.958 | 0.990 |
| tile | 0.935 | 0.536 | 0.980 | 0.927 |
| toothbrush | 0.572 | 0.195 | 1.000 | 0.987 |
| transistor | 0.459 | 0.553 | 0.997 | 0.969 |
| wood | 0.954 | 0.677 | 0.986 | 0.926 |
| zipper | 0.783 | 0.571 | 0.975 | 0.986 |
| **mean** | **0.669** | **0.505** | **0.977** | **0.973** |

</details>

The autoencoder struggles most on object categories with uniform backgrounds (`capsule`, `metal_nut`, `pill`, `screw` — all below 0.5 AUROC, worse than random). The likely cause: its decoder's sigmoid output is bounded to `[0,1]`, but inputs are ImageNet-normalized with negative values, so reconstruction error ends up dominated by this scale mismatch rather than by the actual defect — a design trade-off that would need a separate `[0,1]`-only preprocessing path for the autoencoder to fix.

### Ablations

Run on a 4-category subset (`bottle`, `carpet`, `screw`, `metal_nut`); one setting changed at a time from the baseline.

| Variant | Mean image AUROC | Mean pixel AUROC | Bank size (MB) | Latency (ms) |
|---|---|---|---|---|
| **Baseline** (layer2+3, 10% coreset, σ=4) | 0.9825 | 0.9816 | 27.9 | 55.3 |
| layer2 only (no layer3) | 0.9882 | 0.9825 | 9.3 | 61.5 |
| coreset ratio = 1% | 0.9829 | 0.9782 | **2.8** | **33.6** |
| σ = 0 (no blur) | 0.9784 | 0.9812 | 27.9 | 212.3 |
| σ = 2 | 0.9802 | 0.9814 | 27.9 | 53.3 |
| σ = 8 | 0.9817 | 0.9814 | 27.9 | 56.8 |

**Takeaways:**
- **layer2-only slightly *beat* the full layer2+3 baseline** on this subset (0.988 vs 0.982 image AUROC) at a third of the bank size. The error analysis below suggests this may not hold on harder categories like `grid`, which wasn't in this subset — layer3's broader context likely matters more for subtle, low-contrast defects.
- **A 1% coreset loses almost nothing** (0.983 vs 0.982 image AUROC) while shrinking the bank **10x** and cutting latency ~40% — the strongest accuracy/cost trade-off found, and a strong candidate for a memory-constrained deployment.
- **σ = 0 (no blur) hurts both accuracy and latency.** The unblurred map is noisier (lower AUROC) and FAISS search time jumped to 212ms vs 55ms on this run — likely an unrelated system-load effect during that measurement rather than caused by sigma itself, but worth re-measuring before drawing a firm conclusion.

*Ablations are measured on the test split to understand method behavior, not to pick the deployed configuration — that would leak test information into the design. The deployed API uses the original baseline settings.*

### Latency and size (full 15 categories, CPU, single thread)

Measured with `python benchmark.py --threads 1` (ONNX Runtime and FAISS each limited to one thread, to approximate a small host); batch size 1, median over 100 runs after warm-up, including image decoding.

| | Mean | Total |
|---|---|---|
| Memory bank size | 26.3 MB/category | **393.9 MB** |
| ONNX model size | — | **11.1 MB** |
| Preprocess | 24.6 ms | |
| ONNX forward pass | 24.0 ms | |
| FAISS search | 45.8 ms | |
| Postprocess | 1.9 ms | |
| **Total (median)** | **97.2 ms** | |
| **Total (p95)** | **128.3 ms** | |

All 15 banks total ~394 MB, close to a free host's 512 MB RAM limit — this is why the API uses a lazy LRU cache (holds only 2 categories' banks in memory at a time) instead of loading all 15 up front.

### Error analysis

`grid` is PatchCore's weakest category (0.916 image AUROC). By defect type:

| Defect type | Accuracy |
|---|---|
| glue | 27.3% |
| bent | 41.7% |
| thread | 54.5% |
| metal_contamination | 63.6% |
| broken | 66.7% |
| good (normal) | 100% |

Zero false positives — every error is a missed defect. Inspecting individual false negatives shows the heatmap *does* highlight the true defect location correctly; the image's max score still falls below `grid`'s calibrated threshold. This points to **threshold sensitivity** (mild defects scoring close to the normal range) rather than the model failing to notice the defect — a defect-type-aware or percentile-based threshold could likely close some of this gap, at the cost of more false positives on normal images.

See `backend/notebooks/03_heatmap_gallery.ipynb` for the full gallery and per-category breakdowns.

---

## Design Decisions

- **One model per category, not one shared model.** A screw and a carpet texture share no useful visual structure; training per-category keeps each memory bank small and each model specialized.
- **Train on normal images only.** Matches the realistic constraint of a production line, where defective examples are rare, and matches MVTec AD's intended evaluation protocol.
- **Calibration reads only validation data, never test.** `calibrate.py` and `evaluate.py` are kept as separate scripts specifically so test leakage into threshold-tuning is structurally hard to introduce by accident. The one deliberate exception is the margin of the out-of-distribution guard, which was sanity-checked against MVTec test images (see [Out-of-distribution guard](#out-of-distribution-guard)) — documented rather than hidden.
- **ONNX + ONNX Runtime for serving, not PyTorch.** `api/requirements.txt` has no torch dependency — this keeps the deployed image small enough for a 512 MB free host and avoids installing a multi-hundred-MB deep learning framework just to run a frozen forward pass.
- **Artifacts delivered via GitHub Release + download-on-startup, not git.** Memory banks and the ONNX model total ~405 MB — too large to commit cleanly. `bank_cache.py`'s `ensure_file` downloads each one the first time it's needed and caches it on disk.
- **Lazy LRU bank cache, not "load all 15 at startup."** Free-tier RAM (512 MB) can't hold every category's bank at once; loading on demand and evicting the least-recently-used keeps memory bounded regardless of how many categories exist.
- **Greedy coreset instead of keeping every patch.** A full category's patch set can exceed 150,000 vectors; coreset subsampling keeps a representative 10% (or less, per the ablation) without a proportional accuracy loss.
- **Ablations report latency and bank size alongside accuracy**, not accuracy alone — an accuracy gain that costs 10x the memory or latency may not be worth it on a resource-constrained host.
- **Reject unsupported images instead of guessing.** A one-class detector has no notion of "this isn't even the right kind of image", so a photo from another setup would always come back "defective" with a meaningless heatmap. A calibrated out-of-distribution check on the minimum patch distance turns that into an explicit "Unsupported image" state. The minimum was chosen over the median because global defects (a flipped nut) shift the median almost as much as an unrelated photo does.

---

## Demo

- Pick a category, upload an image or choose a sample, and get a defective/normal verdict with a heatmap overlay.
- Heatmap opacity is adjustable; per-request latency is shown.

*(Screenshot / GIF — add here)*

## Notes on the live demo

- Model artifacts are hosted as [GitHub Release assets](https://github.com/Mayank-jeet/anomaly-inspect/releases/tag/artifacts-v1) and downloaded on first use per category — **the first prediction for a given category will be slower** (downloading its ~26 MB bank); subsequent requests for that category are fast.
- Render's free tier spins down when idle — the first request after inactivity may take a few extra seconds to wake the server.
- The models assume controlled imaging (fixed camera, plain background, like MVTec AD). Use the built-in samples or images taken the same way; arbitrary photos from the web are reported as **"Unsupported image"** instead of being judged (see [Out-of-distribution guard](#out-of-distribution-guard)).

---

## Setup

### Backend — full pipeline
```bash
cd backend
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
# Download MVTec AD separately, then set paths.data_root in configs/default.yaml
# (or set the MVTEC_ROOT environment variable)
```

```bash
python train_ae.py              # autoencoder baseline, all 15 categories
python build_memory_bank.py     # PatchCore memory banks
python evaluate.py              # AE + PatchCore AUROC tables
python calibrate.py             # per-category thresholds + OOD cutoff (validation only)
python run_ablations.py         # ablation study
python export_onnx.py           # ONNX export + verification
python benchmark.py             # latency and size report
python check_ood.py             # sanity check of the out-of-distribution rule
```

> **Windows note:** scripts that run PyTorch on the CPU and FAISS in the same process can abort with `OMP: Error #15` (both libraries bundle their own OpenMP runtime). `export_onnx.py`, `run_ablations.py`, `calibrate.py`, `check_ood.py`, `explore_ood_stats.py` and the API tests set `KMP_DUPLICATE_LIB_OK=TRUE` for this; for any other script, set it in your shell first (`$env:KMP_DUPLICATE_LIB_OK="TRUE"` in PowerShell). The deployed API has no PyTorch, so it never loads two OpenMP runtimes.

### Backend — running the API locally (slim, no torch)
```bash
cd backend
pip install -r api/requirements.txt
uvicorn api.main:app --reload
```

### Frontend
```bash
cd web
npm install
npm run dev
```

## API Overview

| Endpoint | Purpose |
|---|---|
| `GET /health` | ONNX load status, category count |
| `GET /categories` | List available categories |
| `GET /samples/{category}` | Sample images (good + defective) for the demo picker |
| `POST /predict` | Upload an image + category; returns the verdict, score and heatmap — or an out-of-distribution flag for unsupported images |

`/predict` response fields: `category`, `out_of_distribution`, `ood_score`, `ood_threshold`, `score`, `threshold`, `is_defective`, `heatmap_png_base64`, `input_png_base64` (the 224×224 crop the model saw, which the heatmap aligns with), `latency_ms`. When `out_of_distribution` is `true`, `is_defective` and `heatmap_png_base64` are `null`. Errors: `400` (not a PNG/JPEG or undecodable), `413` (over 5 MB), `422` (unknown category), `503` (a category's memory bank is unavailable).

## Deployment

```text
User
  │
  ▼
Vercel (React frontend)
  │
  │  REST API
  ▼
Render (FastAPI + ONNX Runtime, CPU)
  │
  ├── downloads ONNX model + memory bank on first use
  └── GitHub Release  (artifacts-v1: feature_extractor.onnx + 15 × {category}.npy)
```

| Variable (Render) | Purpose |
|---|---|
| `ONNX_URL` | GitHub Release URL for `feature_extractor.onnx` |
| `BANKS_URL` | GitHub Release base URL; `{category}.npy` is appended per request |
| `ALLOWED_ORIGINS` | Comma-separated frontend origin(s) for CORS |
| `BANK_CACHE_SIZE` | Max categories' banks held in memory at once (default 2) |
| `PYTHON_VERSION` | Pinned to `3.11.9` to match local development and available `onnxruntime` wheels |

| Variable (Vercel) | Purpose |
|---|---|
| `VITE_API_URL` | Render backend URL |

---

## Limitations

- Thresholds are calibrated per category from validation data using a single fixed rule (`mean + 3·std`); a defect-severity-aware threshold could likely improve recall further on borderline cases like `grid` (see error analysis).
- The autoencoder baseline's accuracy on some object categories could likely be improved with a preprocessing path matched to its sigmoid output range — a natural next refinement for that baseline specifically.
- Ablations were run on a 4-category subset, not all 15, for faster turnaround; a full 15-category run would confirm whether the trends generalize everywhere.
- The free-tier deployment has no persistent model cache across cold starts — each new instance re-downloads artifacts on first use per category.
- No fine-tuning of the ResNet-18 backbone; features are used exactly as ImageNet pretraining produced them.
- The models assume controlled imaging (same camera, background, lighting). The out-of-distribution guard rejects clearly different images, but it is a calibrated statistical rule: its margin was chosen with a sanity check against MVTec test images and only two external photos, it will not catch subtle shifts, and passing it does not guarantee a correct verdict.

## Future Improvements

- Defect-type-aware or percentile-based thresholds instead of a single `mean + 3·std` cutoff per category
- A `[0,1]`-range preprocessing path for the autoencoder to remove the sigmoid/normalization mismatch
- Full 15-category ablation run (currently a 4-category subset) to confirm the layer2-only finding generalizes
- WideResNet-50 backbone ablation (listed as an option in the original plan, not yet run)
- A persistent artifact cache or a paid tier to avoid repeated cold-start downloads
- Evaluate and tighten the out-of-distribution guard on a larger, more varied set of external images (and compare against a feature-statistics or lightweight learned detector)

---

## Dataset Credit

[MVTec AD](https://www.mvtec.com/company/research/datasets/mvtec-ad) is licensed CC BY-NC-SA 4.0, for non-commercial use, and is not redistributed in this repository — download it from the official source.

## Resume Bullet

> Built an unsupervised defect-detection system on MVTec AD comparing a convolutional-autoencoder baseline against a PatchCore-style method (ResNet-18 + FAISS memory banks), reaching **0.977 mean image AUROC** / **0.973 mean pixel AUROC** across 15 categories (vs. 0.669 / 0.505 for the baseline); exported to ONNX and deployed as a FastAPI + React demo with ~97ms median end-to-end latency on CPU.

## Author

### Mayank Jeet

- GitHub: [@Mayank-jeet](https://github.com/Mayank-jeet)
- LinkedIn: [Mayank Jeet](https://www.linkedin.com/in/mayank-jeet-211583364/)
