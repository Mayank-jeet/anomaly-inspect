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
- Per-category calibration (threshold + heatmap normalization) from validation data only — test data is never used to tune anything
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
│   ├── calibrate.py               # thresholds + normalization (validation only)
│   ├── evaluate.py                # Stage D: AUROC tables (test split)
│   ├── run_ablations.py           # Stage E: ablation study
│   ├── export_onnx.py             # Stage F: ONNX export + verification
│   └── benchmark.py                # latency and size report
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
| `calibrate.py` | Per-category threshold (`mean + 3·std` of normal validation scores) and heatmap min/max — reads validation data only |
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

### Latency and size (full 15 categories, CPU)

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
- **Calibration reads only validation data, never test.** `calibrate.py` and `evaluate.py` are kept as separate scripts specifically so test leakage into threshold-tuning is structurally hard to introduce by accident.
- **ONNX + ONNX Runtime for serving, not PyTorch.** `api/requirements.txt` has no torch dependency — this keeps the deployed image small enough for a 512 MB free host and avoids installing a multi-hundred-MB deep learning framework just to run a frozen forward pass.
- **Artifacts delivered via GitHub Release + download-on-startup, not git.** Memory banks and the ONNX model total ~405 MB — too large to commit cleanly. `bank_cache.py`'s `ensure_file` downloads each one the first time it's needed and caches it on disk.
- **Lazy LRU bank cache, not "load all 15 at startup."** Free-tier RAM (512 MB) can't hold every category's bank at once; loading on demand and evicting the least-recently-used keeps memory bounded regardless of how many categories exist.
- **Greedy coreset instead of keeping every patch.** A full category's patch set can exceed 150,000 vectors; coreset subsampling keeps a representative 10% (or less, per the ablation) without a proportional accuracy loss.
- **Ablations report latency and bank size alongside accuracy**, not accuracy alone — an accuracy gain that costs 10x the memory or latency may not be worth it on a resource-constrained host.

---

## Demo

- Pick a category, upload an image or choose a sample, and get a defective/normal verdict with a heatmap overlay.
- Heatmap opacity is adjustable; per-request latency is shown.

*(Screenshot / GIF — add here)*

## Notes on the live demo

- Model artifacts are hosted as [GitHub Release assets](https://github.com/Mayank-jeet/anomaly-inspect/releases/tag/artifacts-v1) and downloaded on first use per category — **the first prediction for a given category will be slower** (downloading its ~26 MB bank); subsequent requests for that category are fast.
- Render's free tier spins down when idle — the first request after inactivity may take a few extra seconds to wake the server.

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
python calibrate.py             # per-category thresholds (validation only)
python run_ablations.py         # ablation study
python export_onnx.py           # ONNX export + verification
python benchmark.py             # latency and size report
```

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
| `POST /predict` | Upload an image + category, get verdict, score, heatmap |

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

## Future Improvements

- Defect-type-aware or percentile-based thresholds instead of a single `mean + 3·std` cutoff per category
- A `[0,1]`-range preprocessing path for the autoencoder to remove the sigmoid/normalization mismatch
- Full 15-category ablation run (currently a 4-category subset) to confirm the layer2-only finding generalizes
- WideResNet-50 backbone ablation (listed as an option in the original plan, not yet run)
- A persistent artifact cache or a paid tier to avoid repeated cold-start downloads

---

## Dataset Credit

[MVTec AD](https://www.mvtec.com/company/research/datasets/mvtec-ad) is licensed CC BY-NC-SA 4.0, for non-commercial use, and is not redistributed in this repository — download it from the official source.

## Resume Bullet

> Built an unsupervised defect-detection system on MVTec AD comparing a convolutional-autoencoder baseline against a PatchCore-style method (ResNet-18 + FAISS memory banks), reaching **0.977 mean image AUROC** / **0.973 mean pixel AUROC** across 15 categories (vs. 0.669 / 0.505 for the baseline); exported to ONNX and deployed as a FastAPI + React demo with ~97ms median end-to-end latency on CPU.

## Author

### Mayank Jeet

- GitHub: [@Mayank-jeet](https://github.com/Mayank-jeet)
- LinkedIn: [Mayank Jeet](https://www.linkedin.com/in/mayank-jeet-211583364/)
