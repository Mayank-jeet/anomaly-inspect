""" Ablations: Change ONE setting at a time and re-evaluate.

Variants (each differs from the baseline by exactly one setting):
  baseline            layer2+layer3 features, 10% coreset, blur sigma=4
  layer2 only         drop layer3 (features: 384-d -> 128-d)
  coreset_ratio=1%    smaller memory bank
  sigma=0 / 2 / 8     different Gaussian blur on the score map (reuses the baseline bank)

Per variant we report mean image AUROC, mean pixel AUROC, mean bank size (MB) and CPU latency
(feature extraction + FAISS search/post-processing, batch size 1, median per category).

Writes (in results/tables/):
  ablations.csv                 one row per variant (means over the chosen categories)
  ablations_per_category.csv    the same numbers per category

Run (from backend/):
    python run_ablations.py                 # fast subset: bottle, carpet, screw, metal_nut
    python run_ablations.py --all           # all 15 categories (slow: use a GPU for the coreset step)
    python run_ablations.py --categories bottle screw --out ablations_test.csv
"""
import argparse
import copy
import os
import time

# Windows dev workaround: this script runs torch on CPU (latency measurement) and FAISS in the same
# process, and each bundles its own OpenMP runtime ("OMP: Error #15"). Must be set before the imports.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from data.mvtec import MVTecDataset
from models.coreset import greedy_coreset
from models.feature_extractor import PatchFeatureExtractor
from models.patchcore import PatchCoreScorer
from utils.common import load_config, set_seed
from utils.metrics import image_auroc, pixel_auroc

SUBSET_CATEGORIES = ["bottle", "carpet", "screw", "metal_nut"]


class Layer2OnlyExtractor(PatchFeatureExtractor):
    """Same as PatchFeatureExtractor but skips layer3 and the upsample/concat."""

    @torch.no_grad()
    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        f2 = self.layer2(x)       # [B,128,28,28]; no layer3, no concat
        return self.pool(f2)


def build_bank(cfg, category, extractor, device, coreset_ratio):
    """Extract patch features from the TRAIN split (val images excluded) and apply the coreset."""
    train_ds = MVTecDataset(cfg["paths"]["data_root"], category, "train")
    train_dl = DataLoader(train_ds, batch_size=16, shuffle=False)

    chunks = []
    with torch.no_grad():
        for batch in train_dl:
            chunks.append(extractor.extract_patches(batch["image"].to(device)).cpu().numpy())
    patches = np.concatenate(chunks, axis=0).astype(np.float32)

    n_select = max(1, int(patches.shape[0] * coreset_ratio))
    idx = greedy_coreset(patches, n_select, proj_dim=cfg["patchcore"]["proj_dim"], seed=cfg["seed"])
    return patches[idx]


def evaluate_with_bank(cfg, category, bank, extractor, device, sigma):
    """AUROCs on the TEST split for one category."""
    scorer = PatchCoreScorer(bank)
    test_ds = MVTecDataset(cfg["paths"]["data_root"], category, "test")
    test_dl = DataLoader(test_ds, batch_size=16, shuffle=False)

    labels, scores, masks, maps = [], [], [], []
    with torch.no_grad():
        for batch in test_dl:
            x = batch["image"].to(device)
            patches = extractor.extract_patches(x).cpu().numpy()
            B = x.size(0)
            patches = patches.reshape(B, 28 * 28, -1)
            for i in range(B):
                score, score_map = scorer.score_image(patches[i], sigma=sigma)
                scores.append(score)
                labels.append(int(batch["label"][i]))
                maps.append(score_map)
                masks.append(batch["mask"][i, 0].numpy())

    return image_auroc(labels, scores), pixel_auroc(np.stack(masks), np.stack(maps))


def measure_latency(cfg, category, bank, extractor, sigma, n_images=30, warmup=5):
    """CPU latency per image at batch size 1, as the API would see it (minus decoding, which is
    the same for every variant). Always on CPU, even if the ablation itself ran on a GPU, because
    the deployed API is CPU-only. Returns (extract_ms, search_ms) medians."""
    extractor_cpu = copy.deepcopy(extractor).cpu().eval()
    scorer = PatchCoreScorer(bank)
    ds = MVTecDataset(cfg["paths"]["data_root"], category, "test")
    idx = np.linspace(0, len(ds) - 1, num=min(n_images, len(ds))).astype(int)
    images = [ds[int(i)]["image"].unsqueeze(0) for i in idx]

    extract_ms, search_ms = [], []
    with torch.no_grad():
        for j, x in enumerate(images):
            t0 = time.perf_counter()
            patches = extractor_cpu.extract_patches(x).numpy()
            t1 = time.perf_counter()
            scorer.score_image(patches, sigma=sigma)       # FAISS search + upsample + blur + max
            t2 = time.perf_counter()
            if j >= warmup:                                # first calls are slower: discard
                extract_ms.append((t1 - t0) * 1000)
                search_ms.append((t2 - t1) * 1000)
    return float(np.median(extract_ms)), float(np.median(search_ms))


def run_variant(cfg, name, feat_key, extractor, device, ratio, sigma, categories, bank_store):
    """Evaluate one variant on every category. Banks are cached by (features, category, ratio),
    so the sigma variants reuse the baseline bank instead of rebuilding it."""
    rows = []
    for cat in categories:
        key = (feat_key, cat, ratio)
        if key not in bank_store:
            bank_store[key] = build_bank(cfg, cat, extractor, device, ratio)
        bank = bank_store[key]

        img_auroc, pix_auroc = evaluate_with_bank(cfg, cat, bank, extractor, device, sigma)
        extract_ms, search_ms = measure_latency(cfg, cat, bank, extractor, sigma)
        rows.append({
            "variant": name, "category": cat,
            "image_auroc": img_auroc, "pixel_auroc": pix_auroc,
            "bank_mb": bank.nbytes / 1e6,          # MB (1e6 bytes), same unit as benchmark.py
            "extract_ms": extract_ms, "search_ms": search_ms,
        })
        print(f"  [{name}] {cat}: image={img_auroc:.4f} pixel={pix_auroc:.4f} "
              f"bank={bank.nbytes / 1e6:.1f}MB extract={extract_ms:.1f}ms search={search_ms:.1f}ms")
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true", help="use all 15 categories")
    ap.add_argument("--categories", nargs="*", default=None, help="explicit category list")
    ap.add_argument("--out", default="ablations.csv", help="file name inside results/tables/")
    args = ap.parse_args()

    cfg = load_config()
    set_seed(cfg["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    categories = args.categories or (cfg["categories"] if args.all else SUBSET_CATEGORIES)
    base_ratio = cfg["patchcore"]["coreset_ratio"]
    base_sigma = cfg["patchcore"]["sigma"]   # config-driven, like every other script
    print("device:", device)
    print("categories:", categories)

    base = PatchFeatureExtractor().to(device).eval()
    l2 = Layer2OnlyExtractor().to(device).eval()

    # (name, feature key, extractor, coreset ratio, sigma); exactly one setting differs from baseline
    variants = [
        ("baseline (layer2+3, 10% coreset)", "l23", base, base_ratio, base_sigma),
        ("layer2 only",                      "l2",  l2,   base_ratio, base_sigma),
        ("coreset_ratio=1%",                 "l23", base, 0.01,       base_sigma),
        ("sigma=0 (no blur)",                "l23", base, base_ratio, 0),
        ("sigma=2",                          "l23", base, base_ratio, 2),
        ("sigma=8",                          "l23", base, base_ratio, 8),
    ]

    bank_store, frames = {}, []
    for name, feat_key, extractor, ratio, sigma in variants:
        print(f"\n[{name}]")
        frames.append(run_variant(cfg, name, feat_key, extractor, device, ratio, sigma,
                                  categories, bank_store))

    per_cat = pd.concat(frames, ignore_index=True)
    summary = per_cat.groupby("variant", sort=False).agg(
        mean_image_auroc=("image_auroc", "mean"),
        mean_pixel_auroc=("pixel_auroc", "mean"),
        mean_bank_size_mb=("bank_mb", "mean"),
        extract_ms=("extract_ms", "mean"),
        search_ms=("search_ms", "mean"),
    ).reset_index()
    summary["latency_ms"] = summary["extract_ms"] + summary["search_ms"]
    summary.insert(1, "n_categories", len(categories))
    summary.insert(2, "categories", ",".join(categories))

    out_dir = cfg["paths"]["results"] / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / args.out
    per_cat_path = out_dir / (out_path.stem + "_per_category.csv")
    summary.round(4).to_csv(out_path, index=False)
    per_cat.round(4).to_csv(per_cat_path, index=False)

    print(f"\nsaved to {out_path}\nsaved to {per_cat_path}")
    print(summary.drop(columns=["categories"]).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
