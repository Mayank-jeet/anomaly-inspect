"""Explore which image-level statistic best separates "same setup" images from "different setup"
images. Read-only: writes nothing and changes no cutoff.

For every statistic (min, p10, p25, median of the 784 patch distances) it reports each image as a
multiple of the largest value seen on that category's normal VALIDATION images ("x val max").

Table 1 (default): the worst ratio over all MVTec TEST images of each category (good + defective).
  A good statistic keeps this LOW, because test images come from the training setup.
Table 2 (--images): the ratio of your own images. A good statistic makes these HIGH.
Pick the statistic whose Table 2 values sit well above its Table 1 values.

NOTE: Table 1 reads the test split, so this is exploration/sanity only. The deployed cutoff is
always computed from validation images (calibrate.py).

Run (from backend/):
    python explore_ood_stats.py
    python explore_ood_stats.py --category toothbrush --images "C:\\a.jpg" "C:\\b.jpg"
"""
import os

# Windows dev workaround: torch (CPU) and faiss each bundle an OpenMP runtime ("OMP: Error #15").
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

from data.mvtec import MVTecDataset
from data.transforms import preprocess_image
from models.feature_extractor import PatchFeatureExtractor
from models.patchcore import PatchCoreScorer
from utils.common import load_config

STATS = {
    "min":    lambda d: float(d.min()),
    "p10":    lambda d: float(np.percentile(d, 10)),
    "p25":    lambda d: float(np.percentile(d, 25)),
    "median": lambda d: float(np.median(d)),
}


def patch_dists(extractor, scorer, x, device):
    """x: [B,3,224,224] -> list of B arrays, each the 784 patch distances of one image."""
    with torch.no_grad():
        patches = extractor.extract_patches(x.to(device)).cpu().numpy().reshape(x.size(0), 784, -1)
    return [scorer.score_patches(patches[i]) for i in range(x.size(0))]


def split_stats(cfg, cat, split, extractor, scorer, device):
    ds = MVTecDataset(cfg["paths"]["data_root"], cat, split)
    rows = []
    for batch in DataLoader(ds, batch_size=16, shuffle=False):
        for d in patch_dists(extractor, scorer, batch["image"], device):
            rows.append({k: f(d) for k, f in STATS.items()})
    return rows


def val_max(cfg, cat, extractor, scorer, device):
    rows = split_stats(cfg, cat, "val", extractor, scorer, device)
    return {k: max(r[k] for r in rows) for k in STATS}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--categories", nargs="*", default=None)
    ap.add_argument("--category", default=None)
    ap.add_argument("--images", nargs="*", default=None)
    args = ap.parse_args()

    cfg = load_config()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    extractor = PatchFeatureExtractor().to(device).eval()
    art = cfg["paths"]["artifacts"]

    def scorer_for(cat):
        return PatchCoreScorer(np.load(art / "banks" / f"{cat}.npy"))

    if args.images:
        if not args.category:
            raise SystemExit("--images needs --category")
        cat = args.category
        scorer = scorer_for(cat)
        vmax = val_max(cfg, cat, extractor, scorer, device)
        print(f"{cat}: ratio = statistic / its max on normal validation images\n")
        print(f"{'image':<44}" + "".join(f"{k:>9}" for k in STATS))
        for p in args.images:
            x = torch.from_numpy(preprocess_image(Image.open(p))).unsqueeze(0)
            d = patch_dists(extractor, scorer, x, device)[0]
            print(f"{os.path.basename(p)[:43]:<44}" + "".join(f"{STATS[k](d) / vmax[k]:>9.2f}" for k in STATS))
        return

    categories = args.categories or cfg["categories"]
    print("worst ratio over MVTec TEST images (statistic / its max on normal validation images)\n")
    print(f"{'category':<12}" + "".join(f"{k:>9}" for k in STATS))
    overall = {k: 0.0 for k in STATS}
    for cat in categories:
        scorer = scorer_for(cat)
        vmax = val_max(cfg, cat, extractor, scorer, device)
        test_rows = split_stats(cfg, cat, "test", extractor, scorer, device)
        worst = {k: max(r[k] for r in test_rows) / vmax[k] for k in STATS}
        overall = {k: max(overall[k], worst[k]) for k in STATS}
        print(f"{cat:<12}" + "".join(f"{worst[k]:>9.2f}" for k in STATS))
    print(f"{'WORST':<12}" + "".join(f"{overall[k]:>9.2f}" for k in STATS))


if __name__ == "__main__":
    main()
