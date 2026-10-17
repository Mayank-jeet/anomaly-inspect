"""Sanity check for the out-of-distribution (OOD) rule. Read-only: writes nothing.

The rule: an image is "unsupported" when its MIN patch distance (how close its best-matching patch
is to the memory bank) exceeds the cutoff stored in artifacts/stats/{category}.json.

Mode 1 (default): for each category, count how many MVTec TEST images (good + defective) the rule
would reject. Expected: 0, because every MVTec test image comes from the training setup.
NOTE: this reads the test split, so it is a coverage sanity check ONLY. It is never used to pick
the cutoff; calibrate.py derives that from normal validation images alone.

Mode 2 (--images): print the OOD score of arbitrary image files for one category, e.g. your own
photos or web images, to see how far outside the training setup they are.

Run (from backend/):
    python check_ood.py
    python check_ood.py --categories bottle toothbrush
    python check_ood.py --category toothbrush --images C:\\path\\a.jpg C:\\path\\b.png
"""
import os

# Windows dev workaround: torch (CPU) and faiss each bundle an OpenMP runtime ("OMP: Error #15").
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import json

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

from data.mvtec import MVTecDataset
from data.transforms import preprocess_image
from models.feature_extractor import PatchFeatureExtractor
from models.patchcore import PatchCoreScorer
from utils.common import load_config


def load_category(cfg, category):
    art = cfg["paths"]["artifacts"]
    scorer = PatchCoreScorer(np.load(art / "banks" / f"{category}.npy"))
    stats = json.loads((art / "stats" / f"{category}.json").read_text())
    if "val_patchmin_max" not in stats:
        raise SystemExit(f"{category}: stats file has no patch-min OOD cutoff. Run `python calibrate.py` first.")
    return scorer, stats


def min_distances(extractor, scorer, x, device):
    """x: [B,3,224,224] -> min patch distance for each image."""
    with torch.no_grad():
        patches = extractor.extract_patches(x.to(device)).cpu().numpy().reshape(x.size(0), 784, -1)
    return [float(scorer.score_patches(patches[i]).min()) for i in range(x.size(0))]


def check_test_split(cfg, categories, extractor, device):
    print(f"{'category':<11} {'good':>5} {'flag':>5} {'defect':>7} {'flag':>5} "
          f"{'max score / cutoff':>19}")
    total_flagged = 0
    for cat in categories:
        scorer, stats = load_category(cfg, cat)
        ds = MVTecDataset(cfg["paths"]["data_root"], cat, "test")
        scores, labels = [], []
        for batch in DataLoader(ds, batch_size=16, shuffle=False):
            scores += min_distances(extractor, scorer, batch["image"], device)
            labels += [int(v) for v in batch["label"]]
        scores, labels = np.array(scores), np.array(labels)
        flagged = scores > stats["ood_threshold"]
        total_flagged += int(flagged.sum())
        print(f"{cat:<11} {int((labels == 0).sum()):>5} {int(flagged[labels == 0].sum()):>5} "
              f"{int((labels == 1).sum()):>7} {int(flagged[labels == 1].sum()):>5} "
              f"{scores.max() / stats['ood_threshold']:>19.2f}")
    print(f"\ntotal MVTec test images flagged as out-of-distribution: {total_flagged} (expected 0)")


def check_images(cfg, category, paths, extractor, device):
    scorer, stats = load_category(cfg, category)
    print(f"category={category}  ood_threshold={stats['ood_threshold']:.4f}  "
          f"(normal val images: patch-min max {stats['val_patchmin_max']:.4f})")
    for p in paths:
        x = torch.from_numpy(preprocess_image(Image.open(p))).unsqueeze(0)
        m = min_distances(extractor, scorer, x, device)[0]
        verdict = "OUT OF DISTRIBUTION" if m > stats["ood_threshold"] else "in distribution"
        print(f"  {os.path.basename(p)[:44]:<44} min={m:.4f}  ({m / stats['ood_threshold']:.2f}x cutoff)  {verdict}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--categories", nargs="*", default=None)
    ap.add_argument("--category", default=None, help="category for --images mode")
    ap.add_argument("--images", nargs="*", default=None)
    args = ap.parse_args()

    cfg = load_config()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    extractor = PatchFeatureExtractor().to(device).eval()

    if args.images:
        if not args.category:
            raise SystemExit("--images needs --category")
        check_images(cfg, args.category, args.images, extractor, device)
    else:
        check_test_split(cfg, args.categories or cfg["categories"], extractor, device)


if __name__ == "__main__":
    main()
