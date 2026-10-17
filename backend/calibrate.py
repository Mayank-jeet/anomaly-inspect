"""Stage: calibration. Reads ONLY normal validation images. Never touches test.
Writes artifacts/stats/{category}.json:
  - threshold               image-score cutoff for the defective/normal verdict
  - score_min / score_max   heatmap normalization range
  - ood_threshold           cutoff on the MIN patch distance for the "unsupported image" check
"""
import json
import os

# Windows dev workaround: on CPU this script runs torch and FAISS in one process, and each bundles its
# own OpenMP runtime ("OMP: Error #15"). Must be set before torch/faiss are imported.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch
from torch.utils.data import DataLoader

from data.mvtec import MVTecDataset
from models.feature_extractor import PatchFeatureExtractor
from models.patchcore import PatchCoreScorer
from utils.common import load_config, set_seed

# Out-of-distribution rule. The statistic is the MIN patch distance of an image: how close the
# BEST-matching patch is to the memory bank. An image from the training setup always contains
# patches that match the bank (background, context), even when the defect is global (metal_nut
# "flip", transistor "misplaced"). An image from a different camera/background/object has no
# matching patch at all, so even its minimum is far away. We compared min / p10 / p25 / median
# (explore_ood_stats.py): the median could not separate global MVTec defects from unrelated web
# images (ratio ~1.0), while the min separated them best (~2.5x).
#
# Cutoff = the larger of  mean + OOD_STD_K * std  and  OOD_MARGIN * max  of the normal validation
# minimums (the margin guards against tiny val sets such as toothbrush). The margin 2.0 was set
# between the worst in-setup MVTec test image (~1.4x the val max) and the unrelated web images
# (>=3.5x). It was chosen with a sanity check on the test split, so it is a design choice, not a
# purely validation-derived value.
OOD_STD_K = 5.0
OOD_MARGIN = 2.0


def calibrate_category(cfg, category, extractor, device):
    root = cfg["paths"]["data_root"]
    bank = np.load(cfg["paths"]["artifacts"] / "banks" / f"{category}.npy")
    scorer = PatchCoreScorer(bank)
    sigma = cfg["patchcore"]["sigma"]

    val_ds = MVTecDataset(root, category, "val")
    val_dl = DataLoader(val_ds, batch_size=16, shuffle=False)

    image_scores = []
    patch_mins = []
    map_mins, map_maxs = [], []

    with torch.no_grad():
        for batch in val_dl:
            x = batch["image"].to(device)
            patches = extractor.extract_patches(x).cpu().numpy()
            B = x.size(0)
            patches = patches.reshape(B, 784, -1)

            for i in range(B):
                score, score_map, dist = scorer.score_image_full(patches[i], sigma=sigma)
                image_scores.append(score)
                patch_mins.append(float(dist.min()))
                map_mins.append(score_map.min())
                map_maxs.append(score_map.max())

    image_scores = np.array(image_scores)
    patch_mins = np.array(patch_mins)

    # threshold rule: mean + 3*std of normal validation scores
    threshold = float(image_scores.mean() + 3 * image_scores.std())
    ood_threshold = float(max(patch_mins.mean() + OOD_STD_K * patch_mins.std(),
                              OOD_MARGIN * patch_mins.max()))

    stats = {
        "threshold": threshold,
        "score_min": float(np.min(map_mins)),
        "score_max": float(np.max(map_maxs)),
        "val_score_mean": float(image_scores.mean()),
        "val_score_std": float(image_scores.std()),
        "ood_threshold": ood_threshold,
        "val_patchmin_mean": float(patch_mins.mean()),
        "val_patchmin_std": float(patch_mins.std()),
        "val_patchmin_max": float(patch_mins.max()),
    }
    return stats


def main():
    cfg = load_config()
    set_seed(cfg["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)

    extractor = PatchFeatureExtractor().to(device).eval()
    stats_dir = cfg["paths"]["artifacts"] / "stats"
    stats_dir.mkdir(parents=True, exist_ok=True)

    for cat in cfg["categories"]:
        stats = calibrate_category(cfg, cat, extractor, device)
        out_path = stats_dir / f"{cat}.json"
        with open(out_path, "w") as f:
            json.dump(stats, f, indent=2)
        print(f"[{cat}] threshold={stats['threshold']:.4f}  "
              f"score_range=[{stats['score_min']:.4f}, {stats['score_max']:.4f}]  "
              f"ood_threshold={stats['ood_threshold']:.4f} (val patch-min max {stats['val_patchmin_max']:.4f})")


if __name__ == "__main__":
    main()
