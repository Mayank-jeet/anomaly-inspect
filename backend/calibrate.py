"""Stage: calibration. Reads ONLY normal validation images. Never touches test.
Writes artifacts/stats/{category}.json: threshold + score_min/max for heatmap normalization.
"""
import json

import numpy as np
import torch
from torch.utils.data import DataLoader

from data.mvtec import MVTecDataset
from models.feature_extractor import PatchFeatureExtractor
from models.patchcore import PatchCoreScorer
from utils.common import load_config, set_seed


def calibrate_category(cfg, category, extractor, device):
    root = cfg["paths"]["data_root"]
    bank = np.load(cfg["paths"]["artifacts"] / "banks" / f"{category}.npy")
    scorer = PatchCoreScorer(bank)

    val_ds = MVTecDataset(root, category, "val")
    val_dl = DataLoader(val_ds, batch_size=16, shuffle=False)

    image_scores = []
    map_mins, map_maxs = [], []

    with torch.no_grad():
        for batch in val_dl:
            x = batch["image"].to(device)
            patches = extractor.extract_patches(x).cpu().numpy()
            B = x.size(0)
            patches = patches.reshape(B, 784, -1)

            for i in range(B):
                score, score_map = scorer.score_image(patches[i])
                image_scores.append(score)
                map_mins.append(score_map.min())
                map_maxs.append(score_map.max())

    image_scores = np.array(image_scores)
    # threshold rule: mean + 3*std of normal validation scores
    threshold = float(image_scores.mean() + 3 * image_scores.std())

    stats = {
        "threshold": threshold,
        "score_min": float(np.min(map_mins)),
        "score_max": float(np.max(map_maxs)),
        "val_score_mean": float(image_scores.mean()),
        "val_score_std": float(image_scores.std()),
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
              f"score_range=[{stats['score_min']:.4f}, {stats['score_max']:.4f}]")


if __name__ == "__main__":
    main()
