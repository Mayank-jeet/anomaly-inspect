"""Stage E: ablations. Change ONE setting at a time, rebuild banks, re-evaluate.
Writes results/tables/ablations.csv.

Set ABLATION_CATEGORIES below to a subset for a fast first pass, or to
cfg["categories"] (all 15) once the subset run looks sane.
"""
import numpy as np
import torch
from torch.utils.data import DataLoader

from data.mvtec import MVTecDataset
from models.coreset import greedy_coreset
from models.feature_extractor import PatchFeatureExtractor
from models.patchcore import PatchCoreScorer
from utils.common import load_config, set_seed
from utils.metrics import image_auroc, pixel_auroc

import pandas as pd

# --- toggle this: subset for a fast first pass, or cfg["categories"] for all 15 ---
ABLATION_CATEGORIES = ["bottle", "carpet", "screw", "metal_nut"]


def build_bank(cfg, category, extractor, device, coreset_ratio):
    root = cfg["paths"]["data_root"]
    train_ds = MVTecDataset(root, category, "train")
    train_dl = DataLoader(train_ds, batch_size=16, shuffle=False)

    all_patches = []
    with torch.no_grad():
        for batch in train_dl:
            x = batch["image"].to(device)
            patches = extractor.extract_patches(x)
            all_patches.append(patches.cpu().numpy())
    all_patches = np.concatenate(all_patches, axis=0).astype(np.float32)

    n_select = max(1, int(all_patches.shape[0] * coreset_ratio))
    idx = greedy_coreset(all_patches, n_select, proj_dim=cfg["patchcore"]["proj_dim"], seed=cfg["seed"])
    return all_patches[idx]


def evaluate_with_bank(cfg, category, bank, extractor, device):
    root = cfg["paths"]["data_root"]
    scorer = PatchCoreScorer(bank)
    test_ds = MVTecDataset(root, category, "test")
    test_dl = DataLoader(test_ds, batch_size=16, shuffle=False)

    all_labels, all_scores, all_masks, all_maps = [], [], [], []
    with torch.no_grad():
        for batch in test_dl:
            x = batch["image"].to(device)
            patches = extractor.extract_patches(x).cpu().numpy()
            B = x.size(0)
            patches = patches.reshape(B, 784, -1)
            for i in range(B):
                score, score_map = scorer.score_image(patches[i])
                all_scores.append(score)
                all_labels.append(int(batch["label"][i]))
                all_maps.append(score_map)
                all_masks.append(batch["mask"][i, 0].numpy())

    img_auroc = image_auroc(all_labels, all_scores)
    pix_auroc = pixel_auroc(np.stack(all_masks), np.stack(all_maps))
    bank_size_mb = bank.nbytes / (1024 * 1024)
    return img_auroc, pix_auroc, bank_size_mb


def run_variant(cfg, variant_name, extractor, device, coreset_ratio):
    """Runs one ablation variant across ABLATION_CATEGORIES, returns mean row."""
    img_scores, pix_scores, sizes = [], [], []
    for cat in ABLATION_CATEGORIES:
        bank = build_bank(cfg, cat, extractor, device, coreset_ratio)
        img_auroc, pix_auroc, bank_mb = evaluate_with_bank(cfg, cat, bank, extractor, device)
        img_scores.append(img_auroc)
        pix_scores.append(pix_auroc)
        sizes.append(bank_mb)
        print(f"  [{variant_name}] {cat}: image_auroc={img_auroc:.4f} pixel_auroc={pix_auroc:.4f}")

    return {
        "variant": variant_name,
        "mean_image_auroc": float(np.mean(img_scores)),
        "mean_pixel_auroc": float(np.mean(pix_scores)),
        "mean_bank_size_mb": float(np.mean(sizes)),
    }


class Layer2OnlyExtractor(PatchFeatureExtractor):
    """Same as PatchFeatureExtractor but skips layer3 and the concat/upsample."""
    @torch.no_grad()
    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        f2 = self.layer2(x)       # [B,128,28,28] -- no layer3, no concat
        f2 = self.pool(f2)
        return f2


def main():
    cfg = load_config()
    set_seed(cfg["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)
    print("ablation categories:", ABLATION_CATEGORIES)

    rows = []

    # --- baseline (matches Step 12's settings, for reference in the same table) ---
    print("\n[baseline] layer2+layer3, coreset_ratio=0.1")
    extractor = PatchFeatureExtractor().to(device).eval()
    rows.append(run_variant(cfg, "baseline (layer2+3, 10% coreset)", extractor, device,
                             cfg["patchcore"]["coreset_ratio"]))

    # --- ablation 1: layer2 only ---
    print("\n[ablation 1] layer2 only")
    extractor_l2 = Layer2OnlyExtractor().to(device).eval()
    rows.append(run_variant(cfg, "layer2 only", extractor_l2, device,
                             cfg["patchcore"]["coreset_ratio"]))

    # --- ablation 2: coreset ratio 1% instead of 10% ---
    print("\n[ablation 2] coreset_ratio=0.01")
    rows.append(run_variant(cfg, "coreset_ratio=1%", extractor, device, 0.01))

    df = pd.DataFrame(rows)
    out_path = cfg["paths"]["results"] / "tables" / "ablations.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"\nsaved to {out_path}")
    print(df)


if __name__ == "__main__":
    main()
