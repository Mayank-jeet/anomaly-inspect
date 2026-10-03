"""Stage C: build PatchCore memory banks. Reads train split only.
Writes artifacts/banks/{category}.npy, float32 [M, 384].
"""
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from data.mvtec import MVTecDataset
from models.coreset import greedy_coreset
from models.feature_extractor import PatchFeatureExtractor
from utils.common import load_config, set_seed


def build_bank_for_category(cfg, category, extractor, device):
    root = cfg["paths"]["data_root"]
    train_ds = MVTecDataset(root, category, "train")
    train_dl = DataLoader(train_ds, batch_size=16, shuffle=False)

    all_patches = []
    with torch.no_grad():
        for batch in train_dl:
            x = batch["image"].to(device)
            patches = extractor.extract_patches(x)  # [B*784, 384]
            all_patches.append(patches.cpu().numpy())

    all_patches = np.concatenate(all_patches, axis=0).astype(np.float32)
    n_total = all_patches.shape[0]

    ratio = cfg["patchcore"]["coreset_ratio"]
    n_select = max(1, int(n_total * ratio))
    proj_dim = cfg["patchcore"]["proj_dim"]

    t0 = time.time()
    idx = greedy_coreset(all_patches, n_select, proj_dim=proj_dim, seed=cfg["seed"])
    elapsed = time.time() - t0

    bank = all_patches[idx]
    return bank, n_total, elapsed


def main():
    cfg = load_config()
    set_seed(cfg["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)

    extractor = PatchFeatureExtractor().to(device).eval()

    bank_dir = cfg["paths"]["artifacts"] / "banks"
    bank_dir.mkdir(parents=True, exist_ok=True)

    size_rows = []
    for cat in cfg["categories"]:
        out_path = bank_dir / f"{cat}.npy"
        if out_path.exists():
            print(f"[{cat}] bank already exists, skipping")
            continue

        bank, n_total, elapsed = build_bank_for_category(cfg, cat, extractor, device)
        np.save(out_path, bank)

        size_mb = out_path.stat().st_size / (1024 * 1024)
        print(f"[{cat}] total_patches={n_total}  bank_size={bank.shape[0]}  "
              f"coreset_time={elapsed:.1f}s  file_size={size_mb:.1f}MB")
        size_rows.append((cat, n_total, bank.shape[0], size_mb))

    print("\nsummary:")
    for row in size_rows:
        print(row)


if __name__ == "__main__":
    main()