"""Stage D (autoencoder mode): evaluate the AE baseline on the test split.
Writes results/tables/ae_baseline.csv. This is the ONLY script allowed to touch test data.
"""
import torch
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd

from data.mvtec import MVTecDataset
from models.autoencoder import ConvAE
from utils.common import load_config, set_seed
from utils.metrics import image_auroc, pixel_auroc


def evaluate_category(cfg, category, device):
    root = cfg["paths"]["data_root"]
    ckpt_path = cfg["paths"]["checkpoints"] / f"ae_{category}.pt"

    model = ConvAE().to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    test_ds = MVTecDataset(root, category, "test")
    test_dl = DataLoader(test_ds, batch_size=16, shuffle=False)

    all_labels, all_scores = [], []
    all_masks, all_maps = [], []

    with torch.no_grad():
        for batch in test_dl:
            x = batch["image"].to(device)
            out = model(x)
            # per-pixel squared error, averaged over the 3 channels -> [B, H, W]
            err_map = ((out - x) ** 2).mean(dim=1).cpu().numpy()

            for i in range(x.size(0)):
                all_scores.append(err_map[i].max())
                all_labels.append(int(batch["label"][i]))
                all_maps.append(err_map[i])
                all_masks.append(batch["mask"][i, 0].numpy())  # [H,W]

    img_auroc = image_auroc(all_labels, all_scores)
    pix_auroc = pixel_auroc(np.stack(all_masks), np.stack(all_maps))
    return img_auroc, pix_auroc


def main():
    cfg = load_config()
    set_seed(cfg["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)

    rows = []
    for cat in cfg["categories"]:
        img_auroc, pix_auroc = evaluate_category(cfg, cat, device)
        print(f"[{cat}] image_auroc={img_auroc:.4f}  pixel_auroc={pix_auroc:.4f}")
        rows.append({"category": cat, "image_auroc": img_auroc, "pixel_auroc": pix_auroc})

    df = pd.DataFrame(rows)
    mean_row = {"category": "mean", "image_auroc": df["image_auroc"].mean(),
                "pixel_auroc": df["pixel_auroc"].mean()}
    df = pd.concat([df, pd.DataFrame([mean_row])], ignore_index=True)

    out_path = cfg["paths"]["results"] / "tables" / "ae_baseline.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"saved to {out_path}")


if __name__ == "__main__":
    main()