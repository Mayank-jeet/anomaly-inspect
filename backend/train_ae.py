"""Train the autoencoder baseline. Run: python train_ae.py"""
import torch
from torch.utils.data import DataLoader

from data.mvtec import MVTecDataset
from models.autoencoder import ConvAE
from utils.common import load_config, set_seed

EPOCHS = 50
LR = 1e-3


def train_one_category(cfg, category, device):
    ckpt_dir = cfg["paths"]["checkpoints"]
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f"ae_{category}.pt"

    if ckpt_path.exists():
        print(f"[{category}] checkpoint already exists, skipping")
        return None

    root = cfg["paths"]["data_root"]
    train_ds = MVTecDataset(root, category, "train")
    val_ds = MVTecDataset(root, category, "val")
    train_dl = DataLoader(train_ds, batch_size=16, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=16, shuffle=False)

    model = ConvAE().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=5)
    scaler = torch.amp.GradScaler(device, enabled=(device == "cuda"))
    mse = torch.nn.MSELoss()

    best_val = float("inf")

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        for batch in train_dl:
            x = batch["image"].to(device)
            opt.zero_grad()
            with torch.amp.autocast(device, enabled=(device == "cuda")):
                out = model(x)
                loss = mse(out, x)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            train_loss += loss.item() * x.size(0)
        train_loss /= len(train_ds)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_dl:
                x = batch["image"].to(device)
                out = model(x)
                val_loss += mse(out, x).item() * x.size(0)
        val_loss /= len(val_ds)
        sched.step(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), ckpt_path)

        print(f"[{category}] epoch {epoch+1}/{EPOCHS}  train={train_loss:.5f}  val={val_loss:.5f}")

    # Save a zip after EVERY category finishes, not just at the end of the whole script
    import subprocess
    subprocess.run(["zip", "-r", "-q", "/kaggle/working/checkpoints.zip", str(ckpt_dir)])
    print(f"[{category}] checkpoints.zip updated")

    return best_val
def main():
    cfg = load_config()
    set_seed(cfg["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)
    for cat in cfg["categories"]:
        train_one_category(cfg, cat, device)


if __name__ == "__main__":
    main()