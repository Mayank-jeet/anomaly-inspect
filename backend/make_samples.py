"""Create small demo samples: backend/samples/{category}/{good,defective}/*.jpg

Why: the React sample picker (Step 19) needs a few ready-to-use images per category, and the
repo must stay small. Raw MVTec PNGs are ~1 MB each, so we save them with the shorter side
resized to 256 (the same first step as preprocess_image) as JPEG: ~20-30 KB each.
The API then crops to 224 exactly like during evaluation.

MVTec AD is CC BY-NC-SA: keep the attribution in the README and the demo.

Run (from backend/):  python make_samples.py
"""
from PIL import Image

from data.transforms import _resize_shorter_side
from utils.common import BACKEND_DIR, load_config

N_GOOD = 2
N_DEFECT_TYPES = 2   # one image from each of the first N defect folders


def save_small(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    img = _resize_shorter_side(Image.open(src).convert("RGB"), 256, Image.BILINEAR)
    img.save(dst, format="JPEG", quality=92)


def main():
    cfg = load_config()
    root = cfg["paths"]["data_root"]
    out = BACKEND_DIR / "samples"
    total = 0
    for cat in cfg["categories"]:
        test_dir = root / cat / "test"
        goods = sorted((test_dir / "good").glob("*.png"))
        # spread picks across the folder instead of taking the first few consecutive images
        for i in range(N_GOOD):
            src = goods[i * (len(goods) // N_GOOD)]
            save_small(src, out / cat / "good" / f"good_{i + 1}.jpg")
            total += 1
        defect_dirs = [d for d in sorted(test_dir.iterdir()) if d.is_dir() and d.name != "good"]
        for d in defect_dirs[:N_DEFECT_TYPES]:
            src = sorted(d.glob("*.png"))[0]
            save_small(src, out / cat / "defective" / f"{d.name}_{src.stem}.jpg")
            total += 1
    print(f"wrote {total} sample images to {out}")


if __name__ == "__main__":
    main()
