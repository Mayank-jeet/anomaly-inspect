from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from data.transforms import preprocess_image, preprocess_mask


class MVTecDataset(Dataset):
    def __init__(self, root, category, split, resize=256, crop=224, val_frac=0.1, seed=0):
        assert split in ("train", "val", "test")
        self.root = Path(root)
        self.category = category
        self.split = split
        self.resize, self.crop = resize, crop
        self.samples = []  # list of (img_path, mask_path_or_None, label)

        if split in ("train", "val"):
            good_dir = self.root / category / "train" / "good"
            paths = sorted(good_dir.glob("*.png"))
            rng = np.random.default_rng(seed)
            idx = rng.permutation(len(paths))
            n_val = int(len(paths) * val_frac)
            val_idx, train_idx = set(idx[:n_val]), set(idx[n_val:])
            chosen = val_idx if split == "val" else train_idx
            self.samples = [(paths[i], None, 0) for i in sorted(chosen)]
        else:  # test
            test_dir = self.root / category / "test"

            # We need to loop as test folder have images in different types of defects in different folders, e.g. good, crack, hole, etc.
            
            for defect_dir in sorted(test_dir.iterdir()):
                imgs = sorted(defect_dir.glob("*.png"))
                if defect_dir.name == "good":
                    self.samples += [(p, None, 0) for p in imgs]
                else:
                    gt_dir = self.root / category / "ground_truth" / defect_dir.name

                    # ground_truth folder contains the masked images of the defective images, i.e. images in folders in test other than good

                    for p in imgs:
                        mask_p = gt_dir / f"{p.stem}_mask.png"
                        self.samples.append((p, mask_p, 1))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        img_path, mask_path, label = self.samples[i]
        img = preprocess_image(Image.open(img_path), self.resize, self.crop)

        # the dataset can return the same kind of mask value for every image, even when no mask file is available.

        if mask_path is not None:
            mask = preprocess_mask(Image.open(mask_path), self.resize, self.crop)
        else:
            mask = np.zeros((1, self.crop, self.crop), dtype=np.float32)
        return {
            "image": torch.from_numpy(img),
            "mask": torch.from_numpy(mask),
            "label": label,
            "path": str(img_path),
        }