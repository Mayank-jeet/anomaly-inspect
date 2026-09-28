import numpy as np
from PIL import Image

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _resize_shorter_side(img: Image.Image, size: int, resample) -> Image.Image:
    w, h = img.size
    if w < h:
        new_w, new_h = size, round(h * size / w)
    else:
        new_h, new_w = size, round(w * size / h)
    return img.resize((new_w, new_h), resample)


def _center_crop(img: Image.Image, crop: int) -> Image.Image:
    w, h = img.size
    left = (w - crop) // 2
    top = (h - crop) // 2
    return img.crop((left, top, left + crop, top + crop))


def preprocess_image(img: Image.Image, resize: int = 256, crop: int = 224) -> np.ndarray:
    img = img.convert("RGB")
    img = _resize_shorter_side(img, resize, Image.BILINEAR)
    img = _center_crop(img, crop)
    arr = np.asarray(img, dtype=np.float32) / 255.0          # [H,W,3]
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    return arr.transpose(2, 0, 1).copy()                      # [3,H,W]


def preprocess_mask(mask: Image.Image, resize: int = 256, crop: int = 224) -> np.ndarray:
    mask = mask.convert("L")
    mask = _resize_shorter_side(mask, resize, Image.NEAREST)
    mask = _center_crop(mask, crop)
    arr = np.asarray(mask, dtype=np.float32)
    arr = (arr > 127).astype(np.float32)
    return arr[None, :, :]                                    # [1,H,W]