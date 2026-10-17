"""The serving pipeline: bytes -> preprocess -> ONNX -> FAISS -> score map -> verdict.

Everything here is the same code the research scripts use (data.transforms, models.patchcore,
utils.postprocess), so the served score equals the evaluate.py score for the same image.
No torch is imported anywhere in this file.
"""
import base64
import io
import time

import numpy as np
from PIL import Image, UnidentifiedImageError

from api.bank_cache import BankCache
from data.transforms import IMAGENET_MEAN, IMAGENET_STD, preprocess_image
from utils.postprocess import normalize_map


class InvalidImageError(ValueError):
    """Uploaded bytes are not a decodable image."""


def decode_image(data: bytes) -> Image.Image:
    try:
        img = Image.open(io.BytesIO(data))
        img.load()  # force a full decode now, so corrupt files fail here and not later
        return img
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as e:
        raise InvalidImageError("Could not decode the uploaded file as an image") from e


def colorize(heat: np.ndarray) -> np.ndarray:
    """[H,W] in [0,1] -> uint8 [H,W,3], blue (normal) to red (anomalous). Plain NumPy 'jet'
    colormap, because the slim API image has no matplotlib."""
    r = np.clip(1.5 - np.abs(4 * heat - 3), 0, 1)
    g = np.clip(1.5 - np.abs(4 * heat - 2), 0, 1)
    b = np.clip(1.5 - np.abs(4 * heat - 1), 0, 1)
    return (np.stack([r, g, b], axis=-1) * 255).astype(np.uint8)


def to_png_b64(arr_uint8: np.ndarray) -> str:
    buf = io.BytesIO()
    Image.fromarray(arr_uint8).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def denormalize(arr: np.ndarray) -> np.ndarray:
    """[3,H,W] ImageNet-normalized float -> uint8 [H,W,3] (inverse of preprocess_image's last step)."""
    img = arr.transpose(1, 2, 0) * IMAGENET_STD + IMAGENET_MEAN
    return (np.clip(img, 0, 1) * 255).astype(np.uint8)


class Inferencer:
    def __init__(self, session, stats: dict, cache: BankCache, sigma: float):
        self.session = session   # onnxruntime.InferenceSession (thread-safe for run())
        self.stats = stats       # {category: {threshold, score_min, score_max, ...}}
        self.cache = cache
        self.sigma = sigma

    def warmup(self):
        """One dummy forward pass at startup so the first real request is not the slow one."""
        self.session.run(["features"], {"image": np.zeros((1, 3, 224, 224), dtype=np.float32)})

    def predict(self, img_bytes: bytes, category: str) -> dict:
        t0 = time.perf_counter()

        img = decode_image(img_bytes)
        arr = preprocess_image(img)                                       # [3,224,224] float32
        feat = self.session.run(["features"], {"image": arr[None]})[0]    # [1,384,28,28]
        patches = np.ascontiguousarray(feat.transpose(0, 2, 3, 1).reshape(-1, feat.shape[1]))  # [784,384]

        scorer = self.cache.get(category)
        score, score_map = scorer.score_image(patches, sigma=self.sigma)  # same call as evaluate.py

        st = self.stats[category]
        heat = normalize_map(score_map, st["score_min"], st["score_max"])  # stats come from VAL images only

        return {
            "category": category,
            "score": round(score, 5),
            "threshold": round(st["threshold"], 5),
            "is_defective": bool(score > st["threshold"]),
            "heatmap_png_base64": to_png_b64(colorize(heat)),
            "input_png_base64": to_png_b64(denormalize(arr)),
            "latency_ms": round((time.perf_counter() - t0) * 1000.0, 1),
        }
