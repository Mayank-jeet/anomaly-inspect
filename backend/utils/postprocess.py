"""Score-map post-processing. NumPy/SciPy only (used by the API)."""
import numpy as np
from scipy.ndimage import gaussian_filter, zoom


def upsample_map(score_map: np.ndarray, size: int = 224) -> np.ndarray:
    """[h,w] -> [size,size], bilinear (order=1)."""
    h, w = score_map.shape
    return zoom(score_map, (size / h, size / w), order=1).astype(np.float32)


def blur_map(score_map: np.ndarray, sigma: float = 4.0) -> np.ndarray:
    return gaussian_filter(score_map, sigma=sigma).astype(np.float32)


def normalize_map(score_map: np.ndarray, score_min: float, score_max: float) -> np.ndarray:
    """Min-max normalize with GIVEN stats (from validation), clipped to [0,1]."""
    out = (score_map - score_min) / (score_max - score_min + 1e-8)
    return np.clip(out, 0.0, 1.0).astype(np.float32)