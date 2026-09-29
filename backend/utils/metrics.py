"""Image-level and pixel-level AUROC."""
import numpy as np
from sklearn.metrics import roc_auc_score


def image_auroc(labels, image_scores) -> float:
    """labels: [N] in {0,1}; image_scores: [N], higher = more anomalous."""
    labels = np.asarray(labels).astype(int)
    scores = np.asarray(image_scores, dtype=np.float64)
    if len(np.unique(labels)) < 2:
        raise ValueError("image_auroc needs both normal and defective samples")
    return float(roc_auc_score(labels, scores))


def pixel_auroc(masks, score_maps) -> float:
    """masks: [N,H,W] or [N,1,H,W] in {0,1}; score_maps: same shape, floats."""
    m = (np.asarray(masks) > 0.5).astype(int).ravel()
    s = np.asarray(score_maps, dtype=np.float64).ravel()
    if len(np.unique(m)) < 2:
        raise ValueError("pixel_auroc needs both defect and background pixels")
    return float(roc_auc_score(m, s))