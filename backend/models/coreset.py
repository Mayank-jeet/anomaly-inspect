"""Greedy k-center coreset subsampling for PatchCore memory banks."""
import numpy as np


def greedy_coreset(features: np.ndarray, n_select: int, proj_dim: int = 128, seed: int = 0) -> np.ndarray:
    """features: [N, D]. Returns indices of n_select selected rows, chosen by greedy k-center.
    Selection runs on a random projection to proj_dim for speed; the returned indices
    index into the ORIGINAL features, so the caller keeps full-dimensional vectors.
    """
    rng = np.random.default_rng(seed)
    N, D = features.shape
    n_select = min(n_select, N)

    # random projection for speed (Johnson-Lindenstrauss style, just a random linear map)
    if D > proj_dim:
        R = rng.standard_normal((D, proj_dim)).astype(np.float32) / np.sqrt(proj_dim)
        proj = features @ R  # [N, proj_dim]
    else:
        proj = features

    selected = np.zeros(n_select, dtype=np.int64)
    selected[0] = rng.integers(N)

    # min_dist[i] = distance from point i to the nearest currently-selected point
    min_dist = np.linalg.norm(proj - proj[selected[0]], axis=1)

    for i in range(1, n_select):
        next_idx = int(np.argmax(min_dist))
        selected[i] = next_idx
        new_dist = np.linalg.norm(proj - proj[next_idx], axis=1)
        min_dist = np.minimum(min_dist, new_dist)

    return selected