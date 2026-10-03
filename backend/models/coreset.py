import numpy as np
import torch


def greedy_coreset(features: np.ndarray, n_select: int, proj_dim: int = 128, seed: int = 0) -> np.ndarray:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.default_rng(seed)
    N, D = features.shape
    n_select = min(n_select, N)

    feat_t = torch.from_numpy(features).to(device)

    if D > proj_dim:
        R = torch.from_numpy(
           (rng.standard_normal((D, proj_dim)) / np.sqrt(proj_dim)).astype(np.float32)
       ).to(device)
        proj = feat_t @ R
    else:
        proj = feat_t

    selected = np.zeros(n_select, dtype=np.int64)
    selected[0] = rng.integers(N)

    min_dist = torch.norm(proj - proj[selected[0]], dim=1)

    for i in range(1, n_select):
        next_idx = int(torch.argmax(min_dist).item())
        selected[i] = next_idx
        new_dist = torch.norm(proj - proj[next_idx], dim=1)
        min_dist = torch.minimum(min_dist, new_dist)

    return selected