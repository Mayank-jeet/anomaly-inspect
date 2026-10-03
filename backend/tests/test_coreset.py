import time
import numpy as np
from models.coreset import greedy_coreset


def test_coreset_beats_random_on_coverage():
    rng = np.random.default_rng(0)
    # two well-separated clusters, so good coverage requires picking from both
    cluster_a = rng.normal(loc=0, scale=0.3, size=(500, 2))
    cluster_b = rng.normal(loc=10, scale=0.3, size=(500, 2))
    data = np.vstack([cluster_a, cluster_b]).astype(np.float32)

    n_select = 20
    coreset_idx = greedy_coreset(data, n_select, proj_dim=2, seed=0)

    rng2 = np.random.default_rng(1)
    random_idx = rng2.choice(len(data), n_select, replace=False)

    def max_min_dist(idx):
        selected = data[idx]
        dists = np.linalg.norm(data[:, None, :] - selected[None, :, :], axis=2)
        return dists.min(axis=1).max()  # worst-covered point's distance to nearest selected

    coreset_coverage = max_min_dist(coreset_idx)
    random_coverage = max_min_dist(random_idx)
    assert coreset_coverage < random_coverage


def test_coreset_runs_fast_enough():
    rng = np.random.default_rng(0)
    data = rng.standard_normal((20000, 384)).astype(np.float32)
    t0 = time.time()
    idx = greedy_coreset(data, n_select=2000, proj_dim=128, seed=0)
    elapsed = time.time() - t0
    assert len(idx) == 2000
    assert len(set(idx.tolist())) == 2000  # no duplicates
    assert elapsed < 60  # minutes-not-hours sanity check on CPU