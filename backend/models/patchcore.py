"""PatchCore scoring: FAISS nearest-neighbor search against a memory bank, producing anomaly maps."""
import faiss
import numpy as np

from utils.postprocess import upsample_map, blur_map


class PatchCoreScorer:
    def __init__(self, bank: np.ndarray):
        """bank: [M, 384] float32, the coreset memory bank for one category."""
        bank = np.ascontiguousarray(bank, dtype=np.float32)
        self.index = faiss.IndexFlatL2(bank.shape[1])
        self.index.add(bank)

    def score_patches(self, patches: np.ndarray) -> np.ndarray:
        """patches: [N, 384] float32. Returns [N] distances to nearest bank neighbor (L2, not squared)."""
        patches = np.ascontiguousarray(patches, dtype=np.float32)
        sq_dist, _ = self.index.search(patches, k=1)  # [N, 1], squared L2
        return np.sqrt(sq_dist[:, 0])

    def score_image(self, patches: np.ndarray, grid_size: int = 28,
                     out_size: int = 224, sigma: float = 4.0):
        """patches: [784, 384] for ONE image. Returns (image_score, score_map[out_size,out_size])."""
        dist = self.score_patches(patches)                      # [784]
        score_map = dist.reshape(grid_size, grid_size).astype(np.float32)
        score_map = upsample_map(score_map, out_size)
        score_map = blur_map(score_map, sigma)
        image_score = float(score_map.max())
        return image_score, score_map