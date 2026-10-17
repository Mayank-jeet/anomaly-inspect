"""Export the frozen feature extractor to ONNX and verify it.

Why this exists
---------------
The deployed API runs on a small free host WITHOUT PyTorch. ONNX is a
framework-neutral file format that stores the network's graph + weights, so the
API can run the exact same network with just `onnxruntime`.

Only the CNN part (PatchFeatureExtractor: ResNet-18 layer2+layer3 -> upsample ->
concat -> 3x3 avg-pool) goes into ONNX. FAISS search, blur and thresholds stay in
NumPy/FAISS code that already has no torch dependency.

Writes: artifacts/onnx/feature_extractor.onnx

Run (from backend/):  python export_onnx.py
"""
import numpy as np
import torch

from data.mvtec import MVTecDataset
from models.feature_extractor import PatchFeatureExtractor
from models.patchcore import PatchCoreScorer
from utils.common import load_config, set_seed

FEATURE_ATOL = 1e-4   # max allowed abs difference between torch and onnx features
SCORE_ATOL = 1e-3     # max allowed abs difference between torch and onnx image scores


def export(cfg):
    """Trace the PyTorch module once and save it as an .onnx file."""
    model = PatchFeatureExtractor().eval()
    dummy = torch.zeros(1, 3, 224, 224)  # example input: tracing records ops using this shape

    out_dir = cfg["paths"]["artifacts"] / "onnx"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "feature_extractor.onnx"

    torch.onnx.export(
        model,
        dummy,
        str(out_path),
        input_names=["image"],
        output_names=["features"],
        # Only the batch axis is dynamic. H and W stay fixed at 224 so the
        # feature grid is always 28x28 (= 784 patches per image).
        dynamic_axes={"image": {0: "batch"}, "features": {0: "batch"}},
        opset_version=17,
        do_constant_folding=True,  # pre-computes anything that does not depend on the input
    )
    size_mb = out_path.stat().st_size / 1e6
    print(f"exported to {out_path}  ({size_mb:.1f} MB)")
    return out_path


def _session(onnx_path):
    import onnxruntime as ort
    return ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])


def _pick_indices(n_total, n_pick):
    """Evenly spaced test indices so we cover both good and defective images."""
    return sorted(set(np.linspace(0, n_total - 1, num=min(n_pick, n_total)).astype(int).tolist()))


def verify_features(cfg, onnx_path, categories=("bottle", "screw"), n_images=6):
    """Check 1: raw feature tensors from ONNX == PyTorch, on real test images."""
    torch_model = PatchFeatureExtractor().eval()
    session = _session(onnx_path)
    root = cfg["paths"]["data_root"]

    worst = 0.0
    for cat in categories:
        ds = MVTecDataset(root, cat, "test")
        for i in _pick_indices(len(ds), n_images):
            x = ds[i]["image"].unsqueeze(0)
            with torch.no_grad():
                t_out = torch_model(x).numpy()
            o_out = session.run(["features"], {"image": x.numpy()})[0]

            assert t_out.shape == o_out.shape == (1, 384, 28, 28), (t_out.shape, o_out.shape)
            diff = float(np.abs(t_out - o_out).max())
            worst = max(worst, diff)
            print(f"  [{cat}] image {i:3d}: max_abs_diff={diff:.2e}")

    print(f"  worst max_abs_diff={worst:.2e} (limit {FEATURE_ATOL:.0e})")
    assert worst < FEATURE_ATOL, "ONNX features diverge from PyTorch beyond tolerance"


def verify_dynamic_batch(cfg, onnx_path, category="bottle", batch_size=4):
    """Check 2: one exported model must accept batch=1 AND batch=N (the 'dynamic axis')."""
    torch_model = PatchFeatureExtractor().eval()
    session = _session(onnx_path)
    ds = MVTecDataset(cfg["paths"]["data_root"], category, "test")

    idx = _pick_indices(len(ds), batch_size)
    x = torch.stack([ds[i]["image"] for i in idx])  # [B,3,224,224]
    with torch.no_grad():
        t_out = torch_model(x).numpy()
    o_out = session.run(["features"], {"image": x.numpy()})[0]

    assert o_out.shape == (len(idx), 384, 28, 28), o_out.shape
    diff = float(np.abs(t_out - o_out).max())
    print(f"  batch={len(idx)}: output shape={o_out.shape}, max_abs_diff={diff:.2e}")
    assert diff < FEATURE_ATOL


def verify_scores(cfg, onnx_path, categories=("bottle", "screw"), n_images=6):
    """Check 3 (the one that matters): final image score via ONNX == score via the
    PyTorch path that evaluate.py uses (extract_patches -> PatchCoreScorer.score_image).
    """
    torch_model = PatchFeatureExtractor().eval()
    session = _session(onnx_path)
    root = cfg["paths"]["data_root"]
    sigma = cfg["patchcore"]["sigma"]

    worst = 0.0
    for cat in categories:
        bank = np.load(cfg["paths"]["artifacts"] / "banks" / f"{cat}.npy")
        scorer = PatchCoreScorer(bank)
        ds = MVTecDataset(root, cat, "test")

        for i in _pick_indices(len(ds), n_images):
            x = ds[i]["image"].unsqueeze(0)

            # --- PyTorch path: identical to evaluate_category_patchcore ---
            with torch.no_grad():
                t_patches = torch_model.extract_patches(x).numpy()          # [784, 384]
            t_score, _ = scorer.score_image(t_patches, sigma=sigma)

            # --- ONNX path: what the API will do (no torch) ---
            feat = session.run(["features"], {"image": x.numpy()})[0]       # [1,384,28,28]
            o_patches = feat.transpose(0, 2, 3, 1).reshape(-1, feat.shape[1])  # [784, 384]
            o_score, _ = scorer.score_image(o_patches, sigma=sigma)

            diff = abs(t_score - o_score)
            worst = max(worst, diff)
            label = int(ds[i]["label"])
            print(f"  [{cat}] image {i:3d} (label={label}): torch={t_score:.5f}  "
                  f"onnx={o_score:.5f}  diff={diff:.2e}")

    print(f"  worst score diff={worst:.2e} (limit {SCORE_ATOL:.0e})")
    assert worst < SCORE_ATOL, "ONNX image scores diverge from the PyTorch/evaluate.py path"


def main():
    cfg = load_config()
    set_seed(cfg["seed"])
    onnx_path = export(cfg)

    print("\n[1/3] raw features: ONNX vs PyTorch")
    verify_features(cfg, onnx_path)

    print("\n[2/3] dynamic batch axis")
    verify_dynamic_batch(cfg, onnx_path)

    print("\n[3/3] final image scores: ONNX path vs evaluate.py path")
    verify_scores(cfg, onnx_path)

    print("\nALL CHECKS PASSED - ONNX export is safe to use in the API.")


if __name__ == "__main__":
    main()
