"""latency + size benchmark of the SERVING path.

Why this exists
---------------
("X ms per image on CPU, Y MB total") must come from the
same code path the API will run, not from the research code. So this script is
deliberately torch-free: PIL/NumPy preprocessing -> onnxruntime -> FAISS ->
NumPy/SciPy post-processing. It imports exactly the modules the API will import.

It times each stage separately, so you can answer "where does the time go?":
  preprocess : decode image bytes + resize/crop/normalize   (data.transforms)
  onnx       : CNN forward pass -> [1,384,28,28]            (onnxruntime)
  faiss      : nearest-neighbour search of 784 patches      (bank size matters!)
  postproc   : reshape, upsample, blur, max, normalize      (utils.postprocess)
  
"""
import argparse
import io
import json
import time

import numpy as np
import pandas as pd
import faiss
import onnxruntime as ort
from PIL import Image

from data.transforms import preprocess_image
from models.patchcore import PatchCoreScorer
from utils.common import load_config
from utils.postprocess import upsample_map, blur_map, normalize_map


def ms_since(t0):
    return (time.perf_counter() - t0) * 1000.0


def pick_test_files(root, category, n=10):
    """Evenly spaced test images (mix of good + defective). Read as BYTES, because the
    API receives an uploaded file's bytes, so decoding must be part of the timing."""
    files = sorted((root / category / "test").rglob("*.png"))
    idx = sorted(set(np.linspace(0, len(files) - 1, num=min(n, len(files))).astype(int).tolist()))
    return [f.read_bytes() for f in (files[i] for i in idx)]


def run_once(img_bytes, session, scorer, stats, sigma):
    """One full inference, returning per-stage milliseconds. Mirrors what api/inference.py will do."""
    t = time.perf_counter()
    arr = preprocess_image(Image.open(io.BytesIO(img_bytes)))[None]        # [1,3,224,224]
    t_pre = ms_since(t)

    t = time.perf_counter()
    feat = session.run(["features"], {"image": arr})[0]                    # [1,384,28,28]
    t_onnx = ms_since(t)

    t = time.perf_counter()
    patches = np.ascontiguousarray(feat.transpose(0, 2, 3, 1).reshape(-1, feat.shape[1]))  # [784,384]
    dist = scorer.score_patches(patches)                                   # [784]
    t_faiss = ms_since(t)

    t = time.perf_counter()
    smap = blur_map(upsample_map(dist.reshape(28, 28).astype(np.float32), 224), sigma)
    score = float(smap.max())
    _ = normalize_map(smap, stats["score_min"], stats["score_max"])        # heatmap for the UI
    _ = score > stats["threshold"]                                          # verdict
    t_post = ms_since(t)

    return t_pre, t_onnx, t_faiss, t_post


def benchmark_category(cfg, category, session, onnx_mb, runs, warmup):
    art = cfg["paths"]["artifacts"]
    bank_path = art / "banks" / f"{category}.npy"
    bank = np.load(bank_path)
    scorer = PatchCoreScorer(bank)
    stats = json.loads((art / "stats" / f"{category}.json").read_text())
    images = pick_test_files(cfg["paths"]["data_root"], category)
    sigma = cfg["patchcore"]["sigma"]

    for i in range(warmup):                       # warm-up: first calls are slower (allocations, caches)
        run_once(images[i % len(images)], session, scorer, stats, sigma)

    times = np.array([run_once(images[i % len(images)], session, scorer, stats, sigma)
                      for i in range(runs)])      # [runs, 4]
    total = times.sum(axis=1)

    return {
        "category": category,
        "bank_patches": bank.shape[0],
        "bank_mb": bank_path.stat().st_size / 1e6,
        "onnx_mb": onnx_mb,
        "preprocess_ms": float(np.median(times[:, 0])),
        "onnx_ms": float(np.median(times[:, 1])),
        "faiss_ms": float(np.median(times[:, 2])),
        "postprocess_ms": float(np.median(times[:, 3])),
        "total_ms_median": float(np.median(total)),
        "total_ms_p95": float(np.percentile(total, 95)),
    }


def gpu_reference(runs, warmup):
    """Optional: feature-extractor latency on GPU via PyTorch (the API itself is CPU-only)."""
    import os
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")   # dev-only torch+faiss OpenMP clash on Windows
    import torch
    from models.feature_extractor import PatchFeatureExtractor

    if not torch.cuda.is_available():
        print("GPU: CUDA not available, skipping GPU reference.")
        return
    model = PatchFeatureExtractor().cuda().eval()
    x = torch.randn(1, 3, 224, 224, device="cuda")
    for _ in range(warmup):
        model(x)
    torch.cuda.synchronize()
    ts = []
    for _ in range(runs):
        t = time.perf_counter()
        model(x)
        torch.cuda.synchronize()        # GPU calls are async: wait, or you time only the launch
        ts.append(ms_since(t))
    print(f"GPU feature extractor (PyTorch, batch=1): median={np.median(ts):.2f} ms  "
          f"p95={np.percentile(ts, 95):.2f} ms")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=100)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--threads", type=int, default=0, help="0 = library default (all cores)")
    ap.add_argument("--categories", nargs="*", default=None)
    ap.add_argument("--gpu", action="store_true")
    args = ap.parse_args()

    cfg = load_config()
    categories = args.categories or cfg["categories"]

    opts = ort.SessionOptions()
    if args.threads > 0:
        opts.intra_op_num_threads = args.threads
        faiss.omp_set_num_threads(args.threads)
    onnx_path = cfg["paths"]["artifacts"] / "onnx" / "feature_extractor.onnx"
    session = ort.InferenceSession(str(onnx_path), opts, providers=["CPUExecutionProvider"])
    onnx_mb = onnx_path.stat().st_size / 1e6

    print(f"threads={'default' if args.threads == 0 else args.threads}  runs={args.runs}  "
          f"warmup={args.warmup}  batch=1  provider=CPU")
    rows = []
    for cat in categories:
        r = benchmark_category(cfg, cat, session, onnx_mb, args.runs, args.warmup)
        rows.append(r)
        print(f"[{cat:10s}] bank={r['bank_mb']:5.1f} MB | pre={r['preprocess_ms']:5.1f} "
              f"onnx={r['onnx_ms']:5.1f} faiss={r['faiss_ms']:5.1f} post={r['postprocess_ms']:5.1f} "
              f"| total median={r['total_ms_median']:6.1f} ms  p95={r['total_ms_p95']:6.1f} ms")

    df = pd.DataFrame(rows)
    num = df.drop(columns=["category"])
    mean_row = {"category": "mean", **num.mean().to_dict()}
    mean_row["bank_mb"] = df["bank_mb"].mean()
    total_row = {"category": "total_size", "bank_patches": int(df["bank_patches"].sum()),
                 "bank_mb": df["bank_mb"].sum(), "onnx_mb": onnx_mb}
    df = pd.concat([df, pd.DataFrame([mean_row, total_row])], ignore_index=True)

    out = cfg["paths"]["results"] / "tables" / "latency_size.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.round(3).to_csv(out, index=False)

    print(f"\nONNX file: {onnx_mb:.1f} MB | all banks: {df.loc[df.category == 'total_size', 'bank_mb'].iloc[0]:.1f} MB")
    print(f"mean total latency: median={mean_row['total_ms_median']:.1f} ms, p95={mean_row['total_ms_p95']:.1f} ms")
    print(f"saved to {out}")

    if args.gpu:
        gpu_reference(args.runs, args.warmup)


if __name__ == "__main__":
    main()
