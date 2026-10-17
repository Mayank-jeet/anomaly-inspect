"""API tests. Run from backend/:  python -m pytest tests/test_api.py -v

The 'predict' tests need the MVTec test images and the built artifacts (ONNX, banks, stats).
"""
import os

# Dev-only: torch (used in one test to get the reference score) and faiss each bundle an OpenMP
# runtime on Windows; this lets them share a process. Must be set before they are imported.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ["BANK_CACHE_SIZE"] = "2"

import base64
import io

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from api.main import app
from utils.common import load_config

CFG = load_config()
ROOT = CFG["paths"]["data_root"]
needs_data = pytest.mark.skipif(not ROOT.exists(), reason="MVTec AD dataset not found")


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:   # `with` runs the lifespan (loads ONNX + stats)
        yield c


def _png_bytes(size=(64, 64)):
    buf = io.BytesIO()
    Image.new("RGB", size, (120, 120, 120)).save(buf, format="PNG")
    return buf.getvalue()


def _first_test_image(category, good=True):
    d = ROOT / category / "test"
    files = sorted((d / "good").glob("*.png")) if good else sorted(
        p for p in d.rglob("*.png") if p.parent.name != "good")
    return files[0]


def _post(client, data, category, content_type="image/png", name="x.png"):
    return client.post("/predict", files={"file": (name, data, content_type)}, data={"category": category})


# ---------- simple endpoints ----------

def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok" and r.json()["onnx_loaded"] is True


def test_categories(client):
    cats = client.get("/categories").json()["categories"]
    assert len(cats) == 15 and "bottle" in cats


def test_samples_unknown_category(client):
    assert client.get("/samples/not_a_category").status_code == 422


# ---------- validation: 400 / 413 / 422 ----------

def test_unknown_category_422(client):
    assert _post(client, _png_bytes(), "not_a_category").status_code == 422


def test_wrong_content_type_400(client):
    assert _post(client, b"hello", "bottle", content_type="text/plain", name="a.txt").status_code == 400


def test_corrupt_image_400(client):
    assert _post(client, b"this is not an image", "bottle").status_code == 400


def test_too_large_413(client):
    assert _post(client, b"0" * (6 * 1024 * 1024), "bottle").status_code == 413


# ---------- real predictions ----------

@needs_data
def test_predict_response_shape(client):
    path = _first_test_image("bottle", good=True)
    r = _post(client, path.read_bytes(), "bottle")
    assert r.status_code == 200
    body = r.json()
    assert set(body) == {"category", "score", "threshold", "is_defective",
                         "heatmap_png_base64", "input_png_base64", "latency_ms"}
    assert body["category"] == "bottle" and body["latency_ms"] > 0
    assert body["is_defective"] == (body["score"] > body["threshold"])
    for key in ("heatmap_png_base64", "input_png_base64"):
        img = Image.open(io.BytesIO(base64.b64decode(body[key])))
        assert img.size == (224, 224)


@needs_data
def test_served_score_matches_evaluate(client):
    """The key correctness check: the API's score equals the PyTorch path that evaluate.py uses."""
    torch = pytest.importorskip("torch")
    from data.transforms import preprocess_image
    from models.feature_extractor import PatchFeatureExtractor
    from models.patchcore import PatchCoreScorer

    extractor = PatchFeatureExtractor().eval()
    for cat, good in (("bottle", False), ("screw", True)):
        path = _first_test_image(cat, good=good)
        served = _post(client, path.read_bytes(), cat).json()["score"]

        bank = np.load(CFG["paths"]["artifacts"] / "banks" / f"{cat}.npy")
        x = torch.from_numpy(preprocess_image(Image.open(path)))[None]
        with torch.no_grad():
            patches = extractor.extract_patches(x).numpy()
        expected, _ = PatchCoreScorer(bank).score_image(patches, sigma=CFG["patchcore"]["sigma"])
        assert abs(served - expected) < 1e-3, (cat, served, expected)


@needs_data
def test_bank_cache_stays_bounded(client):
    """Request 3 categories with maxsize=2: only the 2 most recent banks may stay in RAM."""
    for cat in ("bottle", "screw", "cable"):
        assert _post(client, _first_test_image(cat).read_bytes(), cat).status_code == 200
    cache = app.state.inferencer.cache
    assert len(cache) <= 2
    assert cache.cached() == ["screw", "cable"]
