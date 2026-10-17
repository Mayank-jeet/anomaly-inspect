"""FastAPI app. Run from backend/:  uvicorn api.main:app --reload

Endpoints: GET /health, GET /categories, GET /samples/{category}, POST /predict
"""
import json
import os
import urllib.error
from contextlib import asynccontextmanager
from pathlib import Path

import onnxruntime as ort
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from api.bank_cache import BankCache, ensure_file
from api.inference import Inferencer, InvalidImageError
from api.schemas import (CategoriesResponse, HealthResponse, PredictResponse,
                         SampleItem, SamplesResponse)
from utils.common import BACKEND_DIR, load_config

MAX_UPLOAD_BYTES = int(float(os.environ.get("MAX_UPLOAD_MB", "5")) * 1024 * 1024)
ALLOWED_TYPES = {"image/png", "image/jpeg"}
SAMPLES_DIR = BACKEND_DIR / "samples"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Runs once at startup: load the ONNX session and all (tiny) stats files. Memory banks are
    NOT loaded here; BankCache loads them lazily on first use."""
    cfg = load_config()
    art = Path(os.environ.get("ARTIFACTS_DIR", cfg["paths"]["artifacts"]))

    onnx_path = art / "onnx" / "feature_extractor.onnx"
    ensure_file(onnx_path, os.environ.get("ONNX_URL"))
    opts = ort.SessionOptions()
    threads = int(os.environ.get("ORT_THREADS", "0"))   # 0 = let onnxruntime decide
    if threads > 0:
        opts.intra_op_num_threads = threads
    session = ort.InferenceSession(str(onnx_path), opts, providers=["CPUExecutionProvider"])

    stats = {}
    for cat in cfg["categories"]:
        p = art / "stats" / f"{cat}.json"
        if p.exists():
            stats[cat] = json.loads(p.read_text())

    cache = BankCache(art / "banks",
                      maxsize=int(os.environ.get("BANK_CACHE_SIZE", "2")),
                      base_url=os.environ.get("BANKS_URL"))
    inferencer = Inferencer(session, stats, cache, sigma=cfg["patchcore"]["sigma"])
    inferencer.warmup()
    app.state.inferencer = inferencer
    yield


app = FastAPI(title="anomaly-inspect", lifespan=lifespan)

origins = [o.strip() for o in os.environ.get(
    "ALLOWED_ORIGINS", "http://localhost:5173,http://localhost:3000").split(",") if o.strip()]
app.add_middleware(CORSMiddleware, allow_origins=origins,
                   allow_methods=["GET", "POST"], allow_headers=["*"])

if SAMPLES_DIR.exists():
    app.mount("/sample-files", StaticFiles(directory=SAMPLES_DIR), name="sample-files")


@app.get("/health", response_model=HealthResponse)
def health():
    inf = app.state.inferencer
    return HealthResponse(status="ok", onnx_loaded=inf.session is not None, categories=len(inf.stats))


@app.get("/categories", response_model=CategoriesResponse)
def categories():
    return CategoriesResponse(categories=sorted(app.state.inferencer.stats.keys()))


@app.get("/samples/{category}", response_model=SamplesResponse)
def samples(category: str):
    if category not in app.state.inferencer.stats:
        raise HTTPException(status_code=422, detail=f"Unknown category '{category}'")
    items = []
    for kind in ("good", "defective"):
        d = SAMPLES_DIR / category / kind
        if d.exists():
            for f in sorted(d.iterdir()):
                if f.suffix.lower() in (".png", ".jpg", ".jpeg"):
                    items.append(SampleItem(name=f.name, kind=kind,
                                            url=f"/sample-files/{category}/{kind}/{f.name}"))
    return SamplesResponse(category=category, samples=items)


# A plain `def` (not `async def`): FastAPI runs it in a thread pool, so the CPU-bound ONNX + FAISS
# work does not block the event loop (which would freeze /health and every other request).
@app.post("/predict", response_model=PredictResponse)
def predict(file: UploadFile = File(...), category: str = Form(...)):
    inf = app.state.inferencer
    if category not in inf.stats:
        raise HTTPException(status_code=422, detail=f"Unknown category '{category}'")
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(status_code=400, detail="Only PNG or JPEG images are accepted")

    data = file.file.read(MAX_UPLOAD_BYTES + 1)   # read one byte past the limit to detect oversize
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413,
                            detail=f"File too large (max {MAX_UPLOAD_BYTES // (1024 * 1024)} MB)")

    try:
        return inf.predict(data, category)
    except InvalidImageError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except (FileNotFoundError, urllib.error.URLError):
        # The category's memory bank is missing and could not be downloaded (bad BANKS_URL, no network).
        raise HTTPException(status_code=503,
                            detail="Model data for this category is temporarily unavailable. Please try again later.")
