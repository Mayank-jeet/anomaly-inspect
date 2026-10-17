"""Response models. They define the JSON contract between the API and the React app,
and FastAPI also uses them to validate responses and generate the /docs page."""
from typing import List, Optional

from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str
    onnx_loaded: bool
    categories: int


class CategoriesResponse(BaseModel):
    categories: List[str]


class SampleItem(BaseModel):
    name: str
    kind: str   # "good" or "defective"
    url: str    # relative to the API base URL


class SamplesResponse(BaseModel):
    category: str
    samples: List[SampleItem]


class PredictResponse(BaseModel):
    category: str
    # True when the image does not look like the setup this category's model was trained on
    # (calibrated on normal validation images only). Then no verdict or heatmap is given.
    out_of_distribution: bool
    ood_score: Optional[float] = None       # min patch distance of this image (best-matching patch)
    ood_threshold: Optional[float] = None   # cutoff for ood_score
    score: float                 # image anomaly score (max of the blurred score map)
    threshold: float             # from calibrate.py (normal validation images only)
    is_defective: Optional[bool] = None       # score > threshold; None if out of distribution
    heatmap_png_base64: Optional[str] = None  # 224x224 colored anomaly map; None if out of distribution
    input_png_base64: str        # the 224x224 crop the model actually saw (heatmap aligns with this)
    latency_ms: float            # server-side time for the whole request
