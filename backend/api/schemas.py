"""Response models. They define the JSON contract between the API and the React app,
and FastAPI also uses them to validate responses and generate the /docs page."""
from typing import List

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
    score: float                 # image anomaly score (max of the blurred score map)
    threshold: float             # from calibrate.py (normal validation images only)
    is_defective: bool           # score > threshold
    heatmap_png_base64: str      # 224x224 colored anomaly map
    input_png_base64: str        # the 224x224 crop the model actually saw (heatmap aligns with this)
    latency_ms: float            # server-side time for the whole request
