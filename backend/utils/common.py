"""Seeding, config loading, paths. Stays torch-free at import time (serving has no torch)."""
import os
import random
from pathlib import Path

import numpy as np
import yaml

BACKEND_DIR = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = BACKEND_DIR / "configs" / "default.yaml"


def set_seed(seed: int = 0) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_config(path=None) -> dict:
    path = Path(path) if path else DEFAULT_CONFIG
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["paths"]["data_root"] = Path(os.environ.get("MVTEC_ROOT", cfg["paths"]["data_root"]))
    for key in ("checkpoints", "artifacts", "results"):
        cfg["paths"][key] = BACKEND_DIR / cfg["paths"][key]
    return cfg