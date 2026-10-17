"""Lazy LRU cache of PatchCore memory banks.

Why: all 15 banks total ~394 MB (see results/tables/latency_size.csv) and a free host has
512 MB RAM for everything. So we load a bank only when a category is first requested and keep
the `maxsize` most recently used ones; the least recently used one is dropped.
"""
import threading
import urllib.request
from collections import OrderedDict
from pathlib import Path
from typing import Optional

import numpy as np

from models.patchcore import PatchCoreScorer


def ensure_file(path: Path, url: Optional[str]) -> None:
    """Make sure `path` exists; download it from `url` if missing (used for deployment, where
    big artifacts are not in git). Writes to a .part file first so a crash never leaves a
    half-downloaded file that looks valid."""
    if path.exists():
        return
    if not url:
        raise FileNotFoundError(f"{path} is missing and no download URL is configured")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    urllib.request.urlretrieve(url, tmp)
    tmp.replace(path)


class BankCache:
    def __init__(self, banks_dir: Path, maxsize: int = 2, base_url: Optional[str] = None):
        self.banks_dir = Path(banks_dir)
        self.maxsize = maxsize
        self.base_url = base_url.rstrip("/") if base_url else None
        self._items: "OrderedDict[str, PatchCoreScorer]" = OrderedDict()
        # /predict runs in a thread pool, so several requests can touch the cache at once.
        self._lock = threading.Lock()

    def __len__(self) -> int:
        return len(self._items)

    def cached(self):
        return list(self._items.keys())

    def get(self, category: str) -> PatchCoreScorer:
        with self._lock:
            if category in self._items:
                self._items.move_to_end(category)      # mark as most recently used
                return self._items[category]

            # Loading inside the lock on purpose: two requests for the same new category must not
            # both load a 25-40 MB bank on a memory-limited host.
            path = self.banks_dir / f"{category}.npy"
            url = f"{self.base_url}/{category}.npy" if self.base_url else None
            ensure_file(path, url)
            scorer = PatchCoreScorer(np.load(path))     # bank array is freed once FAISS has its copy
            self._items[category] = scorer
            while len(self._items) > self.maxsize:
                self._items.popitem(last=False)         # evict least recently used
            return scorer
