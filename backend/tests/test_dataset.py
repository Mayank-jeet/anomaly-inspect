import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from utils.common import load_config
from data.mvtec import MVTecDataset

cfg = load_config()
ROOT = cfg["paths"]["data_root"]

def test_shapes():
    ds = MVTecDataset(ROOT, "bottle", "train")
    item = ds[0]
    assert item["image"].shape == (3, 224, 224)
    assert item["mask"].shape == (1, 224, 224)

def test_label_matches_mask():
    ds = MVTecDataset(ROOT, "bottle", "test")
    for i in range(len(ds)):
        item = ds[i]
        assert item["label"] == int(item["mask"].numpy().any())

def test_no_overlap_between_splits():
    train = {p for p, _, _ in MVTecDataset(ROOT, "bottle", "train").samples}
    val = {p for p, _, _ in MVTecDataset(ROOT, "bottle", "val").samples}
    test = {p for p, _, _ in MVTecDataset(ROOT, "bottle", "test").samples}
    assert train.isdisjoint(val)
    assert train.isdisjoint(test)
    assert val.isdisjoint(test)