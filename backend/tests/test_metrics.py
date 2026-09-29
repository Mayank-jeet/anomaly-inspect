import numpy as np
import pytest
from utils.metrics import image_auroc, pixel_auroc
from utils.postprocess import upsample_map, blur_map, normalize_map

rng = np.random.default_rng(0)

def test_image_auroc_random_about_half():
    labels = rng.integers(0, 2, 5000)
    scores = rng.random(5000)
    assert abs(image_auroc(labels, scores) - 0.5) < 0.05

def test_image_auroc_perfect_and_inverted():
    labels = np.array([0, 0, 0, 1, 1, 1])
    scores = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    assert image_auroc(labels, scores) == 1.0
    assert image_auroc(labels, -scores) == 0.0

def test_pixel_auroc_perfect_and_random():
    masks = np.zeros((4, 32, 32)); masks[:, 8:16, 8:16] = 1
    assert pixel_auroc(masks, masks.astype(float)) == 1.0
    assert abs(pixel_auroc(masks, rng.random(masks.shape)) - 0.5) < 0.05

def test_pixel_auroc_accepts_channel_dim():
    masks = np.zeros((2, 1, 16, 16)); masks[:, :, 4:8, 4:8] = 1
    assert pixel_auroc(masks, masks.astype(float)) == 1.0

def test_single_class_raises():
    with pytest.raises(ValueError):
        image_auroc([0, 0, 0], [0.1, 0.2, 0.3])

def test_postprocess_shapes_and_range():
    m = rng.random((28, 28)).astype(np.float32)
    up = upsample_map(m, 224)
    assert up.shape == (224, 224)
    assert blur_map(up).shape == (224, 224)
    n = normalize_map(up, 0.0, 0.5)
    assert n.min() >= 0.0 and n.max() <= 1.0