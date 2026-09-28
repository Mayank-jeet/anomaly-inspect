import numpy as np
from PIL import Image
from data.transforms import preprocess_image, preprocess_mask

def test_image_shape_and_dtype():
    img = Image.new("RGB", (300, 200), (128, 64, 32))
    out = preprocess_image(img)
    assert out.shape == (3, 224, 224)
    assert out.dtype == np.float32

def test_mask_binary():
    mask = Image.new("L", (300, 200), 200)
    out = preprocess_mask(mask)
    assert out.shape == (1, 224, 224)
    assert set(np.unique(out)).issubset({0.0, 1.0})

def test_grayscale_input_converts_to_rgb():
    img = Image.new("L", (300, 200), 100)
    out = preprocess_image(img)
    assert out.shape == (3, 224, 224)