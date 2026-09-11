import numpy as np
from dg_patch_tool.core.safety_check import check_linear_data


def test_linear_image_detection():
    # Synthetic linear image: background sky ~0.005, 95% pixels < 0.02
    np.random.seed(42)
    linear_img = np.random.exponential(scale=0.003, size=(200, 200, 3)).astype(np.float32)
    linear_img = np.clip(linear_img, 0.0, 1.0)

    is_linear, msg, stats = check_linear_data(linear_img)
    assert is_linear is True
    assert "doesn't look stretched" in msg
    assert stats["median"] < 0.02


def test_stretched_image_detection():
    # Stretched image: sky background pushed to 0.15, median > 0.10
    np.random.seed(42)
    stretched_img = 0.15 + np.random.normal(0, 0.02, size=(200, 200, 3)).astype(np.float32)
    stretched_img = np.clip(stretched_img, 0.0, 1.0)

    is_linear, msg, stats = check_linear_data(stretched_img)
    assert is_linear is False
    assert msg == ""
    assert stats["median"] > 0.05
