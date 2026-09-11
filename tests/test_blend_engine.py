import numpy as np
import cv2
from dg_patch_tool.core.blend_engine import BlendEngine


def test_feather_mask():
    mask = np.zeros((50, 50), dtype=np.uint8)
    mask[15:35, 15:35] = 255

    feathered = BlendEngine.feather_mask(mask, feather_radius=5)
    assert feathered.shape == (50, 50)
    assert feathered.dtype == np.float32
    assert np.max(feathered) == 1.0
    assert np.min(feathered) == 0.0
    # Edge should have intermediate values
    assert 0.1 < feathered[15, 25] < 0.9


def test_poisson_mixed_clone_float32():
    # Destination has background sky
    h, w = 80, 80
    dst = np.full((h, w, 3), 0.2, dtype=np.float32)
    # Source has a bright nebulosity feature with gradients
    src = np.full((h, w, 3), 0.2, dtype=np.float32)
    y, x = np.ogrid[:h, :w]
    feature = 0.6 * np.exp(-((x - 40)**2 + (y - 40)**2) / 50.0)
    for c in range(3):
        src[:, :, c] += feature

    mask = np.zeros((h, w), dtype=np.uint8)
    mask[20:60, 20:60] = 255

    blended = BlendEngine.blend_poisson(dst, src, mask, cv2.MIXED_CLONE)
    assert blended.shape == (h, w, 3)
    assert blended.dtype == np.float32
    # The center should contain the cloned feature
    assert blended[40, 40, 0] > 0.45


def test_poisson_mono_uint16():
    h, w = 60, 60
    dst = np.full((h, w), 10000, dtype=np.uint16)
    src = np.full((h, w), 10000, dtype=np.uint16)
    # Add gradient feature to src
    y, x = np.ogrid[:h, :w]
    feature = (30000 * np.exp(-((x - 30)**2 + (y - 30)**2) / 40.0)).astype(np.uint16)
    src += feature

    mask = np.zeros((h, w), dtype=np.uint8)
    mask[15:45, 15:45] = 255

    blended = BlendEngine.blend_poisson(dst, src, mask, cv2.MIXED_CLONE)
    assert blended.shape == (h, w)
    assert blended.dtype == np.uint16
    assert blended[30, 30] > 20000


def test_laplacian_pyramid_blending():
    h, w = 64, 64
    dst = np.full((h, w, 3), 0.15, dtype=np.float32)
    src = np.full((h, w, 3), 0.40, dtype=np.float32)

    mask = np.zeros((h, w), dtype=np.uint8)
    mask[16:48, 16:48] = 255

    blended = BlendEngine.blend_laplacian_pyramid(dst, src, mask, feather_radius=3, num_levels=3)
    assert blended.shape == (h, w, 3)
    assert blended.dtype == np.float32
    # In the center, blended should be close to src
    assert np.isclose(blended[32, 32, 0], 0.40, atol=0.08)
    # At the outer boundary, blended should be dst
    assert np.isclose(blended[2, 2, 0], 0.15, atol=0.01)


def test_auto_heal_shiftmap():
    h, w = 70, 70
    dst = np.full((h, w, 3), 0.25, dtype=np.float32)
    # Add an artifact in the center
    dst[30:40, 30:40] = 0.95

    mask = np.zeros((h, w), dtype=np.uint8)
    mask[28:42, 28:42] = 255

    inpainted = BlendEngine.inpaint_shiftmap(dst, mask)
    assert inpainted.shape == (h, w, 3)
    assert inpainted.dtype == np.float32
    # The center artifact should be repaired (lowered significantly toward 0.25)
    assert np.mean(inpainted[30:40, 30:40]) < 0.85


def test_opacity_alpha_interpolation():
    h, w = 40, 40
    dst = np.full((h, w, 3), 0.2, dtype=np.float32)
    patched = np.full((h, w, 3), 0.8, dtype=np.float32)
    mask = np.ones((h, w), dtype=np.float32)

    # 100% opacity -> patched
    res100 = BlendEngine.apply_opacity(dst, patched, mask, 1.0)
    assert np.allclose(res100, 0.8)

    # 0% opacity -> dst
    res0 = BlendEngine.apply_opacity(dst, patched, mask, 0.0)
    assert np.allclose(res0, 0.2)

    # 50% opacity -> 0.5 * 0.2 + 0.5 * 0.8 = 0.5
    res50 = BlendEngine.apply_opacity(dst, patched, mask, 0.5)
    assert np.allclose(res50, 0.5)
