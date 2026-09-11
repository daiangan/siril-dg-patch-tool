import numpy as np
from dg_patch_tool.core.image_buffer import ImageBuffer
from dg_patch_tool.core.history import HistoryManager


def test_image_buffer_hwc():
    arr = np.random.uniform(0.0, 1.0, (100, 150, 3)).astype(np.float32)
    buf = ImageBuffer(arr)

    assert buf.width == 150
    assert buf.height == 100
    assert buf.is_color is True
    assert buf.is_chw is False

    roi = buf.get_roi(10, 20, 30, 40)
    assert roi.shape == (40, 30, 3)

    new_val = np.zeros((40, 30, 3), dtype=np.float32)
    buf.set_roi(10, 20, 30, 40, new_val)
    check_roi = buf.get_roi(10, 20, 30, 40)
    assert np.allclose(check_roi, 0.0)

    exported = buf.export_for_siril()
    assert exported.shape == (100, 150, 3)
    assert exported.dtype == np.float32


def test_image_buffer_chw():
    # Siril sometimes returns (C, H, W)
    arr = np.random.uniform(0.0, 1.0, (3, 80, 120)).astype(np.float32)
    buf = ImageBuffer(arr)

    assert buf.width == 120
    assert buf.height == 80
    assert buf.is_chw is True

    exported = buf.export_for_siril()
    assert exported.shape == (3, 80, 120)


def test_history_undo_redo():
    arr = np.zeros((50, 50, 3), dtype=np.float32)
    buf = ImageBuffer(arr)
    history = HistoryManager(max_depth=10)

    assert not history.can_undo()
    assert not history.can_redo()

    before = buf.get_roi(10, 10, 20, 20)
    after = np.ones((20, 20, 3), dtype=np.float32)
    mask = np.ones((20, 20), dtype=np.uint8)

    buf.set_roi(10, 10, 20, 20, after)
    history.record_patch(10, 10, 20, 20, before, after, mask, "Poisson", 1.0)

    assert history.can_undo()
    assert np.allclose(buf.get_roi(10, 10, 20, 20), 1.0)

    # Undo
    history.undo(buf)
    assert np.allclose(buf.get_roi(10, 10, 20, 20), 0.0)
    assert history.can_redo()

    # Redo
    history.redo(buf)
    assert np.allclose(buf.get_roi(10, 10, 20, 20), 1.0)


def test_fits_orientation_flip_and_roundtrip():
    # In FITS, row 0 is bottom, row H-1 is top.
    # Create an asymmetrical image: row 0 (bottom in FITS) has value 42,
    # row -1 (top in FITS) has value 99.
    arr = np.zeros((50, 50), dtype=np.float32)
    arr[0, :] = 42.0   # FITS bottom row
    arr[-1, :] = 99.0  # FITS top row

    buf = ImageBuffer(arr)

    # In Qt / screen space, top of the image is row 0.
    # Therefore, buf.data[0, :] must be the FITS top row (99.0)
    assert np.allclose(buf.data[0, :], 99.0)
    # And buf.data[-1, :] must be the FITS bottom row (42.0)
    assert np.allclose(buf.data[-1, :], 42.0)

    # Exporting back for Siril must invert the flip, perfectly matching original FITS layout
    exported = buf.export_for_siril()
    assert np.allclose(exported, arr)
    assert np.allclose(exported[0, :], 42.0)
    assert np.allclose(exported[-1, :], 99.0)
