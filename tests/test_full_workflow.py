import pytest
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import QPointF
from PyQt6.QtGui import QPolygonF
import numpy as np

from dg_patch_tool.siril_bridge import SirilBridge
from dg_patch_tool.ui.main_window import MainWindow
from dg_patch_tool.ui.selection_tools import SelectionTool
from dg_patch_tool.core.blend_engine import BlendEngine


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield app


def test_full_gui_workflow(qapp):
    # 1. Create realistic synthetic starless image
    bridge = SirilBridge()
    synthetic_img = bridge.generate_synthetic_starless()
    assert synthetic_img.shape == (800, 1200, 3)

    # 2. Instantiate MainWindow
    win = MainWindow(raw_image=synthetic_img, siril_bridge=bridge)
    assert win is not None
    assert win.image_buffer.width == 1200
    assert win.image_buffer.height == 800

    # 3. Simulate selection around the artifact at (350, 260)
    poly = QPolygonF([
        QPointF(330, 240),
        QPointF(370, 240),
        QPointF(370, 280),
        QPointF(330, 280),
        QPointF(330, 240),
    ])
    sel_data = SelectionTool.polygon_to_mask(poly, 1200, 800)
    assert sel_data is not None

    win.canvas_view.set_selection(sel_data)
    assert win.canvas_view.state == win.canvas_view.STATE_PATCHING

    # 4. Simulate moving cursor to a clean area at offset (dx=50, dy=50)
    preview_pixmap = win._calculate_preview_roi(sel_data, dx=50, dy=50)
    assert preview_pixmap is not None
    assert not preview_pixmap.isNull()
    assert preview_pixmap.width() == sel_data.w
    assert preview_pixmap.height() == sel_data.h

    # 5. Adjust opacity to 80%
    win.toolbar.slider_opacity.setValue(80)
    assert win.toolbar.get_current_opacity() == 0.8
    win._update_cached_preview_display()

    # 6. Confirm patch
    original_val = win.image_buffer.get_roi(340, 250, 5, 5).copy()
    win._on_patch_confirmed(sel_data, (50, 50))

    # Selection should now be cleared and ready for next action
    assert win.canvas_view.state == win.canvas_view.STATE_READY
    assert win.history.can_undo() is True
    assert not win.history.can_redo()

    # 7. Undo patch
    win._on_undo()
    assert not win.history.can_undo()
    assert win.history.can_redo() is True

    # 8. Redo patch
    win._on_redo()
    assert win.history.can_undo() is True

    # 9. Test Auto-Heal (ShiftMap automatically executes on selection)
    win.toolbar.combo_algo.setCurrentText(BlendEngine.MODE_SHIFTMAP)
    win.canvas_view.set_selection(sel_data)
    win._on_selection_made(sel_data)
    assert win._cached_preview_data is not None

    # Confirm auto-heal patch
    win._on_patch_confirmed(sel_data, (1, 1))
    assert win.canvas_view.state == win.canvas_view.STATE_READY

    # 10. Test Export for Siril
    export = win.image_buffer.export_for_siril()
    assert export.shape == synthetic_img.shape
    assert export.dtype == synthetic_img.dtype

    win._has_unapplied_changes = False
    win.close()


def test_siril_bridge_undo_save_state():
    class MockSiril:
        def __init__(self):
            self.undo_calls = []
            self.pixeldata_calls = []

        def undo_save_state(self, msg: str):
            self.undo_calls.append(msg)
            return True

        def set_image_pixeldata(self, data):
            self.pixeldata_calls.append(data)
            return True

    bridge = SirilBridge()
    mock_siril = MockSiril()
    bridge.siril = mock_siril
    bridge.connected = True

    dummy_arr = np.zeros((10, 10, 3), dtype=np.float32)
    success = bridge.set_image_pixeldata(dummy_arr)

    assert success is True
    assert len(mock_siril.undo_calls) == 1
    assert mock_siril.undo_calls[0] == "DG_Patch_Tool"
    assert len(mock_siril.pixeldata_calls) == 1


def test_large_roi_downsampled_preview_and_full_confirm(qapp):
    """Tests adaptive downsampling on large ROI (>256px) and full-res confirmation."""
    bridge = SirilBridge()
    synthetic_img = bridge.generate_synthetic_starless()
    win = MainWindow(raw_image=synthetic_img, siril_bridge=bridge)

    # Large selection: 350x300 pixels (exceeds MAX_PREVIEW_DIM = 256)
    p1 = QPointF(100, 100)
    p2 = QPointF(450, 400)
    poly = SelectionTool.rect_to_polygon(p1, p2)
    sel_data = SelectionTool.polygon_to_mask(poly, 1200, 800)
    assert sel_data is not None
    assert sel_data.w == 350
    assert sel_data.h == 300

    win.canvas_view.set_selection(sel_data)

    # Calculate preview at offset dx=100, dy=100
    pixmap = win._calculate_preview_roi(sel_data, dx=100, dy=100)
    assert pixmap is not None
    assert not pixmap.isNull()
    # Visual pixmap must match destination selection size for seamless display
    assert pixmap.width() == sel_data.w
    assert pixmap.height() == sel_data.h

    # Verify that adaptive downsampling was indeed used
    assert win._cached_preview_data is not None
    assert win._cached_preview_data["is_downsampled"] is True
    assert win._cached_preview_data["dst_roi"].shape[0] <= win.MAX_PREVIEW_DIM
    assert win._cached_preview_data["dst_roi"].shape[1] <= win.MAX_PREVIEW_DIM

    # Test opacity slider update on downsampled preview
    win.toolbar.slider_opacity.setValue(70)
    win._update_cached_preview_display()

    # Confirm patch
    sample_before = win.image_buffer.get_roi(sel_data.x + 20, sel_data.y + 20, 10, 10).copy()
    win._on_patch_confirmed(sel_data, (100, 100))

    # Verify committed at full resolution
    sample_after = win.image_buffer.get_roi(sel_data.x + 20, sel_data.y + 20, 10, 10).copy()
    assert not np.array_equal(sample_before, sample_after)
    assert win.history.can_undo() is True

    # Undo
    win._on_undo()
    sample_restored = win.image_buffer.get_roi(sel_data.x + 20, sel_data.y + 20, 10, 10).copy()
    np.testing.assert_allclose(sample_before, sample_restored, atol=1e-5)

    win._has_unapplied_changes = False
    win.close()

