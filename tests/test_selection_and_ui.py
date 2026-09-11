from PyQt6.QtCore import QPointF
from PyQt6.QtGui import QPolygonF
from dg_patch_tool.ui.selection_tools import SelectionTool


def test_rectangle_selection():
    p1 = QPointF(10, 15)
    p2 = QPointF(60, 85)
    poly = SelectionTool.rect_to_polygon(p1, p2)
    assert poly.count() == 5

    data = SelectionTool.polygon_to_mask(poly, img_width=100, img_height=100)
    assert data is not None
    assert data.x == 10
    assert data.y == 15
    assert data.w == 50
    assert data.h == 70
    assert data.mask.shape == (70, 50)
    # Mask inside rectangle should be 255
    assert data.mask[35, 25] == 255


def test_lasso_selection():
    poly = QPolygonF([
        QPointF(20, 20),
        QPointF(40, 20),
        QPointF(40, 40),
        QPointF(20, 40),
        QPointF(20, 20),
    ])
    data = SelectionTool.polygon_to_mask(poly, img_width=100, img_height=100)
    assert data is not None
    assert data.x == 20
    assert data.y == 20
    assert data.w == 20
    assert data.h == 20
    assert data.mask[10, 10] == 255


def test_empty_selection():
    poly = QPolygonF()
    data = SelectionTool.polygon_to_mask(poly, 100, 100)
    assert data is None


def test_toolbar_default_algorithm():
    from dg_patch_tool.core.blend_engine import BlendEngine
    from dg_patch_tool.ui.toolbar import PatchToolBar
    from PyQt6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    tb = PatchToolBar()

    # First item and current selection must be "Poisson (Normal Clone)"
    assert BlendEngine.ALL_MODES[0] == BlendEngine.MODE_POISSON_NORMAL
    assert tb.combo_algo.itemText(0) == BlendEngine.MODE_POISSON_NORMAL
    assert tb.get_current_algorithm() == BlendEngine.MODE_POISSON_NORMAL
