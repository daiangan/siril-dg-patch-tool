import pytest
from PyQt6.QtWidgets import QApplication
from dg_patch_tool.core.settings import (
    load_user_preferences,
    save_user_preferences,
    get_settings,
)
from dg_patch_tool.core.blend_engine import BlendEngine
from dg_patch_tool.ui.toolbar import PatchToolBar
from dg_patch_tool.ui.selection_tools import SelectionTool


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield app


@pytest.fixture(autouse=True)
def isolate_settings():
    """Ensures tests do not overwrite the user's persistent preferences."""
    s = get_settings()
    keys = ["tool", "algorithm", "opacity", "feather"]
    saved = {k: s.value(k) for k in keys if s.contains(k)}
    yield
    for k in keys:
        if k in saved:
            s.setValue(k, saved[k])
        else:
            s.remove(k)
    s.sync()


def test_settings_save_and_load(qapp):
    # Test round-trip persistence with 'rect'
    save_user_preferences(
        tool="rect",
        algorithm=BlendEngine.MODE_LAPLACIAN,
        opacity=0.65,
        feather=8,
    )

    prefs = load_user_preferences()
    assert prefs["tool"] == "rect"
    assert prefs["algorithm"] == BlendEngine.MODE_LAPLACIAN
    assert abs(prefs["opacity"] - 0.65) < 0.01
    assert prefs["feather"] == 8

    # Test round-trip persistence with 'rectangle'
    save_user_preferences(
        tool="rectangle",
        algorithm=BlendEngine.MODE_POISSON_MIXED,
        opacity=0.9,
        feather=5,
    )
    prefs2 = load_user_preferences()
    assert prefs2["tool"] == "rect"


def test_settings_validation_and_clamping(qapp):
    s = get_settings()

    # Invalid tool -> fallback to lasso
    s.setValue("tool", "invalid_tool")
    # Invalid algorithm -> fallback to Poisson (Normal Clone)
    s.setValue("algorithm", "NonExistentAlgo")
    # Out of bounds opacity
    s.setValue("opacity", 2.5)
    # Out of bounds feather
    s.setValue("feather", 50)
    s.sync()

    prefs = load_user_preferences()
    assert prefs["tool"] == "lasso"
    assert prefs["algorithm"] == BlendEngine.MODE_POISSON_NORMAL
    assert prefs["opacity"] == 1.0
    assert prefs["feather"] == 20


def test_toolbar_settings_integration(qapp):
    tb = PatchToolBar()

    # Apply settings using SelectionTool constant
    tb.set_current_tool(SelectionTool.TOOL_RECTANGLE)
    tb.set_current_algorithm(BlendEngine.MODE_POISSON_MIXED)
    tb.set_current_opacity(0.42)
    tb.set_current_feather(12)

    assert tb.get_current_tool() == SelectionTool.TOOL_RECTANGLE
    assert tb.btn_rect.isChecked() is True
    assert tb.btn_lasso.isChecked() is False
    assert tb.get_current_algorithm() == BlendEngine.MODE_POISSON_MIXED
    assert abs(tb.get_current_opacity() - 0.42) < 0.01
    assert tb.get_current_feather() == 12
    assert tb.lbl_feather.text() == "12 px"
    assert tb.lbl_opacity.text() == "42%"

    # Test set_current_tool with "rect" and "rectangle"
    tb.set_current_tool("lasso")
    assert tb.get_current_tool() == "lasso"
    assert tb.btn_lasso.isChecked() is True

    tb.set_current_tool("rect")
    assert tb.get_current_tool() == "rect"
    assert tb.btn_rect.isChecked() is True

    tb.set_current_tool("lasso")
    tb.set_current_tool("rectangle")
    assert tb.get_current_tool() == "rect"
    assert tb.btn_rect.isChecked() is True
