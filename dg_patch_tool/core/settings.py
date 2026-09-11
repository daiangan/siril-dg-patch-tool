"""
User configuration and settings persistence for DG_Patch_Tool.
Persists tool, algorithm, opacity, and feather settings across sessions using QSettings.
"""

from PyQt6.QtCore import QSettings
from dg_patch_tool.core.blend_engine import BlendEngine

ORG_NAME = "DaianGan"
APP_NAME = "DG_Patch_Tool"


def get_settings() -> QSettings:
    """Returns the QSettings instance for DG_Patch_Tool."""
    return QSettings(ORG_NAME, APP_NAME)


def load_user_preferences() -> dict:
    """
    Loads persisted user preferences with sensible fallback defaults.

    Returns:
        dict with keys: 'tool', 'algorithm', 'opacity', 'feather'
    """
    s = get_settings()

    # Tool selection
    tool = str(s.value("tool", "lasso")).lower()
    if tool in ("rect", "rectangle"):
        tool = "rect"
    else:
        tool = "lasso"

    # Algorithm selection
    algo = str(s.value("algorithm", BlendEngine.MODE_POISSON_NORMAL))
    if algo not in BlendEngine.ALL_MODES:
        algo = BlendEngine.MODE_POISSON_NORMAL

    # Opacity (0.0 - 1.0)
    try:
        val = s.value("opacity", 1.0)
        opacity = float(val)
        opacity = max(0.0, min(1.0, opacity))
    except (ValueError, TypeError):
        opacity = 1.0

    # Feather (0 - 20)
    try:
        val = s.value("feather", 3)
        feather = int(val)
        feather = max(0, min(20, feather))
    except (ValueError, TypeError):
        feather = 3

    return {
        "tool": tool,
        "algorithm": algo,
        "opacity": opacity,
        "feather": feather,
    }


def save_user_preferences(tool: str, algorithm: str, opacity: float, feather: int):
    """
    Saves user preferences to persistent storage.

    Args:
        tool: 'lasso' or 'rect'
        algorithm: Blend algorithm mode name string
        opacity: Blend opacity (0.0 to 1.0)
        feather: Feather radius in pixels (0 to 20)
    """
    s = get_settings()
    tool_str = str(tool).lower()
    tool_val = "rect" if tool_str in ("rect", "rectangle") else "lasso"
    s.setValue("tool", tool_val)
    s.setValue("algorithm", algorithm)
    s.setValue("opacity", float(opacity))
    s.setValue("feather", int(feather))
    s.sync()
