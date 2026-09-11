import sys
import numpy as np
from PyQt6.QtWidgets import QApplication

from dg_patch_tool.siril_bridge import SirilBridge
from dg_patch_tool.ui.main_window import MainWindow


_STARTUP_BANNER_LINES = [
    "================================================",
    "DG Patch Tool",
    "Author: Daian Gan",
    "Email:  daian@ganmedia.com",
    "Web:    https://daiangan.com",
    "================================================",
]


def run(image_array: np.ndarray = None):
    """
    Main application entry point.
    If image_array is provided, uses it directly.
    Otherwise attempts to fetch active image from Siril via SirilBridge.
    If no active Siril session is detected, loads a realistic synthetic starless image for testing.
    """
    bridge = SirilBridge()

    # Output startup banner to Siril console log
    for line in _STARTUP_BANNER_LINES:
        bridge.log(line)

    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)

    app.setStyle("Fusion")

    if image_array is None:
        if bridge.is_connected():
            image_array = bridge.get_image_pixeldata()

    if image_array is None:
        # Fallback to synthetic starless test image
        image_array = bridge.generate_synthetic_starless()

    window = MainWindow(raw_image=image_array, siril_bridge=bridge)
    window.show()

    return app.exec()


if __name__ == "__main__":
    sys.exit(run())
