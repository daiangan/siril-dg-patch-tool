from PyQt6.QtCore import Qt, pyqtSignal, QUrl
from PyQt6.QtGui import QIcon, QKeySequence, QShortcut, QDesktopServices
from PyQt6.QtWidgets import (
    QToolBar,
    QComboBox,
    QSlider,
    QLabel,
    QPushButton,
    QButtonGroup,
    QWidget,
    QHBoxLayout,
    QSizePolicy,
    QSpinBox,
)

from dg_patch_tool.core.blend_engine import BlendEngine
from dg_patch_tool.ui.selection_tools import SelectionTool


class PatchToolBar(QToolBar):
    """
    Main toolbar providing selection tools, blending algorithm selector,
    real-time opacity and feather sliders, undo/redo, and the 'Apply to Siril' button.
    """

    DONATE_URL = "https://www.paypal.com/donate/?hosted_button_id=48L9ULQ5PTS9A"

    tool_changed = pyqtSignal(str)
    algorithm_changed = pyqtSignal(str)
    opacity_changed = pyqtSignal(float)
    feather_changed = pyqtSignal(int)
    undo_requested = pyqtSignal()
    redo_requested = pyqtSignal()
    apply_requested = pyqtSignal()
    close_requested = pyqtSignal()
    fit_requested = pyqtSignal()
    zoom_100_requested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__("Patch Tools", parent)
        self.setMovable(False)
        self.setStyleSheet("""
            QToolBar {
                background: #1c1d22;
                border-bottom: 1px solid #2e313b;
                padding: 6px 12px;
                spacing: 10px;
            }
            QLabel {
                color: #d1d5db;
                font-size: 12px;
                font-weight: 500;
            }
            QPushButton {
                background: #2b2e38;
                color: #e5e7eb;
                border: 1px solid #3f4452;
                border-radius: 4px;
                padding: 5px 10px;
                font-size: 12px;
                font-weight: 500;
            }
            QPushButton:hover {
                background: #3a3e4c;
                border-color: #555b6e;
            }
            QPushButton:checked {
                background: #1e3a8a;
                border-color: #3b82f6;
                color: #ffffff;
            }
            QPushButton:disabled {
                background: #1e2026;
                color: #6b7280;
                border-color: #2b2e38;
            }
            QComboBox {
                background: #2b2e38;
                color: #e5e7eb;
                border: 1px solid #3f4452;
                border-radius: 4px;
                padding: 4px 8px;
                font-size: 12px;
                min-width: 180px;
            }
            QComboBox QAbstractItemView {
                background: #2b2e38;
                color: #e5e7eb;
                selection-background-color: #3b82f6;
                selection-color: #ffffff;
            }
            QSlider::groove:horizontal {
                height: 4px;
                background: #374151;
                border-radius: 2px;
            }
            QSlider::sub-page:horizontal {
                background: #3b82f6;
                border-radius: 2px;
            }
            QSlider::handle:horizontal {
                background: #60a5fa;
                border: 1px solid #93c5fd;
                width: 14px;
                margin-top: -5px;
                margin-bottom: -5px;
                border-radius: 7px;
            }
            QSlider::handle:horizontal:hover {
                background: #93c5fd;
            }
        """)

        self._init_ui()

    def _init_ui(self):
        # 1. Tool Selection (Lasso vs Rect)
        self.btn_lasso = QPushButton("Lasso (L)")
        self.btn_lasso.setCheckable(True)
        self.btn_lasso.setChecked(True)
        self.btn_lasso.setToolTip("Freehand Lasso selection (Press L)")

        self.btn_rect = QPushButton("Rect (R)")
        self.btn_rect.setCheckable(True)
        self.btn_rect.setToolTip("Rectangle selection (Press R)")

        self.tool_group = QButtonGroup(self)
        self.tool_group.addButton(self.btn_lasso)
        self.tool_group.addButton(self.btn_rect)
        self.tool_group.buttonClicked.connect(self._on_tool_clicked)

        self.addWidget(self.btn_lasso)
        self.addWidget(self.btn_rect)

        self.addSeparator()

        # 2. Blending Algorithm
        self.combo_algo = QComboBox()
        self.combo_algo.setToolTip("Select blending / healing algorithm")
        for mode in BlendEngine.ALL_MODES:
            self.combo_algo.addItem(mode)
        self.combo_algo.setCurrentText(BlendEngine.MODE_POISSON_NORMAL)
        self.combo_algo.currentTextChanged.connect(self._on_algo_changed)
        self.addWidget(self.combo_algo)

        self.addSeparator()

        # 3. Opacity Slider
        self.addWidget(QLabel("Opacity:"))
        self.slider_opacity = QSlider(Qt.Orientation.Horizontal)
        self.slider_opacity.setRange(0, 100)
        self.slider_opacity.setValue(100)
        self.slider_opacity.setFixedWidth(100)
        self.slider_opacity.setToolTip("Adjust blend opacity (0-100%) in real time")
        self.slider_opacity.valueChanged.connect(self._on_opacity_slider_changed)

        self.lbl_opacity = QLabel("100%")
        self.lbl_opacity.setFixedWidth(38)

        self.addWidget(self.slider_opacity)
        self.addWidget(self.lbl_opacity)

        self.addSeparator()

        # 4. Feather Slider
        self.addWidget(QLabel("Feather:"))
        self.slider_feather = QSlider(Qt.Orientation.Horizontal)
        self.slider_feather.setRange(0, 20)
        self.slider_feather.setValue(3)
        self.slider_feather.setFixedWidth(80)
        self.slider_feather.setToolTip("Feather selection edges (0-20px)")
        self.slider_feather.valueChanged.connect(self._on_feather_slider_changed)

        self.lbl_feather = QLabel("3 px")
        self.lbl_feather.setFixedWidth(32)

        self.addWidget(self.slider_feather)
        self.addWidget(self.lbl_feather)

        self.addSeparator()

        # 5. Undo / Redo
        self.btn_undo = QPushButton("Undo")
        self.btn_undo.setToolTip("Undo last patch (Ctrl+Z)")
        self.btn_undo.setEnabled(False)
        self.btn_undo.clicked.connect(self.undo_requested.emit)

        self.btn_redo = QPushButton("Redo")
        self.btn_redo.setToolTip("Redo patch (Ctrl+Shift+Z / Ctrl+Y)")
        self.btn_redo.setEnabled(False)
        self.btn_redo.clicked.connect(self.redo_requested.emit)

        self.addWidget(self.btn_undo)
        self.addWidget(self.btn_redo)

        self.addSeparator()

        # 6. Zoom controls
        self.btn_fit = QPushButton("Fit")
        self.btn_fit.setToolTip("Fit image in viewport")
        self.btn_fit.clicked.connect(self.fit_requested.emit)

        self.btn_100 = QPushButton("1:1")
        self.btn_100.setToolTip("View at 100% pixel scale")
        self.btn_100.clicked.connect(self.zoom_100_requested.emit)

        self.addWidget(self.btn_fit)
        self.addWidget(self.btn_100)

        # Spacer to push Apply button to the right
        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        self.addWidget(spacer)

        # 7. Apply to Siril Button
        self.btn_apply = QPushButton("Apply to Siril")
        self.btn_apply.setToolTip("Writes composite result directly to Siril memory")
        self.btn_apply.setStyleSheet("""
            QPushButton {
                background: #059669;
                color: #ffffff;
                font-weight: bold;
                border: 1px solid #10b981;
                padding: 6px 14px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background: #10b981;
            }
            QPushButton:pressed {
                background: #047857;
            }
        """)
        self.btn_apply.clicked.connect(self.apply_requested.emit)
        self.addWidget(self.btn_apply)

        # Spacing separation
        btn_sep1 = QWidget()
        btn_sep1.setFixedWidth(8)
        self.addWidget(btn_sep1)

        # 8. Buy Me a Coffee Button
        self.btn_coffee = QPushButton("☕")
        self.btn_coffee.setToolTip("Buy me a coffee — Support this project")
        self.btn_coffee.setStyleSheet("""
            QPushButton {
                background: #92400e;
                color: #ffffff;
                font-size: 14px;
                border: 1px solid #b45309;
                padding: 4px 10px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background: #b45309;
                border-color: #d97706;
            }
            QPushButton:pressed {
                background: #78350f;
            }
        """)
        self.btn_coffee.clicked.connect(lambda: QDesktopServices.openUrl(QUrl(self.DONATE_URL)))
        self.addWidget(self.btn_coffee)

        # Spacing separation
        btn_sep2 = QWidget()
        btn_sep2.setFixedWidth(8)
        self.addWidget(btn_sep2)

        # 9. Close Window Button
        self.btn_close = QPushButton("Close")
        self.btn_close.setToolTip("Close the DG_Patch_Tool window")
        self.btn_close.setStyleSheet("""
            QPushButton {
                background: #374151;
                color: #f3f4f6;
                font-weight: 500;
                border: 1px solid #4b5563;
                padding: 6px 14px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background: #4b5563;
                color: #ffffff;
            }
            QPushButton:pressed {
                background: #1f2937;
            }
        """)
        self.btn_close.clicked.connect(self.close_requested.emit)
        self.addWidget(self.btn_close)

        # Shortcuts
        self._setup_shortcuts()

    def _setup_shortcuts(self):
        shortcut_lasso = QShortcut(QKeySequence("L"), self)
        shortcut_lasso.activated.connect(lambda: self.btn_lasso.click())

        shortcut_rect = QShortcut(QKeySequence("R"), self)
        shortcut_rect.activated.connect(lambda: self.btn_rect.click())

    def _on_tool_clicked(self, button):
        if button == self.btn_lasso:
            self.tool_changed.emit(SelectionTool.TOOL_LASSO)
        else:
            self.tool_changed.emit(SelectionTool.TOOL_RECTANGLE)

    def _on_algo_changed(self, algo_name: str):
        self.algorithm_changed.emit(algo_name)

    def _on_opacity_slider_changed(self, value: int):
        self.lbl_opacity.setText(f"{value}%")
        self.opacity_changed.emit(value / 100.0)

    def _on_feather_slider_changed(self, value: int):
        self.lbl_feather.setText(f"{value} px")
        self.feather_changed.emit(value)

    def update_history_state(self, can_undo: bool, can_redo: bool):
        self.btn_undo.setEnabled(can_undo)
        self.btn_redo.setEnabled(can_redo)

    def get_current_tool(self) -> str:
        if self.btn_rect.isChecked():
            return SelectionTool.TOOL_RECTANGLE
        return SelectionTool.TOOL_LASSO

    def set_current_tool(self, tool: str):
        if str(tool).lower() in (SelectionTool.TOOL_RECTANGLE, "rectangle", "rect"):
            self.btn_rect.setChecked(True)
            self.btn_lasso.setChecked(False)
            self.tool_changed.emit(SelectionTool.TOOL_RECTANGLE)
        else:
            self.btn_lasso.setChecked(True)
            self.btn_rect.setChecked(False)
            self.tool_changed.emit(SelectionTool.TOOL_LASSO)

    def get_current_algorithm(self) -> str:
        return self.combo_algo.currentText()

    def set_current_algorithm(self, algo_name: str):
        if algo_name in BlendEngine.ALL_MODES:
            self.combo_algo.setCurrentText(algo_name)

    def get_current_opacity(self) -> float:
        return self.slider_opacity.value() / 100.0

    def set_current_opacity(self, opacity: float):
        val = int(round(max(0.0, min(1.0, opacity)) * 100))
        self.slider_opacity.setValue(val)
        self.lbl_opacity.setText(f"{val}%")

    def get_current_feather(self) -> int:
        return self.slider_feather.value()

    def set_current_feather(self, feather: int):
        val = max(0, min(20, feather))
        self.slider_feather.setValue(val)
        self.lbl_feather.setText(f"{val} px")
