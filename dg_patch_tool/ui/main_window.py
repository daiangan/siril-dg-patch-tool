import cv2
import numpy as np
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QPixmap, QKeySequence, QShortcut
from PyQt6.QtWidgets import (
    QMainWindow,
    QStatusBar,
    QLabel,
    QMessageBox,
    QFileDialog,
)

from dg_patch_tool.core.image_buffer import ImageBuffer
from dg_patch_tool.core.history import HistoryManager
from dg_patch_tool.core.blend_engine import BlendEngine
from dg_patch_tool.core.safety_check import check_linear_data
from dg_patch_tool.siril_bridge import SirilBridge
from dg_patch_tool.ui.canvas_scene import CanvasScene
from dg_patch_tool.ui.canvas_view import CanvasView
from dg_patch_tool.ui.toolbar import PatchToolBar
from dg_patch_tool.ui.selection_tools import SelectionData
from dg_patch_tool.core.settings import load_user_preferences, save_user_preferences


class MainWindow(QMainWindow):
    """
    Main application window for DG Patch Tool.
    Hosts canvas, image buffer, blend engine, and Siril bridge.
    """

    def __init__(self, raw_image: np.ndarray, siril_bridge: SirilBridge, parent=None):
        super().__init__(parent)
        self.raw_image = raw_image
        self.siril_bridge = siril_bridge

        self.setWindowTitle("DG_Patch_Tool — Starless Cleanup for Siril")
        self.resize(1300, 850)
        self.setStyleSheet("""
            QMainWindow {
                background: #141518;
            }
            QStatusBar {
                background: #1c1d22;
                color: #9ca3af;
                font-size: 11px;
                border-top: 1px solid #2e313b;
            }
            QMessageBox {
                background: #1c1d22;
                color: #e5e7eb;
            }
            QMessageBox QPushButton {
                background: #2b2e38;
                color: #ffffff;
                padding: 6px 14px;
                border-radius: 4px;
                border: 1px solid #3f4452;
            }
        """)

        # Core logic components
        self.image_buffer = ImageBuffer(raw_image)
        self.history = HistoryManager(max_depth=50)

        # Cache for live preview (to avoid re-running blend when only opacity changes)
        self._cached_preview_data = None  # (dest_roi, patched_roi, feathered_mask, roi_rect)
        self._has_unapplied_changes = False

        # Build UI
        self._init_ui()

        # Connect callbacks
        self.canvas_view.preview_calculator = self._calculate_preview_roi

        # Initial display
        self._refresh_canvas()
        QTimer.singleShot(100, self.canvas_view.fit_image_in_view)

        # Check linear data safety guard
        self._check_linear_safety()

    def _init_ui(self):
        # Scene & View
        self.canvas_scene = CanvasScene(self)
        self.canvas_view = CanvasView(self.canvas_scene, self)
        self.setCentralWidget(self.canvas_view)

        # Toolbar
        self.toolbar = PatchToolBar(self)
        self.addToolBar(Qt.ToolBarArea.TopToolBarArea, self.toolbar)

        # Status Bar
        self.statusbar = QStatusBar(self)
        self.setStatusBar(self.statusbar)

        self.lbl_dims = QLabel()
        self.lbl_cursor = QLabel("X: -  Y: -")
        self.lbl_zoom = QLabel("Zoom: 100%")
        self.lbl_msg = QLabel("Draw selection around artifact (Lasso / Rect), then hover source to live-preview.")
        self.lbl_msg.setStyleSheet("color: #60a5fa; font-weight: 500;")

        self.statusbar.addWidget(self.lbl_msg, 1)
        self.statusbar.addPermanentWidget(self.lbl_dims)
        self.statusbar.addPermanentWidget(self.lbl_cursor)
        self.statusbar.addPermanentWidget(self.lbl_zoom)

        h, w = self.image_buffer.height, self.image_buffer.width
        channels_str = "RGB" if self.image_buffer.is_color else "Mono"
        dtype_str = str(self.image_buffer.original_dtype)
        self.lbl_dims.setText(f"{w} × {h} px | {channels_str} ({dtype_str})")

        # Wire Toolbar Signals
        self.toolbar.tool_changed.connect(self._on_tool_changed)
        self.toolbar.algorithm_changed.connect(self._on_algorithm_changed)
        self.toolbar.opacity_changed.connect(self._on_opacity_changed)
        self.toolbar.feather_changed.connect(self._on_feather_changed)
        self.toolbar.undo_requested.connect(self._on_undo)
        self.toolbar.redo_requested.connect(self._on_redo)
        self.toolbar.fit_requested.connect(self.canvas_view.fit_image_in_view)
        self.toolbar.zoom_100_requested.connect(self.canvas_view.reset_zoom_to_100)
        self.toolbar.apply_requested.connect(self._on_apply_to_siril)
        self.toolbar.close_requested.connect(self.close)

        # Restore user preferences from last session
        self._load_preferences()

        # Wire Canvas View Signals
        self.canvas_view.cursor_position_changed.connect(self._on_cursor_moved)
        self.canvas_view.zoom_changed.connect(self._on_zoom_changed)
        self.canvas_view.selection_made.connect(self._on_selection_made)
        self.canvas_view.patch_confirmed.connect(self._on_patch_confirmed)
        self.canvas_view.selection_cancelled.connect(self._on_selection_cancelled)

        # Keyboard shortcuts
        shortcut_undo = QShortcut(QKeySequence("Ctrl+Z"), self)
        shortcut_undo.activated.connect(self._on_undo)

        shortcut_redo = QShortcut(QKeySequence("Ctrl+Shift+Z"), self)
        shortcut_redo.activated.connect(self._on_redo)
        shortcut_redo_alt = QShortcut(QKeySequence("Ctrl+Y"), self)
        shortcut_redo_alt.activated.connect(self._on_redo)

    def _check_linear_safety(self):
        """Pre-flight check for unstretched linear data."""
        is_linear, msg, stats = check_linear_data(self.raw_image)
        if is_linear:
            reply = QMessageBox.warning(
                self,
                "Linear Data Warning",
                msg,
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
                QMessageBox.StandardButton.Yes,
            )
            if reply == QMessageBox.StandardButton.Cancel:
                self.close()

    def _refresh_canvas(self):
        pixmap = self.image_buffer.get_qpixmap()
        self.canvas_scene.set_base_pixmap(pixmap)

    MAX_PREVIEW_DIM = 200

    # --- Live Preview Calculation (ROI bounded with Adaptive Downsampling) ---

    def _calculate_preview_roi(self, selection: SelectionData, dx: int, dy: int) -> QPixmap | None:
        """
        Computes live patch preview restricted strictly to the ROI bounding box + margin.
        Uses adaptive downsampling for large ROIs (>256px) so preview renders in <20ms
        at smooth 60 FPS on multi-megapixel images.
        """
        if dx == 0 and dy == 0:
            return None

        # ROI Bounding Box with margin for boundary integration
        feather = self.toolbar.get_current_feather()
        algo = self.toolbar.get_current_algorithm()
        margin = max(16, feather * 2)

        x0 = max(0, selection.x - margin)
        y0 = max(0, selection.y - margin)
        x1 = min(self.image_buffer.width, selection.x + selection.w + margin)
        y1 = min(self.image_buffer.height, selection.y + selection.h + margin)

        roi_w = x1 - x0
        roi_h = y1 - y0
        if roi_w <= 4 or roi_h <= 4:
            return None

        # Source coordinates shifted by offset
        src_x0 = x0 + dx
        src_y0 = y0 + dy
        src_x1 = src_x0 + roi_w
        src_y1 = src_y0 + roi_h

        # Check bounds for source
        if src_x0 < 0 or src_y0 < 0 or src_x1 > self.image_buffer.width or src_y1 > self.image_buffer.height:
            return None

        # Extract destination and source ROIs
        dst_roi = self.image_buffer.get_roi(x0, y0, roi_w, roi_h)
        src_roi = self.image_buffer.get_roi(src_x0, src_y0, roi_w, roi_h)

        if dst_roi.shape[:2] != (roi_h, roi_w) or src_roi.shape[:2] != (roi_h, roi_w):
            return None

        # Create padded mask corresponding to the padded ROI
        roi_mask = np.zeros((roi_h, roi_w), dtype=np.uint8)
        mask_x_offset = selection.x - x0
        mask_y_offset = selection.y - y0
        roi_mask[mask_y_offset : mask_y_offset + selection.h, mask_x_offset : mask_x_offset + selection.w] = selection.mask

        max_dim = max(roi_w, roi_h)
        is_downsampled = max_dim > self.MAX_PREVIEW_DIM

        if is_downsampled:
            scale = self.MAX_PREVIEW_DIM / float(max_dim)
            new_w = max(4, int(round(roi_w * scale)))
            new_h = max(4, int(round(roi_h * scale)))

            dst_small = cv2.resize(dst_roi, (new_w, new_h), interpolation=cv2.INTER_AREA)
            src_small = cv2.resize(src_roi, (new_w, new_h), interpolation=cv2.INTER_AREA)
            mask_small = cv2.resize(roi_mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
            feather_small = max(1, int(round(feather * scale))) if feather > 0 else 0

            patched_small = BlendEngine.blend(
                dst_roi=dst_small,
                src_roi=src_small,
                mask=mask_small,
                mode=algo,
                feather_radius=feather_small,
            )
            feathered_mask_small = BlendEngine.feather_mask(mask_small, feather_small)

            sub_x = int(round(mask_x_offset * scale))
            sub_y = int(round(mask_y_offset * scale))
            sub_w = max(1, min(int(round(selection.w * scale)), new_w - sub_x))
            sub_h = max(1, min(int(round(selection.h * scale)), new_h - sub_y))

            # Cache for fast opacity slider updates
            self._cached_preview_data = {
                "is_downsampled": True,
                "x0": x0,
                "y0": y0,
                "w": roi_w,
                "h": roi_h,
                "dst_roi": dst_small,
                "patched_roi": patched_small,
                "feathered_mask": feathered_mask_small,
                "sub_x": sub_x,
                "sub_y": sub_y,
                "sub_w": sub_w,
                "sub_h": sub_h,
                "orig_sel_w": selection.w,
                "orig_sel_h": selection.h,
                "dx": dx,
                "dy": dy,
            }

            opacity = self.toolbar.get_current_opacity()
            blended_small = BlendEngine.apply_opacity(dst_small, patched_small, feathered_mask_small, opacity)
            sub_blended = blended_small[sub_y : sub_y + sub_h, sub_x : sub_x + sub_w]

            sub_uint8 = self.image_buffer._convert_to_display_uint8(sub_blended)
            qimg = self.image_buffer.create_qimage_from_array(sub_uint8)
            pixmap = QPixmap.fromImage(qimg)
            # Scale back to selection dimensions with smooth bilinear filtering
            if pixmap.width() != selection.w or pixmap.height() != selection.h:
                pixmap = pixmap.scaled(
                    selection.w,
                    selection.h,
                    Qt.AspectRatioMode.IgnoreAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
            return pixmap

        else:
            # Full native resolution (already small ROI, fast enough)
            patched_roi = BlendEngine.blend(
                dst_roi=dst_roi,
                src_roi=src_roi,
                mask=roi_mask,
                mode=algo,
                feather_radius=feather,
            )
            feathered_mask = BlendEngine.feather_mask(roi_mask, feather)

            self._cached_preview_data = {
                "is_downsampled": False,
                "x0": x0,
                "y0": y0,
                "w": roi_w,
                "h": roi_h,
                "dst_roi": dst_roi,
                "patched_roi": patched_roi,
                "feathered_mask": feathered_mask,
                "sub_x": mask_x_offset,
                "sub_y": mask_y_offset,
                "sub_w": selection.w,
                "sub_h": selection.h,
                "orig_sel_w": selection.w,
                "orig_sel_h": selection.h,
                "dx": dx,
                "dy": dy,
            }

            opacity = self.toolbar.get_current_opacity()
            blended_roi = BlendEngine.apply_opacity(dst_roi, patched_roi, feathered_mask, opacity)
            sub_blended = blended_roi[
                mask_y_offset : mask_y_offset + selection.h,
                mask_x_offset : mask_x_offset + selection.w,
            ]

            sub_uint8 = self.image_buffer._convert_to_display_uint8(sub_blended)
            qimg = self.image_buffer.create_qimage_from_array(sub_uint8)
            return QPixmap.fromImage(qimg)

    def _update_cached_preview_display(self):
        """Re-evaluates opacity on cached preview without re-computing blending."""
        if not self._cached_preview_data:
            return

        cache = self._cached_preview_data
        opacity = self.toolbar.get_current_opacity()

        blended_roi = BlendEngine.apply_opacity(
            cache["dst_roi"],
            cache["patched_roi"],
            cache["feathered_mask"],
            opacity,
        )

        sub_x = cache.get("sub_x", cache.get("selection_rel_x", 0))
        sub_y = cache.get("sub_y", cache.get("selection_rel_y", 0))
        sub_w = cache.get("sub_w", cache.get("sel_w", 0))
        sub_h = cache.get("sub_h", cache.get("sel_h", 0))

        sub_blended = blended_roi[sub_y : sub_y + sub_h, sub_x : sub_x + sub_w]

        sub_uint8 = self.image_buffer._convert_to_display_uint8(sub_blended)
        qimg = self.image_buffer.create_qimage_from_array(sub_uint8)
        pixmap = QPixmap.fromImage(qimg)

        target_w = cache.get("orig_sel_w", self.canvas_view.current_selection.w if self.canvas_view.current_selection else pixmap.width())
        target_h = cache.get("orig_sel_h", self.canvas_view.current_selection.h if self.canvas_view.current_selection else pixmap.height())

        if pixmap.width() != target_w or pixmap.height() != target_h:
            pixmap = pixmap.scaled(
                target_w,
                target_h,
                Qt.AspectRatioMode.IgnoreAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )

        self.canvas_scene.set_preview_pixmap(
            pixmap,
            self.canvas_view.current_selection.x,
            self.canvas_view.current_selection.y,
        )

    # --- Interaction Callbacks ---

    def _on_selection_made(self, selection: SelectionData):
        algo = self.toolbar.get_current_algorithm()
        if algo == BlendEngine.MODE_SHIFTMAP:
            # Auto-Heal: immediately compute and display inpainting preview
            self._execute_auto_heal(selection)
        else:
            self.lbl_msg.setText("Destination selected. Move cursor over clean source area to preview, then Click to apply.")

    def _on_patch_confirmed(self, selection: SelectionData, source_offset: tuple[int, int]):
        """Commits the patch into the image buffer and history stack at full native resolution."""
        if not self._cached_preview_data:
            return

        cache = self._cached_preview_data
        opacity = self.toolbar.get_current_opacity()
        algo = self.toolbar.get_current_algorithm()
        feather = self.toolbar.get_current_feather()

        if cache.get("is_downsampled", False):
            # Compute full native resolution patch
            dx, dy = source_offset
            margin = max(16, feather * 2)
            x0 = max(0, selection.x - margin)
            y0 = max(0, selection.y - margin)
            x1 = min(self.image_buffer.width, selection.x + selection.w + margin)
            y1 = min(self.image_buffer.height, selection.y + selection.h + margin)
            roi_w = x1 - x0
            roi_h = y1 - y0

            src_x0 = x0 + dx
            src_y0 = y0 + dy
            src_x1 = src_x0 + roi_w
            src_y1 = src_y0 + roi_h

            if (src_x0 < 0 or src_y0 < 0 or
                src_x1 > self.image_buffer.width or src_y1 > self.image_buffer.height or
                roi_w <= 4 or roi_h <= 4):
                return

            dst_roi = self.image_buffer.get_roi(x0, y0, roi_w, roi_h)
            src_roi = self.image_buffer.get_roi(src_x0, src_y0, roi_w, roi_h)

            roi_mask = np.zeros((roi_h, roi_w), dtype=np.uint8)
            mask_x_offset = selection.x - x0
            mask_y_offset = selection.y - y0
            roi_mask[mask_y_offset : mask_y_offset + selection.h, mask_x_offset : mask_x_offset + selection.w] = selection.mask

            patched_roi = BlendEngine.blend(
                dst_roi=dst_roi,
                src_roi=src_roi,
                mask=roi_mask,
                mode=algo,
                feather_radius=feather,
            )
            feathered_mask = BlendEngine.feather_mask(roi_mask, feather)

            before_roi = dst_roi
            final_roi = BlendEngine.apply_opacity(
                dst_roi,
                patched_roi,
                feathered_mask,
                opacity,
            )
        else:
            # Re-use already computed full-resolution ROI
            x0, y0, roi_w, roi_h = cache["x0"], cache["y0"], cache["w"], cache["h"]
            before_roi = cache["dst_roi"]
            feathered_mask = cache["feathered_mask"]
            final_roi = BlendEngine.apply_opacity(
                before_roi,
                cache["patched_roi"],
                feathered_mask,
                opacity,
            )

        # Record undo
        self.history.record_patch(
            x=x0,
            y=y0,
            w=roi_w,
            h=roi_h,
            before_data=before_roi,
            after_data=final_roi,
            mask=feathered_mask,
            algorithm=algo,
            opacity=opacity,
        )

        # Update image buffer
        self.image_buffer.set_roi(x0, y0, roi_w, roi_h, final_roi)

        # Update base pixmap in scene
        self.canvas_scene.update_base_pixmap(self.image_buffer.get_qpixmap())

        # Reset selection state
        self._cached_preview_data = None
        self._has_unapplied_changes = True
        self.canvas_view.cancel_selection()
        self.toolbar.update_history_state(self.history.can_undo(), self.history.can_redo())
        self.lbl_msg.setText("Patch applied. Ready for next selection.")

    def _execute_auto_heal(self, selection: SelectionData):
        """Runs ShiftMap inpainting on the active selection."""
        margin = max(24, self.toolbar.get_current_feather() * 3)
        x0 = max(0, selection.x - margin)
        y0 = max(0, selection.y - margin)
        x1 = min(self.image_buffer.width, selection.x + selection.w + margin)
        y1 = min(self.image_buffer.height, selection.y + selection.h + margin)

        roi_w = x1 - x0
        roi_h = y1 - y0

        dst_roi = self.image_buffer.get_roi(x0, y0, roi_w, roi_h)

        roi_mask = np.zeros((roi_h, roi_w), dtype=np.uint8)
        mask_x_offset = selection.x - x0
        mask_y_offset = selection.y - y0
        roi_mask[mask_y_offset : mask_y_offset + selection.h, mask_x_offset : mask_x_offset + selection.w] = selection.mask

        self.lbl_msg.setText("Calculating Auto-Heal (ShiftMap)...")
        self.statusbar.repaint()

        patched_roi = BlendEngine.inpaint_shiftmap(dst_roi, roi_mask)
        feather = self.toolbar.get_current_feather()
        feathered_mask = BlendEngine.feather_mask(roi_mask, feather)

        self._cached_preview_data = {
            "is_downsampled": False,
            "x0": x0,
            "y0": y0,
            "w": roi_w,
            "h": roi_h,
            "dst_roi": dst_roi,
            "patched_roi": patched_roi,
            "feathered_mask": feathered_mask,
            "sub_x": mask_x_offset,
            "sub_y": mask_y_offset,
            "sub_w": selection.w,
            "sub_h": selection.h,
            "orig_sel_w": selection.w,
            "orig_sel_h": selection.h,
            "dx": 1,
            "dy": 1,
        }

        # Set source offset dummy so click confirms
        self.canvas_view.current_source_offset = (1, 1)
        self._update_cached_preview_display()
        self.lbl_msg.setText("Auto-Heal computed. Adjust opacity slider or click canvas to confirm patch.")

    def _on_selection_cancelled(self):
        self._cached_preview_data = None
        self.lbl_msg.setText("Selection cleared. Draw around artifact to start.")

    def _on_tool_changed(self, tool_name: str):
        self.canvas_view.set_active_tool(tool_name)
        self._save_preferences()

    def _on_algorithm_changed(self, algo_name: str):
        self._save_preferences()
        if self.canvas_view.state == CanvasView.STATE_PATCHING:
            if algo_name == BlendEngine.MODE_SHIFTMAP and self.canvas_view.current_selection:
                self._execute_auto_heal(self.canvas_view.current_selection)
            else:
                self.canvas_view._trigger_preview_update()

    def _on_opacity_changed(self, opacity: float):
        self._save_preferences()
        if self._cached_preview_data:
            self._update_cached_preview_display()

    def _on_feather_changed(self, feather: int):
        self._save_preferences()
        if self.canvas_view.state == CanvasView.STATE_PATCHING:
            self.canvas_view._trigger_preview_update()

    def _load_preferences(self):
        """Loads and applies persisted settings from previous session."""
        try:
            prefs = load_user_preferences()
            self.toolbar.set_current_tool(prefs["tool"])
            self.toolbar.set_current_algorithm(prefs["algorithm"])
            self.toolbar.set_current_opacity(prefs["opacity"])
            self.toolbar.set_current_feather(prefs["feather"])
            self.canvas_view.set_active_tool(prefs["tool"])
        except Exception:
            pass

    def _save_preferences(self):
        """Persists current toolbar settings to QSettings."""
        try:
            save_user_preferences(
                tool=self.toolbar.get_current_tool(),
                algorithm=self.toolbar.get_current_algorithm(),
                opacity=self.toolbar.get_current_opacity(),
                feather=self.toolbar.get_current_feather(),
            )
        except Exception:
            pass

    def _on_undo(self):
        action = self.history.undo(self.image_buffer)
        if action:
            self.canvas_scene.update_base_pixmap(self.image_buffer.get_qpixmap())
            self.toolbar.update_history_state(self.history.can_undo(), self.history.can_redo())
            if not self.history.can_undo():
                self._has_unapplied_changes = False
            self.lbl_msg.setText("Undid last patch.")

    def _on_redo(self):
        action = self.history.redo(self.image_buffer)
        if action:
            self._has_unapplied_changes = True
            self.canvas_scene.update_base_pixmap(self.image_buffer.get_qpixmap())
            self.toolbar.update_history_state(self.history.can_undo(), self.history.can_redo())
            self.lbl_msg.setText("Redid patch.")

    def _on_cursor_moved(self, x: int, y: int):
        self.lbl_cursor.setText(f"X: {x}  Y: {y}")

    def _on_zoom_changed(self, zoom: float):
        self.lbl_zoom.setText(f"Zoom: {int(zoom * 100)}%")

    def _on_apply_to_siril(self, show_confirmation: bool = True) -> bool:
        """Sends composite result to Siril via set_image_pixeldata()."""
        export_data = self.image_buffer.export_for_siril()
        if self.siril_bridge and self.siril_bridge.is_connected():
            success = self.siril_bridge.set_image_pixeldata(export_data)
            if success:
                self._has_unapplied_changes = False
                self.lbl_msg.setText("Successfully applied composite to Siril!")
                if show_confirmation:
                    QMessageBox.information(
                        self,
                        "Applied to Siril",
                        "The patched image has been successfully written back to Siril in memory.\n\n"
                        "The tool will remain open so you can continue cleaning if needed.",
                    )
                return True
            else:
                QMessageBox.critical(self, "Error", "Failed to write image data back to Siril.")
                return False
        else:
            # Standalone mode: offer to save file
            file_path, _ = QFileDialog.getSaveFileName(
                self,
                "Save Patched Image",
                "patched_starless.tiff",
                "TIFF Image (*.tif *.tiff);;PNG Image (*.png);;FITS (*.fit *.fits)",
            )
            if file_path:
                try:
                    import cv2
                    if export_data.dtype == np.float32 or export_data.dtype == np.float64:
                        save_arr = (np.clip(export_data, 0.0, 1.0) * 65535.0).astype(np.uint16)
                    else:
                        save_arr = export_data

                    if save_arr.ndim == 3 and save_arr.shape[2] == 3:
                        save_arr = cv2.cvtColor(save_arr, cv2.COLOR_RGB2BGR)

                    cv2.imwrite(file_path, save_arr)
                    self._has_unapplied_changes = False
                    self.lbl_msg.setText(f"Saved to {file_path}")
                    if show_confirmation:
                        QMessageBox.information(self, "Saved", f"Image successfully saved to:\n{file_path}")
                    return True
                except Exception as e:
                    QMessageBox.critical(self, "Save Error", f"Error saving image: {e}")
                    return False
            return False

    def showEvent(self, event):
        super().showEvent(event)
        # Ensure image is automatically fitted to screen when window appears
        QTimer.singleShot(50, self.canvas_view.fit_image_in_view)

    def closeEvent(self, event):
        """Safeguard: warn user if closing with unapplied patch changes."""
        self._save_preferences()
        if self._has_unapplied_changes:
            reply = QMessageBox.question(
                self,
                "Unapplied Changes",
                "You have patches that have not been applied to Siril yet.\n\n"
                "Do you want to apply them before closing?",
                QMessageBox.StandardButton.Save | QMessageBox.StandardButton.Discard | QMessageBox.StandardButton.Cancel,
                QMessageBox.StandardButton.Save,
            )
            if reply == QMessageBox.StandardButton.Save:
                success = self._on_apply_to_siril(show_confirmation=False)
                if success:
                    event.accept()
                else:
                    event.ignore()
            elif reply == QMessageBox.StandardButton.Discard:
                event.accept()
            else:
                event.ignore()
        else:
            event.accept()
