from PyQt6.QtCore import Qt, QPointF, QTimer, pyqtSignal
from PyQt6.QtGui import (
    QPainter,
    QPainterPath,
    QPolygonF,
    QMouseEvent,
    QWheelEvent,
    QKeyEvent,
    QCursor,
    QBrush,
    QColor,
)
from PyQt6.QtWidgets import QGraphicsView

from dg_patch_tool.ui.canvas_scene import CanvasScene
from dg_patch_tool.ui.selection_tools import SelectionTool, SelectionData


class CanvasView(QGraphicsView):
    """
    High-performance QGraphicsView for large astrophotography images.
    Provides smooth pan & zoom, Lasso / Rectangle selection drawing,
    and live patch source preview on cursor movement.
    """

    cursor_position_changed = pyqtSignal(int, int)
    zoom_changed = pyqtSignal(float)
    selection_made = pyqtSignal(object)       # SelectionData
    patch_confirmed = pyqtSignal(object, tuple)  # (SelectionData, (dx, dy))
    selection_cancelled = pyqtSignal()

    STATE_READY = 0       # Waiting for selection
    STATE_DRAWING = 1     # Drawing lasso or rectangle
    STATE_PATCHING = 2    # Destination selected, moving cursor for source preview

    def __init__(self, scene: CanvasScene, parent=None):
        super().__init__(scene, parent)
        self.custom_scene = scene

        # View settings for smooth rendering
        self.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
        self.setDragMode(QGraphicsView.DragMode.NoDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setStyleSheet("QGraphicsView { border: none; background-color: #141518; }")
        self.setBackgroundBrush(QBrush(QColor("#141518")))

        # State
        self.state = self.STATE_READY
        self.active_tool = SelectionTool.TOOL_LASSO
        self.current_zoom = 1.0
        self._has_been_fitted = False

        # Panning state
        self._is_panning = False
        self._pan_start = QPointF()
        self._space_pressed = False

        # Drawing state
        self._draw_start_scene = QPointF()
        self._draw_points = []

        # Current destination selection
        self.current_selection: SelectionData | None = None

        # Source cursor tracking
        self.current_source_offset = (0, 0)
        self._preview_timer = QTimer(self)
        self._preview_timer.setSingleShot(True)
        self._preview_timer.setInterval(16)  # ~60fps throttle for live blend preview
        self._preview_timer.timeout.connect(self._trigger_preview_update)
        self._pending_source_center = None

        # Callbacks
        self.preview_calculator = None  # Function: (selection, dx, dy) -> QPixmap

        self.setMouseTracking(True)
        self._update_cursor()

    def set_active_tool(self, tool_name: str):
        if str(tool_name).lower() in (SelectionTool.TOOL_RECTANGLE, "rectangle", "rect"):
            self.active_tool = SelectionTool.TOOL_RECTANGLE
        else:
            self.active_tool = SelectionTool.TOOL_LASSO
        self._update_cursor()

    def _update_cursor(self):
        if self._is_panning or self._space_pressed:
            self.setCursor(Qt.CursorShape.ClosedHandCursor if self._is_panning else Qt.CursorShape.OpenHandCursor)
        elif self.state == self.STATE_PATCHING:
            self.setCursor(Qt.CursorShape.PointingHandCursor)
        else:
            self.setCursor(Qt.CursorShape.CrossCursor)

    # --- Zoom and Pan ---

    def wheelEvent(self, event: QWheelEvent):
        """Smooth zoom anchored under cursor."""
        zoom_in_factor = 1.2
        zoom_out_factor = 1.0 / zoom_in_factor

        if event.angleDelta().y() > 0:
            factor = zoom_in_factor
        else:
            factor = zoom_out_factor

        new_zoom = self.current_zoom * factor
        if 0.05 <= new_zoom <= 32.0:
            self.scale(factor, factor)
            self.current_zoom = new_zoom
            self.zoom_changed.emit(self.current_zoom)

    def keyPressEvent(self, event: QKeyEvent):
        if event.key() == Qt.Key.Key_Space and not event.isAutoRepeat():
            self._space_pressed = True
            self._update_cursor()
        elif event.key() == Qt.Key.Key_Escape:
            self.cancel_selection()
        else:
            super().keyPressEvent(event)

    def keyReleaseEvent(self, event: QKeyEvent):
        if event.key() == Qt.Key.Key_Space and not event.isAutoRepeat():
            self._space_pressed = False
            self._update_cursor()
        else:
            super().keyReleaseEvent(event)

    # --- Mouse Events ---

    def mousePressEvent(self, event: QMouseEvent):
        # Pan with Middle Mouse OR Space + Left Mouse
        if event.button() == Qt.MouseButton.MiddleButton or (
            event.button() == Qt.MouseButton.LeftButton and self._space_pressed
        ):
            self._is_panning = True
            self._pan_start = event.position()
            self._update_cursor()
            event.accept()
            return

        if event.button() == Qt.MouseButton.RightButton:
            # Right click cancels selection/patching
            self.cancel_selection()
            event.accept()
            return

        if event.button() == Qt.MouseButton.LeftButton:
            scene_pos = self.mapToScene(event.pos())
            img_rect = self.custom_scene.sceneRect()

            if self.state == self.STATE_READY:
                # Start drawing selection
                if img_rect.contains(scene_pos):
                    self.state = self.STATE_DRAWING
                    self._draw_start_scene = scene_pos
                    self._draw_points = [scene_pos]
                    event.accept()
                    return

            elif self.state == self.STATE_PATCHING:
                # Left click in patching state confirms the patch!
                if self.current_selection and self.current_source_offset != (0, 0):
                    self.patch_confirmed.emit(self.current_selection, self.current_source_offset)
                event.accept()
                return

        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent):
        scene_pos = self.mapToScene(event.pos())
        self.cursor_position_changed.emit(int(scene_pos.x()), int(scene_pos.y()))

        # Handle Pan
        if self._is_panning:
            delta = event.position() - self._pan_start
            self._pan_start = event.position()
            self.horizontalScrollBar().setValue(int(self.horizontalScrollBar().value() - delta.x()))
            self.verticalScrollBar().setValue(int(self.verticalScrollBar().value() - delta.y()))
            event.accept()
            return

        # Handle Selection Drawing
        if self.state == self.STATE_DRAWING:
            if self.active_tool == SelectionTool.TOOL_LASSO:
                self._draw_points.append(scene_pos)
                path = QPainterPath()
                if self._draw_points:
                    path.moveTo(self._draw_points[0])
                    for pt in self._draw_points[1:]:
                        path.lineTo(pt)
                self.custom_scene.show_drawing_path(path)
            elif self.active_tool in (SelectionTool.TOOL_RECTANGLE, "rectangle", "rect"):
                poly = SelectionTool.rect_to_polygon(self._draw_start_scene, scene_pos)
                path = QPainterPath()
                path.addPolygon(poly)
                self.custom_scene.show_drawing_path(path)
            event.accept()
            return

        # Handle Live Patch Source Preview
        if self.state == self.STATE_PATCHING and self.current_selection:
            # The cursor defines the source sample center
            src_cx = int(scene_pos.x())
            src_cy = int(scene_pos.y())
            dest_cx, dest_cy = self.current_selection.center
            dx = src_cx - dest_cx
            dy = src_cy - dest_cy

            self.current_source_offset = (dx, dy)

            # Update yellow source indicator outline
            source_poly = self.current_selection.polygon.translated(dx, dy)
            self.custom_scene.set_source_indicator(source_poly)

            # Request debounced/throttled ROI preview update
            self._pending_source_center = (dx, dy)
            if not self._preview_timer.isActive():
                self._preview_timer.start()

            event.accept()
            return

        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent):
        if event.button() in (Qt.MouseButton.MiddleButton, Qt.MouseButton.LeftButton) and self._is_panning:
            self._is_panning = False
            self._update_cursor()
            event.accept()
            return

        if event.button() == Qt.MouseButton.LeftButton and self.state == self.STATE_DRAWING:
            self.state = self.STATE_READY
            self.custom_scene.hide_drawing_path()

            # Finalize selection polygon
            scene_rect = self.custom_scene.sceneRect()
            w_img = int(scene_rect.width())
            h_img = int(scene_rect.height())

            polygon = QPolygonF()
            if self.active_tool == SelectionTool.TOOL_LASSO:
                if len(self._draw_points) > 2:
                    polygon = QPolygonF(self._draw_points)
                    # Close the polygon
                    polygon.append(self._draw_points[0])
            elif self.active_tool in (SelectionTool.TOOL_RECTANGLE, "rectangle", "rect"):
                scene_pos = self.mapToScene(event.pos())
                polygon = SelectionTool.rect_to_polygon(self._draw_start_scene, scene_pos)

            self._draw_points = []

            selection_data = SelectionTool.polygon_to_mask(polygon, w_img, h_img)
            if selection_data:
                self.set_selection(selection_data)
                self.selection_made.emit(selection_data)

            event.accept()
            return

        super().mouseReleaseEvent(event)

    def set_selection(self, selection_data: SelectionData):
        """Sets active destination selection and transitions view to patching mode."""
        self.current_selection = selection_data
        self.custom_scene.set_destination_selection(selection_data.polygon)
        self.state = self.STATE_PATCHING
        self._update_cursor()

    def cancel_selection(self):
        """Cancels active selection and returns to ready state."""
        self.current_selection = None
        self.current_source_offset = (0, 0)
        self.custom_scene.clear_selection_overlays()
        self.state = self.STATE_READY
        self._update_cursor()
        self.selection_cancelled.emit()

    def _trigger_preview_update(self):
        """Computes and updates live preview pixmap on the destination ROI."""
        if not self.current_selection or not self.preview_calculator:
            return

        dx, dy = self.current_source_offset
        preview_pixmap = self.preview_calculator(self.current_selection, dx, dy)
        if preview_pixmap and not preview_pixmap.isNull():
            self.custom_scene.set_preview_pixmap(
                preview_pixmap,
                self.current_selection.x,
                self.current_selection.y,
            )

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if not self._has_been_fitted:
            rect = self.custom_scene.sceneRect()
            if not rect.isEmpty() and self.viewport().width() > 50 and self.viewport().height() > 50:
                self._has_been_fitted = True
                self.fit_image_in_view()

    def fit_image_in_view(self):
        """Fits the entire astronomical image inside the viewport."""
        rect = self.custom_scene.sceneRect()
        if not rect.isEmpty() and self.viewport().width() > 10 and self.viewport().height() > 10:
            self.resetTransform()
            self.fitInView(rect, Qt.AspectRatioMode.KeepAspectRatio)
            # Calculate current zoom scale from transform
            self.current_zoom = self.transform().m11()
            self.zoom_changed.emit(self.current_zoom)

    def reset_zoom_to_100(self):
        """Resets canvas zoom to 100% (1:1 pixel scale)."""
        self.resetTransform()
        self.current_zoom = 1.0
        self.zoom_changed.emit(self.current_zoom)
