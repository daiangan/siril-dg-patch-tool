from PyQt6.QtCore import Qt, QRectF
from PyQt6.QtGui import (
    QPen,
    QColor,
    QBrush,
    QPixmap,
    QPainterPath,
    QPolygonF,
)
from PyQt6.QtWidgets import (
    QGraphicsScene,
    QGraphicsPixmapItem,
    QGraphicsPathItem,
    QGraphicsPolygonItem,
)


class CanvasScene(QGraphicsScene):
    """
    QGraphicsScene that hosts the astronomical image, drawing overlays,
    destination selection, source sampling indicator, and live patch preview.
    """

    def __init__(self, parent=None):
        super().__init__(parent)

        # Base image item
        self.base_item = QGraphicsPixmapItem()
        self.base_item.setTransformationMode(Qt.TransformationMode.SmoothTransformation)
        self.addItem(self.base_item)

        # Live patch preview item (rendered directly over the destination ROI)
        self.preview_item = QGraphicsPixmapItem()
        self.preview_item.setTransformationMode(Qt.TransformationMode.SmoothTransformation)
        self.preview_item.setZValue(10)
        self.preview_item.setVisible(False)
        self.addItem(self.preview_item)

        # Drawing outline (while user is drawing lasso or rectangle)
        self.drawing_item = QGraphicsPathItem()
        pen_draw = QPen(QColor(0, 210, 255, 230), 1.5, Qt.PenStyle.DashLine)
        pen_draw.setCosmetic(True)
        self.drawing_item.setPen(pen_draw)
        self.drawing_item.setBrush(QBrush(QColor(0, 210, 255, 35)))
        self.drawing_item.setZValue(20)
        self.drawing_item.setVisible(False)
        self.addItem(self.drawing_item)

        # Destination selection boundary
        self.dest_outline_item = QGraphicsPolygonItem()
        pen_dest = QPen(QColor(0, 230, 255, 240), 1.5, Qt.PenStyle.DashLine)
        pen_dest.setCosmetic(True)
        self.dest_outline_item.setPen(pen_dest)
        self.dest_outline_item.setBrush(QBrush(QColor(0, 230, 255, 25)))
        self.dest_outline_item.setZValue(15)
        self.dest_outline_item.setVisible(False)
        self.addItem(self.dest_outline_item)

        # Source indicator boundary (shows area being sampled)
        self.source_outline_item = QGraphicsPolygonItem()
        pen_source = QPen(QColor(255, 215, 0, 230), 1.5, Qt.PenStyle.DotLine)
        pen_source.setCosmetic(True)
        self.source_outline_item.setPen(pen_source)
        self.source_outline_item.setBrush(QBrush(QColor(255, 215, 0, 20)))
        self.source_outline_item.setZValue(14)
        self.source_outline_item.setVisible(False)
        self.addItem(self.source_outline_item)

    def set_base_pixmap(self, pixmap: QPixmap):
        self.base_item.setPixmap(pixmap)
        self.setSceneRect(QRectF(pixmap.rect()))

    def update_base_pixmap(self, pixmap: QPixmap):
        self.base_item.setPixmap(pixmap)

    def show_drawing_path(self, path: QPainterPath):
        self.drawing_item.setPath(path)
        self.drawing_item.setVisible(True)

    def hide_drawing_path(self):
        self.drawing_item.setVisible(False)
        self.drawing_item.setPath(QPainterPath())

    def set_destination_selection(self, polygon: QPolygonF):
        self.dest_outline_item.setPolygon(polygon)
        self.dest_outline_item.setVisible(True)

    def hide_destination_selection(self):
        self.dest_outline_item.setVisible(False)

    def set_source_indicator(self, polygon: QPolygonF):
        self.source_outline_item.setPolygon(polygon)
        self.source_outline_item.setVisible(True)

    def hide_source_indicator(self):
        self.source_outline_item.setVisible(False)

    def set_preview_pixmap(self, pixmap: QPixmap, x: int, y: int):
        self.preview_item.setPixmap(pixmap)
        self.preview_item.setPos(x, y)
        self.preview_item.setVisible(True)

    def hide_preview(self):
        self.preview_item.setVisible(False)

    def clear_selection_overlays(self):
        self.hide_drawing_path()
        self.hide_destination_selection()
        self.hide_source_indicator()
        self.hide_preview()
