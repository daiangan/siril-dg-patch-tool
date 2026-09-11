from dataclasses import dataclass
import numpy as np
from PyQt6.QtCore import QPointF, QRectF
from PyQt6.QtGui import QPainter, QPainterPath, QPolygonF, QImage


@dataclass
class SelectionData:
    """Stores selection geometry and rasterized binary mask."""
    x: int
    y: int
    w: int
    h: int
    mask: np.ndarray          # 2D uint8 (h, w) binary mask [0 or 255]
    polygon: QPolygonF        # Selection boundary in scene coordinates

    @property
    def center(self) -> tuple[int, int]:
        return (self.x + self.w // 2, self.y + self.h // 2)


class SelectionTool:
    TOOL_LASSO = "lasso"
    TOOL_RECTANGLE = "rect"
    TOOL_RECT = "rect"

    @classmethod
    def polygon_to_mask(cls, polygon: QPolygonF, img_width: int, img_height: int) -> SelectionData | None:
        """
        Converts a QPolygonF in image coordinates to a cropped ROI binary mask and bounding rect.
        """
        if polygon.isEmpty() or polygon.count() < 3:
            return None

        boundingRect = polygon.boundingRect()
        x0 = max(0, int(np.floor(boundingRect.left())))
        y0 = max(0, int(np.floor(boundingRect.top())))
        x1 = min(img_width, int(np.ceil(boundingRect.right())))
        y1 = min(img_height, int(np.ceil(boundingRect.bottom())))

        w = x1 - x0
        h = y1 - y0

        if w <= 1 or h <= 1:
            return None

        # Render polygon into a 1-bit / 8-bit QImage
        mask_img = QImage(w, h, QImage.Format.Format_Grayscale8)
        mask_img.fill(0)

        painter = QPainter(mask_img)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)
        # Shift polygon to ROI local coordinates
        local_poly = polygon.translated(-x0, -y0)
        painter.setBrush(painter.pen().color().fromRgb(255, 255, 255))
        painter.setPen(painter.pen().color().fromRgb(255, 255, 255))
        painter.drawPolygon(local_poly)
        painter.end()

        # Convert QImage to numpy array
        ptr = mask_img.bits()
        ptr.setsize(h * mask_img.bytesPerLine())
        mask_np = np.frombuffer(ptr, dtype=np.uint8).reshape((h, mask_img.bytesPerLine()))[:, :w].copy()

        # Ensure binary 0 or 255
        mask_binary = (mask_np > 127).astype(np.uint8) * 255

        if np.max(mask_binary) == 0:
            return None

        return SelectionData(x=x0, y=y0, w=w, h=h, mask=mask_binary, polygon=polygon)

    @classmethod
    def rect_to_polygon(cls, p1: QPointF, p2: QPointF) -> QPolygonF:
        """Creates a rectangular QPolygonF between two points."""
        x0 = min(p1.x(), p2.x())
        y0 = min(p1.y(), p2.y())
        x1 = max(p1.x(), p2.x())
        y1 = max(p1.y(), p2.y())
        poly = QPolygonF()
        poly.append(QPointF(x0, y0))
        poly.append(QPointF(x1, y0))
        poly.append(QPointF(x1, y1))
        poly.append(QPointF(x0, y1))
        poly.append(QPointF(x0, y0))
        return poly
