"""
AprilTag Marker QGraphicsItem.
Interactive visual rendering of AprilTag fiducial anchors on PySide6 QGraphicsScene.
Supports drag-and-drop placement, grid snapping, and selection highlights.
"""

import cv2
from PySide6.QtCore import QPointF, QRectF, Qt, Signal, QObject
from PySide6.QtGui import QBrush, QColor, QFont, QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import QGraphicsItem

from core.models.marker_model import MarkerModel
from core.geometry.layout_geometry import snap_to_grid
from config.app_config import AppConfig


class MarkerGraphicsItemSignals(QObject):
    selected_changed = Signal(object)  # MarkerModel or None
    position_changed = Signal(object)  # MarkerModel


class MarkerGraphicsItem(QGraphicsItem):
    _pixmap_cache: dict[int, QPixmap] = {}

    def __init__(
        self,
        marker: MarkerModel,
        config: AppConfig | None = None,
        is_interactive: bool = True,
        parent: QGraphicsItem | None = None,
    ):
        super().__init__(parent)
        self.marker = marker
        self.config = config or AppConfig()
        self.is_interactive = is_interactive
        self.signals = MarkerGraphicsItemSignals()
        self.is_valid = True

        if is_interactive:
            self.setFlag(QGraphicsItem.ItemIsSelectable, True)
            self.setFlag(QGraphicsItem.ItemIsMovable, True)
            self.setFlag(QGraphicsItem.ItemSendsGeometryChanges, True)
        else:
            self.setFlag(QGraphicsItem.ItemIsSelectable, False)
            self.setFlag(QGraphicsItem.ItemIsMovable, False)

        self.setPos(marker.x_mm, marker.y_mm)
        self._pixmap = self._get_cached_pixmap(marker.id)

    @classmethod
    def _get_cached_pixmap(cls, tag_id: int) -> QPixmap:
        if tag_id not in cls._pixmap_cache:
            tag_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
            img_arr = cv2.aruco.generateImageMarker(tag_dict, tag_id, 200)
            height, width = img_arr.shape
            qimg = QImage(img_arr.data, width, height, width, QImage.Format_Grayscale8)
            cls._pixmap_cache[tag_id] = QPixmap.fromImage(qimg)
        return cls._pixmap_cache[tag_id]

    def set_tag_id(self, tag_id: int) -> None:
        self.marker.id = tag_id
        self._pixmap = self._get_cached_pixmap(tag_id)
        self.update()

    def boundingRect(self) -> QRectF:
        half = self.marker.size_mm / 2.0
        gap_mm = 2.0
        badge_h = 2.4
        return QRectF(
            -half,
            -half - gap_mm - badge_h,
            self.marker.size_mm,
            self.marker.size_mm + gap_mm + badge_h,
        )

    def paint(self, painter: QPainter, option, widget=None) -> None:
        painter.setRenderHint(QPainter.SmoothPixmapTransform)
        half = self.marker.size_mm / 2.0
        rect = QRectF(-half, -half, self.marker.size_mm, self.marker.size_mm)

        # Draw AprilTag Pixmap
        painter.drawPixmap(rect, self._pixmap, QRectF(self._pixmap.rect()))

        # Selection or invalid border highlight
        if not self.is_valid:
            painter.setPen(QPen(QColor(239, 68, 68), 1.2))
            painter.drawRect(rect)
        elif self.isSelected():
            painter.setPen(QPen(QColor(0, 159, 239), 1.5))
            painter.drawRect(rect)
        else:
            painter.setPen(QPen(QColor(100, 100, 100), 0.5))
            painter.drawRect(rect)

        # Subtle, Crisp Tag ID Overlay with 2.0mm gap above marker top border
        painter.setRenderHint(QPainter.TextAntialiasing)
        font = QFont("Helvetica")
        font.setPointSizeF(2.2)
        font.setWeight(QFont.Normal)
        painter.setFont(font)
        painter.setPen(QPen(QColor(100, 100, 100, 220)))
        gap_mm = 2.0
        badge_h = 2.4
        badge_rect = QRectF(
            rect.left(),
            rect.top() - gap_mm - badge_h,
            self.marker.size_mm,
            badge_h,
        )
        painter.drawText(badge_rect, Qt.AlignCenter, f"#{self.marker.id}")

    def itemChange(self, change: QGraphicsItem.GraphicsItemChange, value):
        if change == QGraphicsItem.ItemPositionChange and self.scene() and self.is_interactive:
            new_pos = value
            new_x = new_pos.x()
            new_y = new_pos.y()

            # Clamp position strictly within interior active surface zone
            half = self.marker.size_mm / 2.0
            min_x = self.config.interior_x_min + half
            max_x = self.config.interior_x_max - half
            min_y = self.config.interior_y_min + half
            max_y = self.config.interior_y_max - half

            new_x = max(min_x, min(new_x, max_x))
            new_y = max(min_y, min(new_y, max_y))

            if self.config.grid_snap_enabled:
                new_x = snap_to_grid(new_x, self.config.grid_size_mm)
                new_y = snap_to_grid(new_y, self.config.grid_size_mm)
                new_x = max(min_x, min(new_x, max_x))
                new_y = max(min_y, min(new_y, max_y))

            self.marker.x_mm = new_x
            self.marker.y_mm = new_y
            self.signals.position_changed.emit(self.marker)
            return QPointF(new_x, new_y)

        elif change == QGraphicsItem.ItemSelectedHasChanged and self.is_interactive:
            is_sel = bool(value)
            self.signals.selected_changed.emit(self.marker if is_sel else None)


        return super().itemChange(change, value)
