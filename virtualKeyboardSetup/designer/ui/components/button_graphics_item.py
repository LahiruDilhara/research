"""
Interactive Button QGraphicsItem.
Manages interactive drag-and-drop key placement, 8 sizing handles, grid snapping,
and real-time validation color cues on QGraphicsScene.
"""

from PySide6.QtCore import QPointF, QRectF, Qt, Signal, QObject
from PySide6.QtGui import QBrush, QColor, QCursor, QFont, QPainter, QPen

from PySide6.QtWidgets import QGraphicsItem

from core.models.button_model import ButtonModel
from core.geometry.layout_geometry import rects_overlap, rect_within_interior, snap_to_grid
from config.app_config import AppConfig


class ButtonGraphicsItemSignals(QObject):
    selected_changed = Signal(object)  # ButtonModel or None
    position_changed = Signal(object)  # ButtonModel
    geometry_changed = Signal(object)  # ButtonModel


class ButtonGraphicsItem(QGraphicsItem):
    HANDLE_SIZE_MM = 2.4

    # Resize handles: 0: TL, 1: T, 2: TR, 3: R, 4: BR, 5: B, 6: BL, 7: L
    HANDLE_TL, HANDLE_T, HANDLE_TR, HANDLE_R = 0, 1, 2, 3
    HANDLE_BR, HANDLE_B, HANDLE_BL, HANDLE_L = 4, 5, 6, 7

    HANDLE_CURSORS = {
        HANDLE_TL: Qt.SizeFDiagCursor,
        HANDLE_BR: Qt.SizeFDiagCursor,
        HANDLE_TR: Qt.SizeBDiagCursor,
        HANDLE_BL: Qt.SizeBDiagCursor,
        HANDLE_T: Qt.SizeVerCursor,
        HANDLE_B: Qt.SizeVerCursor,
        HANDLE_L: Qt.SizeHorCursor,
        HANDLE_R: Qt.SizeHorCursor,
    }

    def __init__(
        self,
        button: ButtonModel,
        config: AppConfig,
        parent: QGraphicsItem | None = None,
    ):
        super().__init__(parent)
        self.button = button
        self.config = config
        self.signals = ButtonGraphicsItemSignals()

        self.setFlag(QGraphicsItem.ItemIsSelectable, True)
        self.setFlag(QGraphicsItem.ItemIsMovable, True)
        self.setFlag(QGraphicsItem.ItemSendsGeometryChanges, True)
        self.setAcceptHoverEvents(True)

        self.setPos(button.x_mm, button.y_mm)
        self.is_valid = True
        self._active_handle = None
        self._drag_start_pos = QPointF()
        self._drag_start_rect = (button.x_mm, button.y_mm, button.width_mm, button.height_mm)

    def boundingRect(self) -> QRectF:
        h = self.HANDLE_SIZE_MM
        return QRectF(-h, -h, self.button.width_mm + 2 * h, self.button.height_mm + 2 * h)

    def get_handle_rects(self) -> dict[int, QRectF]:
        w, h = self.button.width_mm, self.button.height_mm
        hs = self.HANDLE_SIZE_MM
        half_hs = hs / 2.0

        return {
            self.HANDLE_TL: QRectF(-half_hs, -half_hs, hs, hs),
            self.HANDLE_T: QRectF(w / 2 - half_hs, -half_hs, hs, hs),
            self.HANDLE_TR: QRectF(w - half_hs, -half_hs, hs, hs),
            self.HANDLE_R: QRectF(w - half_hs, h / 2 - half_hs, hs, hs),
            self.HANDLE_BR: QRectF(w - half_hs, h - half_hs, hs, hs),
            self.HANDLE_B: QRectF(w / 2 - half_hs, h - half_hs, hs, hs),
            self.HANDLE_BL: QRectF(-half_hs, h - half_hs, hs, hs),
            self.HANDLE_L: QRectF(-half_hs, h / 2 - half_hs, hs, hs),
        }

    def paint(self, painter: QPainter, option, widget=None) -> None:
        painter.setRenderHint(QPainter.Antialiasing)

        # Body background and outline colors
        if not self.is_valid:
            fill_color = QColor(239, 68, 68, 40)
            stroke_color = QColor(239, 68, 68)
        elif self.isSelected():
            fill_color = QColor(0, 159, 239, 30)
            stroke_color = QColor(0, 159, 239)
        else:
            fill_color = QColor(255, 255, 255, 220)
            stroke_color = QColor(40, 40, 40)

        pen = QPen(stroke_color, self.config.button_stroke_width_mm)
        painter.setPen(pen)
        painter.setBrush(QBrush(fill_color))

        rect = QRectF(0, 0, self.button.width_mm, self.button.height_mm)
        radius = self.config.button_corner_radius_mm
        if radius > 0:
            painter.drawRoundedRect(rect, radius, radius)
        else:
            painter.drawRect(rect)

        # Label Text
        if self.button.text:
            painter.setPen(QPen(QColor(20, 20, 20)))
            font = QFont("Helvetica", self.button.font_size_pt)
            font.setPointSizeF(self.button.font_size_pt * 0.4)
            painter.setFont(font)
            painter.drawText(rect, Qt.AlignCenter, self.button.text)

        # Draw handles when selected
        if self.isSelected():
            painter.setPen(QPen(QColor(0, 159, 239), 0.5))
            painter.setBrush(QBrush(QColor(255, 255, 255)))
            for handle_rect in self.get_handle_rects().values():
                painter.drawRect(handle_rect)

    def hoverMoveEvent(self, event) -> None:
        if self.isSelected():
            pos = event.pos()
            for handle_id, handle_rect in self.get_handle_rects().items():
                if handle_rect.contains(pos):
                    self.setCursor(QCursor(self.HANDLE_CURSORS[handle_id]))
                    event.accept()
                    return
            self.setCursor(QCursor(Qt.SizeAllCursor))
        else:
            self.unsetCursor()
        super().hoverMoveEvent(event)

    def hoverLeaveEvent(self, event) -> None:
        self.unsetCursor()
        super().hoverLeaveEvent(event)

    def mousePressEvent(self, event) -> None:
        if self.isSelected():
            pos = event.pos()
            for handle_id, handle_rect in self.get_handle_rects().items():
                if handle_rect.contains(pos):
                    self._active_handle = handle_id
                    self._drag_start_pos = event.scenePos()
                    self._drag_start_rect = (
                        self.button.x_mm,
                        self.button.y_mm,
                        self.button.width_mm,
                        self.button.height_mm,
                    )
                    self.setCursor(QCursor(self.HANDLE_CURSORS[handle_id]))
                    event.accept()
                    return

        self._active_handle = None
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        if self._active_handle is not None:
            delta = event.scenePos() - self._drag_start_pos
            orig_x, orig_y, orig_w, orig_h = self._drag_start_rect

            new_x, new_y = orig_x, orig_y
            new_w, new_h = orig_w, orig_h

            if self._active_handle in (self.HANDLE_TL, self.HANDLE_L, self.HANDLE_BL):
                new_w = max(self.config.button_min_width_mm, orig_w - delta.x())
                new_x = orig_x + (orig_w - new_w)
            elif self._active_handle in (self.HANDLE_TR, self.HANDLE_R, self.HANDLE_BR):
                new_w = max(self.config.button_min_width_mm, orig_w + delta.x())

            if self._active_handle in (self.HANDLE_TL, self.HANDLE_T, self.HANDLE_TR):
                new_h = max(self.config.button_min_height_mm, orig_h - delta.y())
                new_y = orig_y + (orig_h - new_h)
            elif self._active_handle in (self.HANDLE_BL, self.HANDLE_B, self.HANDLE_BR):
                new_h = max(self.config.button_min_height_mm, orig_h + delta.y())

            if self.config.grid_snap_enabled:
                new_x = snap_to_grid(new_x, self.config.grid_size_mm)
                new_y = snap_to_grid(new_y, self.config.grid_size_mm)
                new_w = snap_to_grid(new_w, self.config.grid_size_mm)
                new_h = snap_to_grid(new_h, self.config.grid_size_mm)

            # Clamp inside active surface zone
            new_x = max(self.config.interior_x_min, min(new_x, self.config.interior_x_max - new_w))
            new_y = max(self.config.interior_y_min, min(new_y, self.config.interior_y_max - new_h))

            self.prepareGeometryChange()
            self.button.x_mm = new_x
            self.button.y_mm = new_y
            self.button.width_mm = new_w
            self.button.height_mm = new_h
            self.setPos(new_x, new_y)

            self.setCursor(QCursor(self.HANDLE_CURSORS[self._active_handle]))
            self.signals.geometry_changed.emit(self.button)
            event.accept()
            return

        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        self._active_handle = None
        pos = event.pos()
        if self.isSelected():
            for handle_id, handle_rect in self.get_handle_rects().items():
                if handle_rect.contains(pos):
                    self.setCursor(QCursor(self.HANDLE_CURSORS[handle_id]))
                    super().mouseReleaseEvent(event)
                    return
            self.setCursor(QCursor(Qt.SizeAllCursor))
        else:
            self.unsetCursor()
        super().mouseReleaseEvent(event)

    def itemChange(self, change: QGraphicsItem.GraphicsItemChange, value):
        if change == QGraphicsItem.ItemPositionChange and self.scene():
            new_pos = value
            new_x = new_pos.x()
            new_y = new_pos.y()

            # Clamp item dragging within active surface zone
            max_x = self.config.interior_x_max - self.button.width_mm
            max_y = self.config.interior_y_max - self.button.height_mm
            new_x = max(self.config.interior_x_min, min(new_x, max_x))
            new_y = max(self.config.interior_y_min, min(new_y, max_y))

            if self.config.grid_snap_enabled and self._active_handle is None:
                new_x = snap_to_grid(new_x, self.config.grid_size_mm)
                new_y = snap_to_grid(new_y, self.config.grid_size_mm)
                new_x = max(self.config.interior_x_min, min(new_x, max_x))
                new_y = max(self.config.interior_y_min, min(new_y, max_y))

            self.button.x_mm = new_x
            self.button.y_mm = new_y
            self.signals.position_changed.emit(self.button)
            return QPointF(new_x, new_y)

        elif change == QGraphicsItem.ItemSelectedHasChanged:
            is_sel = bool(value)
            if not is_sel:
                self.unsetCursor()
            self.signals.selected_changed.emit(self.button if is_sel else None)

        return super().itemChange(change, value)

