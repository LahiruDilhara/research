"""
ui/components/interactive_layout_map.py

Interactive 2D visual layout map for paper keyboard layout inspection and action assignment.
Renders paper boundaries, AprilTag fiducial anchors, project title, and clickable button regions with action status badges.
"""

from __future__ import annotations

import math
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import (
    QBrush,
    QColor,
    QFont,
    QFontMetrics,
    QPainter,
    QPainterPath,
    QPen,
)
from PySide6.QtWidgets import QSizePolicy, QWidget

from config.constants import UI_ACCENT, UI_BG_CARD, UI_BG_DARK, UI_TEXT_PRI, UI_TEXT_SEC
from core.action.action_executor import ActionData
from core.layout.layout_parser import ButtonData, LayoutData, MarkerData


_ACTION_COLORS: dict[str, QColor] = {
    "keystroke": QColor("#38BDF8"),  # Soft Sky Blue
    "shortcut":  QColor("#A78BFA"),  # Soft Purple / Lavender
    "shell":     QColor("#FBBF24"),  # Soft Amber
    "macro":     QColor("#34D399"),  # Soft Emerald
    "none":      QColor("#64748B"),  # Subtle Slate
}


class InteractiveLayoutMapWidget(QWidget):
    """Interactive paper layout canvas where users can visually inspect and click buttons to configure."""

    button_clicked = Signal(str)  # button_id

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("interactiveLayoutMap")
        self.setAttribute(Qt.WA_Hover, True)
        self.setMouseTracking(True)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMinimumSize(460, 360)

        self._layout: LayoutData | None = None
        self._actions: dict[str, ActionData] = {}
        self._selected_button_id: str | None = None
        self._hovered_button_id: str | None = None

        # Transformation cache: mm to widget pixel space
        self._scale: float = 1.0
        self._offset_x: float = 0.0
        self._offset_y: float = 0.0
        self._paper_rect: QRectF = QRectF()

    # ── Public API ─────────────────────────────────────────────────────────────

    def set_layout(self, layout: LayoutData) -> None:
        self._layout = layout
        if layout.buttons and self._selected_button_id is None:
            self._selected_button_id = layout.buttons[0].id
        self.update()

    def set_actions(self, actions: dict[str, ActionData]) -> None:
        self._actions = dict(actions)
        self.update()

    def set_action_for_button(self, button_id: str, action: ActionData) -> None:
        self._actions[button_id] = action
        self.update()

    def select_button(self, button_id: str) -> None:
        if self._selected_button_id != button_id:
            self._selected_button_id = button_id
            self.update()

    # ── Paint Event ────────────────────────────────────────────────────────────

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.setRenderHint(QPainter.TextAntialiasing, True)

        # Background
        painter.fillRect(self.rect(), QColor("#101015"))

        if not self._layout:
            painter.setPen(QColor(UI_TEXT_SEC))
            painter.setFont(QFont("Segoe UI", 13))
            painter.drawText(self.rect(), Qt.AlignCenter, "No layout loaded")
            return

        self._compute_transform()

        # Draw paper surface
        self._draw_paper(painter)

        # Draw AprilTag fiducial anchors
        self._draw_markers(painter)

        # Draw Key Buttons
        self._draw_buttons(painter)

    def _compute_transform(self) -> None:
        if not self._layout:
            return

        w_widget = self.width()
        h_widget = self.height()
        margin_x = 28.0
        margin_y = 36.0  # extra margin for header title

        avail_w = max(10.0, w_widget - margin_x * 2)
        avail_h = max(10.0, h_widget - margin_y * 2)

        scale_x = avail_w / self._layout.paper_width_mm
        scale_y = avail_h / self._layout.paper_height_mm
        self._scale = min(scale_x, scale_y)

        paper_px_w = self._layout.paper_width_mm * self._scale
        paper_px_h = self._layout.paper_height_mm * self._scale

        self._offset_x = (w_widget - paper_px_w) / 2.0
        self._offset_y = (h_widget - paper_px_h) / 2.0 + 10.0

        self._paper_rect = QRectF(self._offset_x, self._offset_y, paper_px_w, paper_px_h)

    def _mm_to_px(self, x_mm: float, y_mm: float) -> tuple[float, float]:
        return (
            self._offset_x + x_mm * self._scale,
            self._offset_y + y_mm * self._scale,
        )

    def _px_to_mm(self, px_x: float, px_y: float) -> tuple[float, float]:
        if self._scale <= 0:
            return 0.0, 0.0
        return (
            (px_x - self._offset_x) / self._scale,
            (px_y - self._offset_y) / self._scale,
        )

    # ── Drawing Helpers ────────────────────────────────────────────────────────

    def _draw_paper(self, painter: QPainter) -> None:
        # Paper drop shadow
        shadow_rect = self._paper_rect.adjusted(-3, -3, 3, 3)
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(0, 0, 0, 100))
        painter.drawRoundedRect(shadow_rect, 10, 10)

        # Paper sheet surface
        painter.setPen(QPen(QColor(255, 255, 255, 30), 1.2))
        painter.setBrush(QColor("#1A1A24"))
        painter.drawRoundedRect(self._paper_rect, 8, 8)

        # Layout Title / Project Name at top of canvas (Centered)
        title_text = self._layout.project_name or "Paper Virtual Keyboard"
        painter.setPen(QColor("#E2E8F0"))
        painter.setFont(QFont("Segoe UI", 12, QFont.Bold))
        title_rect = QRectF(self._paper_rect.left(), self._paper_rect.top() - 28, self._paper_rect.width(), 22)
        painter.drawText(title_rect, Qt.AlignCenter, title_text)

        # Dimension watermark & Button count at bottom (Centered)
        dim_text = f"{self._layout.paper_width_mm:.0f} × {self._layout.paper_height_mm:.0f} mm  •  {len(self._layout.buttons)} keys  •  {len(self._layout.markers)} anchors"
        painter.setPen(QColor("#64748B"))
        painter.setFont(QFont("Segoe UI", 9, QFont.Medium))
        dim_rect = QRectF(self._paper_rect.left(), self._paper_rect.bottom() + 6, self._paper_rect.width(), 20)
        painter.drawText(dim_rect, Qt.AlignCenter, dim_text)

    def _draw_markers(self, painter: QPainter) -> None:
        for m in self._layout.markers:
            cx, cy = self._mm_to_px(m.center_x_mm, m.center_y_mm)
            sz_px = m.size_mm * self._scale
            m_rect = QRectF(cx - sz_px / 2.0, cy - sz_px / 2.0, sz_px, sz_px)

            # Outer black square
            painter.setPen(Qt.NoPen)
            painter.setBrush(QColor("#000000"))
            painter.drawRect(m_rect)

            # Inner white square
            inner_sz = sz_px * 0.65
            inner_rect = QRectF(cx - inner_sz / 2.0, cy - inner_sz / 2.0, inner_sz, inner_sz)
            painter.setBrush(QColor("#FFFFFF"))
            painter.drawRect(inner_rect)

            # Center black square
            core_sz = sz_px * 0.35
            core_rect = QRectF(cx - core_sz / 2.0, cy - core_sz / 2.0, core_sz, core_sz)
            painter.setBrush(QColor("#000000"))
            painter.drawRect(core_rect)

        # Reset brush after drawing all markers
        painter.setBrush(Qt.NoBrush)

    def _draw_buttons(self, painter: QPainter) -> None:
        for b in self._layout.buttons:
            px_x, px_y = self._mm_to_px(b.x_mm, b.y_mm)
            px_w = b.width_mm * self._scale
            px_h = b.height_mm * self._scale
            btn_rect = QRectF(px_x, px_y, px_w, px_h)

            is_selected = (b.id == self._selected_button_id)
            is_hovered  = (b.id == self._hovered_button_id)

            act = self._actions.get(b.id, ActionData(type="none", value=""))
            has_action = (act.type != "none" and bool(act.value))
            tint = _ACTION_COLORS.get(act.type, _ACTION_COLORS["none"])

            # Button background path
            path = QPainterPath()
            path.addRoundedRect(btn_rect, 5, 5)

            # All buttons share the exact same clean base background
            painter.fillPath(path, QColor("#1E1E28"))

            if is_selected:
                border_pen = QPen(QColor(UI_ACCENT), 1.8)
            elif is_hovered:
                border_pen = QPen(QColor(255, 255, 255, 90), 1.2)
            elif has_action:
                border_pen = QPen(QColor(tint.red(), tint.green(), tint.blue(), 80), 1.0)
            else:
                border_pen = QPen(QColor(255, 255, 255, 30), 1.0)

            # Stroke border only (no brush fill leakage)
            painter.strokePath(path, border_pen)

            # Standardized Key Label (moderated font size)
            label_text = b.label if b.label.strip() else b.id
            font_sz = max(8, int(min(px_h * 0.28, px_w * 0.20, 12)))
            font = QFont("Segoe UI", font_sz, QFont.Bold)
            painter.setFont(font)
            painter.setPen(QColor("#FFFFFF" if (is_selected or is_hovered or has_action) else "#CBD5E1"))

            # Calculate label and subtitle boxes
            has_sub = bool(has_action or px_h > 22)
            if has_sub:
                lbl_box = QRectF(btn_rect.left() + 3, btn_rect.top() + 2, btn_rect.width() - 6, btn_rect.height() * 0.52)
                sub_box = QRectF(btn_rect.left() + 3, btn_rect.top() + btn_rect.height() * 0.48, btn_rect.width() - 6, btn_rect.height() * 0.46)
            else:
                lbl_box = btn_rect
                sub_box = QRectF()

            painter.drawText(lbl_box, Qt.AlignCenter, label_text)

            # Action subtitle badge
            if has_action:
                painter.setPen(tint)
                sub_font = QFont("Segoe UI", max(7, font_sz - 2), QFont.Medium)
                painter.setFont(sub_font)
                act_str = f"[{act.value}]" if len(act.value) <= 10 else f"[{act.value[:8]}...]"
                painter.drawText(sub_box, Qt.AlignCenter, act_str)
            elif has_sub:
                painter.setPen(QColor("#64748B"))
                sub_font = QFont("Segoe UI", max(7, font_sz - 3))
                painter.setFont(sub_font)
                painter.drawText(sub_box, Qt.AlignCenter, b.id)

            # Small status dot in top right
            dot_r = 3.0
            dot_center = QPointF(btn_rect.right() - 6, btn_rect.top() + 6)
            painter.setPen(Qt.NoPen)
            if has_action:
                painter.setBrush(tint)
            else:
                painter.setBrush(QColor(255, 255, 255, 20))
            painter.drawEllipse(dot_center, dot_r, dot_r)
            painter.setBrush(Qt.NoBrush)

    # ── Mouse Interaction ──────────────────────────────────────────────────────

    def mouseMoveEvent(self, event) -> None:
        if not self._layout:
            return

        pos = event.position()
        mm_x, mm_y = self._px_to_mm(pos.x(), pos.y())

        hovered_btn = None
        for b in self._layout.buttons:
            if b.contains_mm(mm_x, mm_y):
                hovered_btn = b.id
                break

        if self._hovered_button_id != hovered_btn:
            self._hovered_button_id = hovered_btn
            if hovered_btn:
                self.setCursor(Qt.PointingHandCursor)
            else:
                self.setCursor(Qt.ArrowCursor)
            self.update()

    def mousePressEvent(self, event) -> None:
        if not self._layout or event.button() != Qt.LeftButton:
            return

        pos = event.position()
        mm_x, mm_y = self._px_to_mm(pos.x(), pos.y())

        for b in self._layout.buttons:
            if b.contains_mm(mm_x, mm_y):
                self._selected_button_id = b.id
                self.button_clicked.emit(b.id)
                self.update()
                break

    def leaveEvent(self, event) -> None:
        if self._hovered_button_id is not None:
            self._hovered_button_id = None
            self.setCursor(Qt.ArrowCursor)
            self.update()
