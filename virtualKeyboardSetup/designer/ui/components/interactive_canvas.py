"""
Interactive QGraphicsView Canvas.
Manages sub-pixel grid overlay, paper boundary graphics, wheel zoom/pan,
user-placed interactive AprilTag markers, and real-time layout validation checking.
"""

from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QBrush, QColor, QFont, QPainter, QPen, QWheelEvent
from PySide6.QtWidgets import QFrame, QGraphicsScene, QGraphicsView, QSizePolicy, QWidget

from .button_graphics_item import ButtonGraphicsItem
from .marker_graphics_item import MarkerGraphicsItem
from core.models.button_model import ButtonModel
from core.models.marker_model import MarkerModel
from core.models.paper_layout import PaperLayoutModel
from core.geometry.marker_generator import generate_marker_layout
from core.geometry.layout_geometry import rects_overlap, rect_within_interior
from config.app_config import AppConfig


class InteractiveCanvas(QGraphicsView):
    selection_changed = Signal(object)  # ButtonModel or None
    marker_selection_changed = Signal(object)  # MarkerModel or None
    layout_updated = Signal(object)  # PaperLayoutModel

    def __init__(self, config: AppConfig, parent: QWidget | None = None):
        super().__init__(parent)
        self.config = config
        self.layout_model = PaperLayoutModel(
            paper_width_mm=config.paper_width_mm,
            paper_height_mm=config.paper_height_mm,
            paper_margin_mm=config.paper_margin_mm,
            marker_size_mm=config.marker_size_mm,
            marker_spacing_mm=config.marker_spacing_mm,
        )

        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)

        self.button_items: dict[str, ButtonGraphicsItem] = {}
        self.marker_items: dict[int, MarkerGraphicsItem] = {}
        self._zoom_factor = 1.0
        self._is_panning = False
        self._pan_start_pos = QPointF()
        self._show_outer_markers_in_view = True

        self.setRenderHint(QPainter.Antialiasing)
        self.setRenderHint(QPainter.SmoothPixmapTransform)
        self.setDragMode(QGraphicsView.RubberBandDrag)
        self.setViewportUpdateMode(QGraphicsView.FullViewportUpdate)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        # Show scrollbars only when content overflows
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setFrameShape(QFrame.NoFrame)
        self.setStyleSheet("QGraphicsView { border: none; outline: none; background-color: #1e1e23; }")

        self._setup_scene_bounds()
        self.refresh_markers()

    def _setup_scene_bounds(self) -> None:
        margin = 15.0
        self._scene.setSceneRect(
            -margin,
            -margin,
            self.config.paper_width_mm + (2 * margin),
            self.config.paper_height_mm + (2 * margin),
        )

    def toggle_view_outer_markers(self) -> bool:
        """Toggle outer AprilTag markers visibility for editing view purpose only."""
        self._show_outer_markers_in_view = not self._show_outer_markers_in_view
        self.refresh_markers()
        return self._show_outer_markers_in_view

    def refresh_markers(self) -> None:
        """Draw both outer perimeter AprilTag markers (if enabled in view) and user-placed custom markers."""
        # 1. Clean up ALL existing MarkerGraphicsItem objects from the scene
        for item in list(self._scene.items()):
            if isinstance(item, MarkerGraphicsItem):
                self._scene.removeItem(item)
        self.marker_items.clear()

        # 2. Draw outer perimeter AprilTag markers if toggle is enabled
        if self.config.show_outer_markers and self._show_outer_markers_in_view:
            outer_markers = generate_marker_layout(self.config)
            self.layout_model.markers = outer_markers
            for idx, m in enumerate(outer_markers):
                item = MarkerGraphicsItem(m, self.config, is_interactive=False)
                self._scene.addItem(item)
                self.marker_items[1000 + idx] = item
        else:
            self.layout_model.markers.clear()

        # 3. Draw user-placed custom interior markers
        if self.layout_model.use_custom_markers and self.layout_model.custom_markers:
            for m in self.layout_model.custom_markers:
                item = MarkerGraphicsItem(m, self.config, is_interactive=True)
                item.signals.selected_changed.connect(self._on_marker_selected)
                item.signals.position_changed.connect(self._on_marker_modified)
                self._scene.addItem(item)
                self.marker_items[m.id] = item

    def drawBackground(self, painter: QPainter, rect: QRectF) -> None:
        super().drawBackground(painter, rect)

        # 1. Outer Dark Canvas Fill
        painter.fillRect(rect, QColor(30, 30, 35))

        # 2. Paper Sheet Outline
        paper_rect = QRectF(0, 0, self.config.paper_width_mm, self.config.paper_height_mm)
        painter.fillRect(paper_rect, QColor(252, 252, 252))
        painter.setPen(QPen(QColor(180, 180, 180), 0.8))
        painter.drawRect(paper_rect)

        # 3. Grid Overlay
        if self.config.grid_snap_enabled and self.config.grid_size_mm > 0:
            grid_pen = QPen(QColor(220, 225, 230), 0.2, Qt.DashLine)
            painter.setPen(grid_pen)

            grid_sz = self.config.grid_size_mm
            x = 0.0
            while x <= self.config.paper_width_mm:
                painter.drawLine(QPointF(x, 0), QPointF(x, self.config.paper_height_mm))
                x += grid_sz

            y = 0.0
            while y <= self.config.paper_height_mm:
                painter.drawLine(QPointF(0, y), QPointF(self.config.paper_width_mm, y))
                y += grid_sz

        # 4. Interior Active Region Guideline
        interior_rect = QRectF(
            self.config.interior_x_min,
            self.config.interior_y_min,
            self.config.interior_width_mm,
            self.config.interior_height_mm,
        )
        interior_pen = QPen(QColor(0, 159, 239), 0.6, Qt.DashDotLine)
        painter.setPen(interior_pen)
        painter.drawRect(interior_rect)

        # 5. Project Name Text on Outer Margin Ring (Outside tags with gap)
        proj_name = getattr(self.layout_model, "project_name", "")
        if proj_name:
            top_ring_rect = QRectF(
                self.config.paper_margin_mm,
                1.0,
                self.config.paper_width_mm - (2 * self.config.paper_margin_mm),
                max(3.5, self.config.paper_margin_mm - 2.5),
            )
            name_font = QFont("Helvetica", 4.5, QFont.Bold)
            painter.setFont(name_font)
            painter.setPen(QPen(QColor(120, 125, 140)))
            painter.drawText(
                top_ring_rect,
                Qt.AlignCenter,
                f"PROJECT: {proj_name.upper()}",
            )

        # Draw "Active Surface Zone" text label in the middle of canvas ONLY if canvas has zero elements
        has_elements = bool(
            self.button_items
            or (self.layout_model.use_custom_markers and self.layout_model.custom_markers)
            or self.layout_model.buttons
        )
        if not has_elements:
            font = QFont("Helvetica", 9)
            painter.setFont(font)
            painter.setPen(QPen(QColor(0, 159, 239, 140)))
            painter.drawText(
                interior_rect,
                Qt.AlignCenter,
                "Active Surface Zone",
            )


    def add_button(self, button: ButtonModel) -> ButtonGraphicsItem:
        if not any(b.id == button.id for b in self.layout_model.buttons):
            self.layout_model.add_button(button)

        if button.id in self.button_items:
            item = self.button_items[button.id]
            item.prepareGeometryChange()
            item.setPos(button.x_mm, button.y_mm)
            item.update()
            self.validate_layout()
            return item

        item = ButtonGraphicsItem(button, self.config)

        item.signals.selected_changed.connect(self._on_button_selected)
        item.signals.position_changed.connect(self._on_button_modified)
        item.signals.geometry_changed.connect(self._on_button_modified)

        self._scene.clearSelection()
        self._scene.addItem(item)
        self.button_items[button.id] = item

        item.setSelected(True)
        self.validate_layout()
        self.layout_updated.emit(self.layout_model)
        return item

    def add_user_marker(self, marker: MarkerModel) -> MarkerGraphicsItem:
        if not any(m.id == marker.id for m in self.layout_model.custom_markers):
            self.layout_model.add_custom_marker(marker)

        if marker.id in self.marker_items:
            item = self.marker_items[marker.id]
            item.set_tag_id(marker.id)
            item.setPos(marker.x_mm, marker.y_mm)
            item.update()
            self.validate_layout()
            return item

        item = MarkerGraphicsItem(marker, self.config, is_interactive=True)

        item.signals.selected_changed.connect(self._on_marker_selected)
        item.signals.position_changed.connect(self._on_marker_modified)

        self._scene.clearSelection()
        self._scene.addItem(item)
        self.marker_items[marker.id] = item

        item.setSelected(True)
        self.validate_layout()
        self.layout_updated.emit(self.layout_model)
        return item



    def remove_selected_button(self) -> ButtonModel | None:
        for item in self._scene.selectedItems():
            if isinstance(item, ButtonGraphicsItem):
                b_id = item.button.id
                self.layout_model.remove_button(b_id)
                self._scene.removeItem(item)
                del self.button_items[b_id]
                self.validate_layout()
                self.layout_updated.emit(self.layout_model)
                self.selection_changed.emit(None)
                return item.button
        return None

    def remove_selected_marker(self) -> MarkerModel | None:
        for item in self._scene.selectedItems():
            if isinstance(item, MarkerGraphicsItem) and item.is_interactive:
                m_id = item.marker.id
                self.layout_model.remove_custom_marker(m_id)
                self._scene.removeItem(item)
                del self.marker_items[m_id]
                self.refresh_markers()
                self.validate_layout()
                self.layout_updated.emit(self.layout_model)
                self.marker_selection_changed.emit(None)
                return item.marker
        return None

    def update_paper_dimensions(
        self,
        width_mm: float,
        height_mm: float,
    ) -> None:
        """Dynamically update user configurable paper dimensions, recalculate scene bounds, and redraw AprilTag markers."""
        self.config.paper_width_mm = width_mm
        self.config.paper_height_mm = height_mm

        self.layout_model.paper_width_mm = width_mm
        self.layout_model.paper_height_mm = height_mm

        self._setup_scene_bounds()
        self.refresh_markers()
        self.fit_layout_to_view()
        self.viewport().update()
        self.validate_layout()
        self.layout_updated.emit(self.layout_model)

    def load_layout(self, layout: PaperLayoutModel) -> None:
        self.clear_all_buttons()
        self.layout_model = layout

        self.config.paper_width_mm = layout.paper_width_mm
        self.config.paper_height_mm = layout.paper_height_mm

        self._setup_scene_bounds()
        self.refresh_markers()
        self.fit_layout_to_view()

        for b in layout.buttons:
            item = ButtonGraphicsItem(b, self.config)
            item.signals.selected_changed.connect(self._on_button_selected)
            item.signals.position_changed.connect(self._on_button_modified)
            item.signals.geometry_changed.connect(self._on_button_modified)
            self._scene.addItem(item)
            self.button_items[b.id] = item

        self.validate_layout()
        self.layout_updated.emit(self.layout_model)

    def clear_all_buttons(self) -> None:
        for item in list(self.button_items.values()):
            self._scene.removeItem(item)
        self.button_items.clear()
        self.layout_model.clear()
        self.refresh_markers()
        self.validate_layout()
        self.layout_updated.emit(self.layout_model)

    def validate_layout(self) -> bool:
        """Check all buttons and markers for paper bounds and mutual collision overlaps."""
        all_valid = True
        buttons = [item.button for item in self.button_items.values()]
        button_gap = self.config.button_min_gap_mm
        marker_gap = self.config.marker_min_gap_mm

        outer_markers = []
        outer_ids = set()
        if self.config.show_outer_markers:
            outer_markers = generate_marker_layout(self.config)
            outer_ids = {om.id for om in outer_markers}

        custom_items = [item for item in self.marker_items.values() if item.is_interactive]
        custom_ids = [item.marker.id for item in custom_items]

        # 1. Validate Buttons (must fit in interior active surface zone and not overlap)
        for i, item in enumerate(self.button_items.values()):
            btn = item.button
            rect = btn.rect_tuple

            in_active_zone = rect_within_interior(rect, self.config)

            has_overlap = False
            for j, other_btn in enumerate(buttons):
                if i != j and rects_overlap(rect, other_btn.rect_tuple, gap_mm=button_gap):
                    has_overlap = True
                    break

            # Check overlap with custom markers
            if self.layout_model.use_custom_markers:
                btn_marker_gap = max(button_gap, marker_gap)
                for m_item in custom_items:
                    if rects_overlap(rect, m_item.marker.rect_tuple, gap_mm=btn_marker_gap):
                        has_overlap = True
                        break

            item.is_valid = in_active_zone and not has_overlap
            if not item.is_valid:
                all_valid = False

            item.update()

        # 2. Validate Custom Markers (must fit within paper margins, not overlap key buttons/markers, and have unique Tag IDs)
        for m_item in custom_items:
            m_id = m_item.marker.id
            m_rect = m_item.marker.rect_tuple
            in_bounds = (
                m_rect[0] + 1e-4 >= self.config.paper_margin_mm
                and m_rect[1] + 1e-4 >= self.config.paper_margin_mm
                and m_rect[0] + m_rect[2] - 1e-4 <= self.config.paper_width_mm - self.config.paper_margin_mm
                and m_rect[1] + m_rect[3] - 1e-4 <= self.config.paper_height_mm - self.config.paper_margin_mm
            )

            has_overlap = False
            btn_marker_gap = max(button_gap, marker_gap)
            for btn in buttons:
                if rects_overlap(m_rect, btn.rect_tuple, gap_mm=btn_marker_gap):
                    has_overlap = True
                    break

            if not has_overlap:
                for other_item in custom_items:
                    if other_item.marker.id != m_id:
                        if rects_overlap(m_rect, other_item.marker.rect_tuple, gap_mm=marker_gap):
                            has_overlap = True
                            break

            if not has_overlap and outer_markers:
                for om in outer_markers:
                    if rects_overlap(m_rect, om.rect_tuple, gap_mm=marker_gap):
                        has_overlap = True
                        break

            is_duplicate_id = (custom_ids.count(m_id) > 1) or (m_id in outer_ids)

            m_item.is_valid = in_bounds and not has_overlap and not is_duplicate_id
            if not m_item.is_valid:
                all_valid = False
            m_item.update()

        return all_valid


    def _on_button_selected(self, button: ButtonModel | None) -> None:
        self.selection_changed.emit(button)

    def _on_button_modified(self, button: ButtonModel) -> None:
        self.validate_layout()
        self.layout_updated.emit(self.layout_model)

    def _on_marker_selected(self, marker: MarkerModel | None) -> None:
        self.marker_selection_changed.emit(marker)

    def _on_marker_modified(self, marker: MarkerModel) -> None:
        self.validate_layout()
    def remove_button_by_id(self, b_id: str) -> None:
        """Remove button GraphicsItem by ID."""
        if b_id in self.button_items:
            item = self.button_items[b_id]
            self.layout_model.remove_button(b_id)
            self._scene.removeItem(item)
            del self.button_items[b_id]
            self.validate_layout()
            self.layout_updated.emit(self.layout_model)
            self.selection_changed.emit(None)

    def remove_marker_by_id(self, m_id: int) -> None:
        """Remove custom marker GraphicsItem by tag ID."""
        if m_id in self.marker_items:
            item = self.marker_items[m_id]
            self.layout_model.remove_custom_marker(m_id)
            if item in self._scene.items():
                self._scene.removeItem(item)
            del self.marker_items[m_id]
            self.refresh_markers()
            self.validate_layout()
            self.layout_updated.emit(self.layout_model)
            self.marker_selection_changed.emit(None)

    def fit_layout_to_view(self) -> None:
        """Scale and center paper sheet layout inside view canvas bounds."""
        margin = 15.0
        paper_rect = QRectF(
            -margin,
            -margin,
            self.config.paper_width_mm + (2 * margin),
            self.config.paper_height_mm + (2 * margin),
        )
        self.resetTransform()
        self.fitInView(paper_rect, Qt.KeepAspectRatio)
        self._zoom_factor = 1.0

    def showEvent(self, event) -> None:
        super().showEvent(event)
        self.fit_layout_to_view()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if abs(self._zoom_factor - 1.0) < 0.05:
            self.fit_layout_to_view()

    def remove_selected_element(self) -> None:
        """Remove any currently selected item (whether key button or custom marker)."""
        selected_items = list(self._scene.selectedItems())
        if not selected_items:
            # Fallback check if model tracks selected button or marker
            self.remove_selected_button()
            self.remove_selected_marker()
            return

        for item in selected_items:
            if isinstance(item, ButtonGraphicsItem):
                b_id = item.button.id
                self.layout_model.remove_button(b_id)
                self._scene.removeItem(item)
                if b_id in self.button_items:
                    del self.button_items[b_id]
            elif isinstance(item, MarkerGraphicsItem) and item.is_interactive:
                m_id = item.marker.id
                self.layout_model.remove_custom_marker(m_id)
                self._scene.removeItem(item)
                if m_id in self.marker_items:
                    del self.marker_items[m_id]

        self.refresh_markers()
        self.validate_layout()
        self.layout_updated.emit(self.layout_model)
        self.selection_changed.emit(None)
        self.marker_selection_changed.emit(None)

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MiddleButton:
            self._is_panning = True
            self._pan_start_pos = event.position().toPoint()
            self.setCursor(Qt.ClosedHandCursor)
            event.accept()
            return

        if event.button() == Qt.LeftButton:
            item = self.itemAt(event.position().toPoint())
            if item is None:
                # Clicked on empty canvas background: clear all selections!
                self._scene.clearSelection()
                self.selection_changed.emit(None)
                self.marker_selection_changed.emit(None)

        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        if self._is_panning:
            delta = event.position().toPoint() - self._pan_start_pos
            self._pan_start_pos = event.position().toPoint()
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - delta.x())
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - delta.y())
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        if event.button() == Qt.MiddleButton and self._is_panning:
            self._is_panning = False
            self.setCursor(Qt.ArrowCursor)
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def keyPressEvent(self, event) -> None:
        if event.key() in (Qt.Key_Delete, Qt.Key_Backspace):
            self.remove_selected_element()
            return
        super().keyPressEvent(event)

    def zoom_by(self, factor: float) -> None:
        """Zoom view in or out by given scaling factor, updating tracking state."""
        new_zoom = self._zoom_factor * factor
        if 0.3 <= new_zoom <= 8.0:
            self._zoom_factor = new_zoom
            self.scale(factor, factor)

    def wheelEvent(self, event: QWheelEvent) -> None:
        delta = event.angleDelta().y()
        if delta != 0:
            factor = 1.15 if delta > 0 else 0.85
            self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
            self.zoom_by(factor)
            self.setTransformationAnchor(QGraphicsView.AnchorViewCenter)
            event.accept()
        else:
            super().wheelEvent(event)
