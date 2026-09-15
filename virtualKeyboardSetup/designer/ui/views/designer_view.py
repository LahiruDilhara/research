"""
Designer View Component (MVVM View Layer).
Pure UI view presenting interactive layout canvas and property control side panel,
bound reactively to DesignerViewModel.
"""

from PySide6.QtCore import QEasingCurve, QPropertyAnimation, Qt
from PySide6.QtWidgets import QHBoxLayout, QSizePolicy, QWidget
from qfluentwidgets import CardWidget, FluentIcon, TransparentToolButton

from ui.components.interactive_canvas import InteractiveCanvas
from ui.components.side_panel import SidePanel
from core.models.button_model import ButtonModel
from core.models.marker_model import MarkerModel
from viewmodels.designer_viewmodel import DesignerViewModel


class DesignerView(QWidget):
    def __init__(self, viewmodel: DesignerViewModel, parent: QWidget | None = None):
        super().__init__(parent)
        self.setObjectName("designerView")
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.viewmodel = viewmodel

        self._is_panel_expanded = True
        self._setup_ui()
        self._setup_animation()
        self._bind_viewmodel()

    def _setup_ui(self) -> None:
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Interactive Canvas
        self.canvas = InteractiveCanvas(self.viewmodel.config, self)
        layout.addWidget(self.canvas, stretch=3)

        # Floating Top Canvas Toolbar with high-contrast dark elevation background
        self.toolbar_card = CardWidget(self.canvas)
        self.toolbar_card.setStyleSheet(
            "CardWidget { background-color: #282830; border: 1px solid #3e3e4a; border-radius: 8px; }"
        )
        self.toolbar_layout = QHBoxLayout(self.toolbar_card)
        self.toolbar_layout.setContentsMargins(6, 4, 6, 4)
        self.toolbar_layout.setSpacing(4)

        self.btn_tool_open = TransparentToolButton(FluentIcon.FOLDER, self.toolbar_card)
        self.btn_tool_open.setToolTip("Open Layout")
        self.btn_tool_open.clicked.connect(self._on_open_layout)

        self.btn_tool_save = TransparentToolButton(FluentIcon.SAVE, self.toolbar_card)
        self.btn_tool_save.setToolTip("Save Layout (Overwrite / Quick Save)")
        self.btn_tool_save.clicked.connect(self._on_save_layout)

        self.btn_tool_save_as = TransparentToolButton(FluentIcon.SAVE_AS, self.toolbar_card)
        self.btn_tool_save_as.setToolTip("Save Layout As...")
        self.btn_tool_save_as.clicked.connect(self._on_save_layout_as)

        self.btn_tool_markers = TransparentToolButton(FluentIcon.TAG, self.toolbar_card)
        self.btn_tool_markers.setToolTip("Toggle Outer AprilTag Markers (Editor View Only)")
        self.btn_tool_markers.clicked.connect(self._on_toggle_outer_markers_view)

        self.btn_tool_fit = TransparentToolButton(FluentIcon.ZOOM_IN, self.toolbar_card)
        self.btn_tool_fit.setToolTip("Fit Paper to View")
        self.btn_tool_fit.clicked.connect(self.canvas.fit_layout_to_view)

        self.btn_tool_zoomin = TransparentToolButton(FluentIcon.ADD, self.toolbar_card)
        self.btn_tool_zoomin.setToolTip("Zoom In")
        self.btn_tool_zoomin.clicked.connect(lambda: self.canvas.zoom_by(1.2))

        self.btn_tool_zoomout = TransparentToolButton(FluentIcon.REMOVE, self.toolbar_card)
        self.btn_tool_zoomout.setToolTip("Zoom Out")
        self.btn_tool_zoomout.clicked.connect(lambda: self.canvas.zoom_by(0.8))

        self.btn_tool_grid = TransparentToolButton(FluentIcon.ALIGNMENT, self.toolbar_card)
        self.btn_tool_grid.setToolTip("Toggle Grid Snapping")
        self.btn_tool_grid.clicked.connect(self._on_toggle_grid_snap)

        self.toolbar_layout.addWidget(self.btn_tool_open)
        self.toolbar_layout.addWidget(self.btn_tool_save)
        self.toolbar_layout.addWidget(self.btn_tool_save_as)
        self.toolbar_layout.addWidget(self.btn_tool_markers)
        self.toolbar_layout.addWidget(self.btn_tool_fit)
        self.toolbar_layout.addWidget(self.btn_tool_zoomin)
        self.toolbar_layout.addWidget(self.btn_tool_zoomout)
        self.toolbar_layout.addWidget(self.btn_tool_grid)


        self.toolbar_card.setFixedHeight(38)
        self.toolbar_card.adjustSize()

        # Set initial toggle button active styles
        self._update_toggle_button_style(self.btn_tool_markers, True)
        self._update_toggle_button_style(self.btn_tool_grid, self.viewmodel.config.grid_snap_enabled)

        # Expand Side Panel Floating Button (anchored at top-right of canvas)
        self.btn_expand_panel = TransparentToolButton(FluentIcon.LEFT_ARROW, self.canvas)
        self.btn_expand_panel.setToolTip("Expand Side Panel")
        self.btn_expand_panel.setFixedSize(36, 36)
        self.btn_expand_panel.setStyleSheet(
            "TransparentToolButton { background-color: #282830; border: 1px solid #3e3e4a; border-radius: 6px; }"
        )
        self.btn_expand_panel.hide()

        # Side Control Panel
        self.side_panel = SidePanel(self.viewmodel.config, self)
        self.side_panel.setFixedWidth(310)
        layout.addWidget(self.side_panel, stretch=1)

    def _setup_animation(self) -> None:
        self.anim_side_panel = QPropertyAnimation(self.side_panel, b"maximumWidth", self)
        self.anim_side_panel.setDuration(250)
        self.anim_side_panel.setEasingCurve(QEasingCurve.OutCubic)
        self.anim_side_panel.valueChanged.connect(self._on_animation_step)
        self.anim_side_panel.finished.connect(self._on_animation_finished)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._update_floating_overlay_positions()

    def _update_floating_overlay_positions(self) -> None:
        if hasattr(self, "btn_expand_panel") and hasattr(self, "canvas"):
            btn_w = self.btn_expand_panel.width()
            self.btn_expand_panel.move(max(0, self.canvas.width() - btn_w - 12), 12)
            self.btn_expand_panel.raise_()

        if hasattr(self, "toolbar_card") and hasattr(self, "canvas"):
            tb_w = self.toolbar_card.width()
            pos_x = max(10, int((self.canvas.width() - tb_w) / 2))
            self.toolbar_card.move(pos_x, 12)
            self.toolbar_card.raise_()

    def _update_toggle_button_style(self, button: TransparentToolButton, is_active: bool) -> None:
        if is_active:
            button.setStyleSheet(
                "TransparentToolButton { background-color: rgba(255, 255, 255, 0.14); border: 1px solid rgba(255, 255, 255, 0.25); border-radius: 6px; padding: 4px; }"
            )
        else:
            button.setStyleSheet(
                "TransparentToolButton { background-color: transparent; border: 1px solid transparent; border-radius: 6px; padding: 4px; }"
            )

    def _on_open_layout(self) -> None:
        if hasattr(self.window(), "_on_action_open"):
            self.window()._on_action_open()
        else:
            from PySide6.QtWidgets import QFileDialog
            filepath, _ = QFileDialog.getOpenFileName(
                self, "Open Layout", "", "XML Files (*.xml)"
            )
            if filepath:
                self.viewmodel.load_project_xml(filepath)

    def _on_save_layout(self) -> None:
        if hasattr(self.window(), "_on_action_save"):
            self.window()._on_action_save()
        else:
            if self.viewmodel.current_filepath:
                self.viewmodel.save_project_xml(self.viewmodel.current_filepath)
            else:
                self._on_save_layout_as()

    def _on_save_layout_as(self) -> None:
        if hasattr(self.window(), "_on_action_save_as"):
            self.window()._on_action_save_as()
        else:
            import re
            from PySide6.QtWidgets import QFileDialog
            proj_name = self.viewmodel.layout.project_name if self.viewmodel.layout else "My Paper Keyboard"
            clean_name = re.sub(r'[^\w\-]+', '_', proj_name.strip()).strip('_').lower()
            default_name = f"{clean_name if clean_name else 'paper_layout'}.xml"
            filepath, _ = QFileDialog.getSaveFileName(
                self, "Save Layout As", default_name, "XML Files (*.xml)"
            )
            if filepath:
                self.viewmodel.save_project_xml(filepath)


    def _on_toggle_outer_markers_view(self) -> None:
        visible = self.canvas.toggle_view_outer_markers()
        self._update_toggle_button_style(self.btn_tool_markers, visible)

    def _on_toggle_grid_snap(self) -> None:
        self.viewmodel.config.grid_snap_enabled = not self.viewmodel.config.grid_snap_enabled
        self.canvas.viewport().update()
        self._update_toggle_button_style(self.btn_tool_grid, self.viewmodel.config.grid_snap_enabled)

    def toggle_side_panel(self) -> None:
        """Toggle side panel visibility with smooth sliding transition."""
        if self.anim_side_panel.state() == QPropertyAnimation.Running:
            self.anim_side_panel.stop()

        current_width = self.side_panel.width()

        if self._is_panel_expanded:
            self._is_panel_expanded = False
            self.side_panel.setMinimumWidth(0)
            self.anim_side_panel.setStartValue(current_width)
            self.anim_side_panel.setEndValue(0)
            self.btn_expand_panel.show()
            self._update_floating_overlay_positions()
            self.anim_side_panel.start()
        else:
            self._is_panel_expanded = True
            self.btn_expand_panel.hide()
            self.side_panel.show()
            self.anim_side_panel.setStartValue(current_width)
            self.anim_side_panel.setEndValue(310)
            self.anim_side_panel.start()

    def _on_animation_step(self) -> None:
        self._update_floating_overlay_positions()
        self.canvas.fit_layout_to_view()

    def _on_animation_finished(self) -> None:
        if self._is_panel_expanded:
            self.side_panel.setFixedWidth(310)
        else:
            self.side_panel.hide()
        self.canvas.fit_layout_to_view()
        self._update_floating_overlay_positions()

    def _on_remove_selected_element(self) -> None:
        self.canvas.remove_selected_element()
        self.viewmodel.remove_selected_element()

    def _bind_viewmodel(self) -> None:
        """Bind View UI events to ViewModel and subscribe to ViewModel signals."""
        # 1. View User Interactions -> ViewModel Actions
        self.side_panel.add_button_requested.connect(self.viewmodel.add_button)
        self.side_panel.add_marker_requested.connect(self.viewmodel.add_custom_marker)
        self.side_panel.duplicate_button_requested.connect(self.viewmodel.duplicate_selected_button)
        self.side_panel.remove_button_requested.connect(self._on_remove_selected_element)
        self.side_panel.clear_all_requested.connect(self.viewmodel.create_new_layout)
        self.side_panel.fit_view_requested.connect(self.canvas.fit_layout_to_view)
        self.side_panel.collapse_requested.connect(self.toggle_side_panel)
        self.btn_expand_panel.clicked.connect(self.toggle_side_panel)

        self.side_panel.button_updated.connect(self.viewmodel.update_button_properties)
        self.side_panel.marker_updated.connect(self.viewmodel.update_marker_properties)
        self.side_panel.project_name_updated.connect(self.viewmodel.update_project_name)

        self.canvas.selection_changed.connect(self.viewmodel.select_button)
        self.canvas.marker_selection_changed.connect(self.viewmodel.select_marker)
        self.canvas.layout_updated.connect(self._on_canvas_layout_updated)

        # 2. ViewModel Signals -> View Updates
        self.viewmodel.button_added.connect(self.canvas.add_button)
        self.viewmodel.marker_added.connect(self.canvas.add_user_marker)
        self.viewmodel.button_removed.connect(self.canvas.remove_button_by_id)
        self.viewmodel.marker_removed.connect(self.canvas.remove_marker_by_id)
        self.viewmodel.button_updated.connect(self._on_button_updated_from_vm)
        self.viewmodel.marker_updated.connect(self._on_marker_updated_from_vm)

        self.viewmodel.selection_changed.connect(self.side_panel.set_selected_button)
        self.viewmodel.marker_selection_changed.connect(self.side_panel.set_selected_marker)

        self.viewmodel.stats_changed.connect(self.side_panel.update_stats)
        self.viewmodel.layout_changed.connect(self._on_layout_changed_from_vm)

    def _on_layout_changed_from_vm(self, layout) -> None:
        self.canvas.load_layout(layout)
        if hasattr(layout, "project_name"):
            self.side_panel.set_project_name(layout.project_name)

    def _on_canvas_layout_updated(self, layout) -> None:
        valid = self.canvas.validate_layout()
        custom_marker_count = len(layout.custom_markers) if layout.use_custom_markers else 0
        self.side_panel.update_stats(len(layout.buttons), custom_marker_count, valid)
        self.viewmodel.mark_dirty()


    def _on_button_updated_from_vm(self, button: ButtonModel) -> None:
        target_item = None
        for item in self.canvas.button_items.values():
            if item.button is button:
                target_item = item
                break
        if target_item is None and button.id in self.canvas.button_items:
            target_item = self.canvas.button_items[button.id]

        if target_item is not None:
            target_item.prepareGeometryChange()
            target_item.setPos(button.x_mm, button.y_mm)
            target_item.update()
            self.canvas.button_items = {item.button.id: item for item in self.canvas.button_items.values()}
            self.canvas.validate_layout()

    def _on_marker_updated_from_vm(self, marker: MarkerModel) -> None:
        target_item = None
        for item in self.canvas.marker_items.values():
            if item.marker is marker:
                target_item = item
                break
        if target_item is None and marker.id in self.canvas.marker_items:
            target_item = self.canvas.marker_items[marker.id]

        if target_item is not None:
            target_item.set_tag_id(marker.id)
            target_item.setPos(marker.x_mm, marker.y_mm)
            target_item.update()
            self.canvas.marker_items = {item.marker.id: item for item in self.canvas.marker_items.values()}
            self.canvas.validate_layout()

