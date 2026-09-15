"""
Fluent Side Control Panel.
Built with PySide6-Fluent-Widgets cards and controls for button and marker property editing,
alignment tools, quick creation, and layout statistics.
"""

from PySide6.QtCore import Signal, Qt
from PySide6.QtWidgets import QFormLayout, QGridLayout, QHBoxLayout, QVBoxLayout, QWidget
from qfluentwidgets import (
    BodyLabel,
    CaptionLabel,
    CardWidget,
    DoubleSpinBox,
    FluentIcon,
    LineEdit,
    PrimaryPushButton,
    PushButton,
    SingleDirectionScrollArea,
    SpinBox,
    StrongBodyLabel,
    TransparentToolButton,
)

from core.models.button_model import ButtonModel
from core.models.marker_model import MarkerModel
from config.app_config import AppConfig



class SidePanel(QWidget):
    add_button_requested = Signal()
    add_marker_requested = Signal()
    duplicate_button_requested = Signal()
    remove_button_requested = Signal()
    remove_marker_requested = Signal()
    clear_all_requested = Signal()
    fit_view_requested = Signal()
    collapse_requested = Signal()
    button_updated = Signal(object)  # ButtonModel
    marker_updated = Signal(object)  # MarkerModel
    project_name_updated = Signal(str)

    def __init__(self, config: AppConfig, parent: QWidget | None = None):
        super().__init__(parent)
        self.config = config
        self._current_button: ButtonModel | None = None
        self._current_marker: MarkerModel | None = None
        self._is_updating = False

        self._setup_ui()
        self.setFixedWidth(310)

    def _setup_ui(self) -> None:
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # Scroll area to prevent vertical clutter and overcrowding
        scroll_area = SingleDirectionScrollArea(self, orient=Qt.Vertical)
        scroll_area.setWidgetResizable(True)

        scroll_area.setStyleSheet("QScrollArea { border: none; background: transparent; }")

        container = QWidget()
        container.setStyleSheet("QWidget { background: transparent; }")
        layout = QVBoxLayout(container)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)

        # 1. Quick Actions Card
        actions_card = CardWidget(container)
        actions_layout = QVBoxLayout(actions_card)
        actions_layout.setContentsMargins(12, 10, 12, 10)
        actions_layout.setSpacing(8)

        header_layout = QHBoxLayout()
        header_layout.setContentsMargins(0, 0, 0, 0)
        actions_title = StrongBodyLabel("Placement Actions", actions_card)
        actions_title.setStyleSheet("color: #009FEF; font-size: 13px; font-weight: bold;")
        self.btn_collapse = TransparentToolButton(FluentIcon.CHEVRON_RIGHT_MED, actions_card)
        self.btn_collapse.setToolTip("Collapse Side Panel")
        header_layout.addWidget(actions_title)
        header_layout.addStretch(1)
        header_layout.addWidget(self.btn_collapse)
        actions_layout.addLayout(header_layout)

        # Equal 2-column grid layout for actions buttons so all buttons align perfectly
        btn_grid = QGridLayout()
        btn_grid.setContentsMargins(0, 0, 0, 0)
        btn_grid.setSpacing(6)
        btn_grid.setColumnStretch(0, 1)
        btn_grid.setColumnStretch(1, 1)

        self.btn_add = PrimaryPushButton(FluentIcon.ADD, "Add Key", actions_card)
        self.btn_add_marker = PushButton(FluentIcon.TAG, "Add Marker", actions_card)
        self.btn_duplicate = PushButton(FluentIcon.COPY, "Duplicate", actions_card)
        self.btn_remove = PushButton(FluentIcon.DELETE, "Delete Selected", actions_card)
        self.btn_fit_view = PushButton(FluentIcon.ZOOM_IN, "Fit View", actions_card)
        self.btn_clear = PushButton(FluentIcon.CANCEL, "Clear All", actions_card)

        for btn in (self.btn_add, self.btn_add_marker, self.btn_duplicate, self.btn_remove, self.btn_fit_view, self.btn_clear):
            btn.setFixedHeight(32)

        btn_grid.addWidget(self.btn_add, 0, 0)
        btn_grid.addWidget(self.btn_add_marker, 0, 1)
        btn_grid.addWidget(self.btn_duplicate, 1, 0)
        btn_grid.addWidget(self.btn_remove, 1, 1)
        btn_grid.addWidget(self.btn_fit_view, 2, 0)
        btn_grid.addWidget(self.btn_clear, 2, 1)

        actions_layout.addLayout(btn_grid)
        layout.addWidget(actions_card)

        # 2. Selected Key Properties Card
        props_card = CardWidget(container)
        props_layout = QVBoxLayout(props_card)
        props_layout.setContentsMargins(12, 10, 12, 10)
        props_layout.setSpacing(8)

        props_title = StrongBodyLabel("Key Properties", props_card)
        props_title.setStyleSheet("color: #009FEF; font-size: 13px; font-weight: bold;")
        props_layout.addWidget(props_title)

        form_layout = QFormLayout()
        form_layout.setContentsMargins(0, 0, 0, 0)
        form_layout.setSpacing(6)

        self.input_id = LineEdit(props_card)
        self.input_text = LineEdit(props_card)
        self.spin_font = SpinBox(props_card)
        self.spin_font.setRange(6, 48)

        self.spin_x = DoubleSpinBox(props_card)
        self.spin_x.setRange(0.0, 1000.0)
        self.spin_x.setSingleStep(1.0)

        self.spin_y = DoubleSpinBox(props_card)
        self.spin_y.setRange(0.0, 1000.0)
        self.spin_y.setSingleStep(1.0)

        self.spin_w = DoubleSpinBox(props_card)
        self.spin_w.setRange(5.0, 500.0)
        self.spin_w.setSingleStep(1.0)

        self.spin_h = DoubleSpinBox(props_card)
        self.spin_h.setRange(5.0, 500.0)
        self.spin_h.setSingleStep(1.0)

        def make_caption(text: str) -> CaptionLabel:
            lbl = CaptionLabel(text, props_card)
            lbl.setStyleSheet("color: #CBD5E1; font-weight: 500;")
            return lbl

        form_layout.addRow(make_caption("Key ID:"), self.input_id)
        form_layout.addRow(make_caption("Label Text:"), self.input_text)
        form_layout.addRow(make_caption("Font Size (pt):"), self.spin_font)
        form_layout.addRow(make_caption("Pos X (mm):"), self.spin_x)
        form_layout.addRow(make_caption("Pos Y (mm):"), self.spin_y)
        form_layout.addRow(make_caption("Width (mm):"), self.spin_w)
        form_layout.addRow(make_caption("Height (mm):"), self.spin_h)

        props_layout.addLayout(form_layout)
        layout.addWidget(props_card)

        # 3. Selected AprilTag Marker Card
        marker_card = CardWidget(container)
        marker_layout = QVBoxLayout(marker_card)
        marker_layout.setContentsMargins(12, 10, 12, 10)
        marker_layout.setSpacing(8)

        marker_title = StrongBodyLabel("AprilTag Marker Properties", marker_card)
        marker_title.setStyleSheet("color: #009FEF; font-size: 13px; font-weight: bold;")
        marker_layout.addWidget(marker_title)

        form_marker = QFormLayout()
        form_marker.setContentsMargins(0, 0, 0, 0)
        form_marker.setSpacing(6)

        self.spin_marker_id = SpinBox(marker_card)
        self.spin_marker_id.setRange(0, 100)

        self.spin_marker_x = DoubleSpinBox(marker_card)
        self.spin_marker_x.setRange(0.0, 1000.0)
        self.spin_marker_x.setSingleStep(1.0)

        self.spin_marker_y = DoubleSpinBox(marker_card)
        self.spin_marker_y.setRange(0.0, 1000.0)
        self.spin_marker_y.setSingleStep(1.0)

        def make_marker_caption(text: str) -> CaptionLabel:
            lbl = CaptionLabel(text, marker_card)
            lbl.setStyleSheet("color: #CBD5E1; font-weight: 500;")
            return lbl

        form_marker.addRow(make_marker_caption("Tag ID:"), self.spin_marker_id)
        form_marker.addRow(make_marker_caption("Pos X (mm):"), self.spin_marker_x)
        form_marker.addRow(make_marker_caption("Pos Y (mm):"), self.spin_marker_y)

        marker_layout.addLayout(form_marker)
        layout.addWidget(marker_card)

        # 4. Layout Overview Card
        stats_card = CardWidget(container)
        stats_layout = QVBoxLayout(stats_card)
        stats_layout.setContentsMargins(12, 10, 12, 10)
        stats_layout.setSpacing(6)

        stats_title = StrongBodyLabel("Layout Overview", stats_card)
        stats_title.setStyleSheet("color: #009FEF; font-size: 13px; font-weight: bold;")
        stats_layout.addWidget(stats_title)

        lbl_proj_caption = CaptionLabel("Project Name:", stats_card)
        lbl_proj_caption.setStyleSheet("color: #CBD5E1; font-weight: 500;")
        self.input_project_name = LineEdit(stats_card)
        self.input_project_name.setPlaceholderText("Enter project name...")
        self.input_project_name.setText("My Paper Keyboard")

        form_proj = QFormLayout()
        form_proj.setContentsMargins(0, 2, 0, 4)
        form_proj.setSpacing(6)
        form_proj.addRow(lbl_proj_caption, self.input_project_name)
        stats_layout.addLayout(form_proj)

        self.lbl_stats_count = BodyLabel("Total Keys: 0", stats_card)
        self.lbl_stats_markers = BodyLabel("Total Custom Markers: 0", stats_card)
        self.lbl_stats_status = BodyLabel("Status: Valid", stats_card)

        self.lbl_stats_count.setStyleSheet("color: #F8FAFC;")
        self.lbl_stats_markers.setStyleSheet("color: #F8FAFC;")
        self.lbl_stats_status.setStyleSheet("color: #4ADE80; font-weight: bold;")

        self.lbl_stats_count.setWordWrap(True)
        self.lbl_stats_markers.setWordWrap(True)
        self.lbl_stats_status.setWordWrap(True)

        stats_layout.addWidget(self.lbl_stats_count)
        stats_layout.addWidget(self.lbl_stats_markers)
        stats_layout.addWidget(self.lbl_stats_status)

        layout.addWidget(stats_card)
        layout.addStretch(1)

        scroll_area.setWidget(container)
        main_layout.addWidget(scroll_area)

        # Connect Action Signals
        self.btn_add.clicked.connect(self.add_button_requested)
        self.btn_add_marker.clicked.connect(self.add_marker_requested)
        self.btn_duplicate.clicked.connect(self.duplicate_button_requested)
        self.btn_remove.clicked.connect(self.remove_button_requested)
        self.btn_clear.clicked.connect(self.clear_all_requested)
        self.btn_fit_view.clicked.connect(self.fit_view_requested)
        self.btn_collapse.clicked.connect(self.collapse_requested)

        self.input_project_name.textChanged.connect(self._on_project_name_input_changed)

        self.input_id.textChanged.connect(self._on_property_changed)
        self.input_text.textChanged.connect(self._on_property_changed)
        self.spin_font.valueChanged.connect(self._on_property_changed)
        self.spin_x.valueChanged.connect(self._on_property_changed)
        self.spin_y.valueChanged.connect(self._on_property_changed)
        self.spin_w.valueChanged.connect(self._on_property_changed)
        self.spin_h.valueChanged.connect(self._on_property_changed)

        self.spin_marker_id.valueChanged.connect(self._on_marker_property_changed)
        self.spin_marker_x.valueChanged.connect(self._on_marker_property_changed)
        self.spin_marker_y.valueChanged.connect(self._on_marker_property_changed)

        self.set_selected_button(None)
        self.set_selected_marker(None)


    def _update_delete_button_state(self) -> None:
        has_selection = (self._current_button is not None) or (self._current_marker is not None)
        self.btn_remove.setEnabled(has_selection)

    def set_selected_button(self, button: ButtonModel | None) -> None:
        self._current_button = button
        self._is_updating = True

        if button is None:
            self.input_id.setText("")
            self.input_text.setText("")
            self.spin_font.setValue(self.config.default_font_size_pt)
            self.spin_x.setValue(0.0)
            self.spin_y.setValue(0.0)
            self.spin_w.setValue(self.config.button_min_width_mm)
            self.spin_h.setValue(self.config.button_min_height_mm)

            self.input_id.setEnabled(False)
            self.input_text.setEnabled(False)
            self.spin_font.setEnabled(False)
            self.spin_x.setEnabled(False)
            self.spin_y.setEnabled(False)
            self.spin_w.setEnabled(False)
            self.spin_h.setEnabled(False)
            self.btn_duplicate.setEnabled(False)
        else:
            self.input_id.setEnabled(True)
            self.input_text.setEnabled(True)
            self.spin_font.setEnabled(True)
            self.spin_x.setEnabled(True)
            self.spin_y.setEnabled(True)
            self.spin_w.setEnabled(True)
            self.spin_h.setEnabled(True)
            self.btn_duplicate.setEnabled(True)

            self.input_id.setText(button.id)
            self.input_text.setText(button.text)
            self.spin_font.setValue(button.font_size_pt)
            self.spin_x.setValue(button.x_mm)
            self.spin_y.setValue(button.y_mm)
            self.spin_w.setValue(button.width_mm)
            self.spin_h.setValue(button.height_mm)

        self._update_delete_button_state()
        self._is_updating = False

    def set_selected_marker(self, marker: MarkerModel | None) -> None:
        self._current_marker = marker
        self._is_updating = True

        if marker is None:
            self.spin_marker_id.setValue(0)
            self.spin_marker_x.setValue(0.0)
            self.spin_marker_y.setValue(0.0)

            self.spin_marker_id.setEnabled(False)
            self.spin_marker_x.setEnabled(False)
            self.spin_marker_y.setEnabled(False)
        else:
            self.spin_marker_id.setEnabled(True)
            self.spin_marker_x.setEnabled(True)
            self.spin_marker_y.setEnabled(True)

            self.spin_marker_id.setValue(marker.id)
            self.spin_marker_x.setValue(marker.x_mm)
            self.spin_marker_y.setValue(marker.y_mm)

        self._update_delete_button_state()
        self._is_updating = False


    def update_stats(self, key_count: int, marker_count: int, is_valid: bool) -> None:
        self.lbl_stats_count.setText(f"Total Keys: {key_count}")
        self.lbl_stats_markers.setText(f"Total Custom Markers: {marker_count}")
        status_str = "Valid" if is_valid else "Invalid Bounds / Overlap"
        self.lbl_stats_status.setText(f"Status: {status_str}")

    def set_project_name(self, project_name: str) -> None:
        self._is_updating = True
        self.input_project_name.setText(project_name)
        self._is_updating = False

    def _on_project_name_input_changed(self, text: str) -> None:
        if not self._is_updating:
            self.project_name_updated.emit(text)

    def _on_property_changed(self) -> None:
        if self._is_updating or self._current_button is None:
            return

        self._current_button.id = self.input_id.text().strip()
        self._current_button.text = self.input_text.text()
        self._current_button.font_size_pt = self.spin_font.value()
        self._current_button.x_mm = self.spin_x.value()
        self._current_button.y_mm = self.spin_y.value()
        self._current_button.width_mm = self.spin_w.value()
        self._current_button.height_mm = self.spin_h.value()

        self.button_updated.emit(self._current_button)

    def _on_marker_property_changed(self) -> None:
        if self._is_updating or self._current_marker is None:
            return

        self._current_marker.id = self.spin_marker_id.value()
        self._current_marker.x_mm = self.spin_marker_x.value()
        self._current_marker.y_mm = self.spin_marker_y.value()

        self.marker_updated.emit(self._current_marker)
