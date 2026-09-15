"""
Settings View Component (MVVM View Layer).
Application settings presentation layer for customizable paper surface dimensions,
button margins, stroke widths, marker settings, and grid options.
"""

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QHBoxLayout, QSizePolicy, QVBoxLayout, QWidget
from qfluentwidgets import (
    BodyLabel,
    CaptionLabel,
    CardWidget,
    ComboBox,
    DoubleSpinBox,
    FluentIcon,
    PrimaryPushButton,
    SingleDirectionScrollArea,
    StrongBodyLabel,
    SubtitleLabel,
    SwitchButton,
)

from viewmodels.settings_viewmodel import SettingsViewModel


class SettingsView(QWidget):
    def __init__(self, viewmodel: SettingsViewModel, parent: QWidget | None = None):
        super().__init__(parent)
        self.setObjectName("settingsView")
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.viewmodel = viewmodel
        self._setup_ui()
        self._bind_viewmodel()

    def _setup_ui(self) -> None:
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # Main Scroll Area for Settings View
        scroll_area = SingleDirectionScrollArea(self, orient=Qt.Vertical)
        scroll_area.setWidgetResizable(True)
        scroll_area.setStyleSheet("QScrollArea { border: none; background: transparent; }")

        container = QWidget()
        container.setStyleSheet("QWidget { background: transparent; }")
        layout = QVBoxLayout(container)
        layout.setContentsMargins(28, 24, 28, 24)
        layout.setSpacing(20)

        # Header Title
        title_layout = QVBoxLayout()
        title_layout.setSpacing(4)
        title = SubtitleLabel("Flexible Paper & System Settings", container)
        title.setStyleSheet("color: #009FEF; font-size: 20px; font-weight: bold;")
        subtitle = CaptionLabel(
            "Configure paper surface dimensions, AprilTag markers, button spacing/stroke geometry, grid snap, and database options.",
            container,
        )
        subtitle.setStyleSheet("color: #94A3B8; font-size: 13px;")
        title_layout.addWidget(title)
        title_layout.addWidget(subtitle)
        layout.addLayout(title_layout)

        INPUT_WIDTH = 240

        # Helper method to create aligned setting rows inside cards
        def create_setting_row(
            title_text: str, desc_text: str | None, widget: QWidget, card: CardWidget
        ) -> QWidget:
            row_widget = QWidget(card)
            row_layout = QHBoxLayout(row_widget)
            row_layout.setContentsMargins(16, 10, 16, 10)
            row_layout.setSpacing(16)

            text_layout = QVBoxLayout()
            text_layout.setSpacing(2)
            lbl_title = BodyLabel(title_text, row_widget)
            lbl_title.setStyleSheet("color: #F8FAFC; font-weight: 600;")
            text_layout.addWidget(lbl_title)
            if desc_text:
                lbl_desc = CaptionLabel(desc_text, row_widget)
                lbl_desc.setStyleSheet("color: #94A3B8;")
                text_layout.addWidget(lbl_desc)

            row_layout.addLayout(text_layout)
            row_layout.addStretch(1)

            if not isinstance(widget, SwitchButton) and not isinstance(widget, BodyLabel):
                widget.setFixedWidth(INPUT_WIDTH)
            row_layout.addWidget(widget, 0, Qt.AlignRight | Qt.AlignVCenter)

            return row_widget

        # 1. User Configurable Paper Dimensions Card
        paper_card = CardWidget(container)
        paper_layout = QVBoxLayout(paper_card)
        paper_layout.setContentsMargins(0, 12, 0, 12)
        paper_layout.setSpacing(0)

        paper_title = StrongBodyLabel("1. Paper Surface Dimensions & Margins", paper_card)
        paper_title.setStyleSheet("color: #009FEF; font-size: 14px; font-weight: bold; padding: 0 16px 8px 16px;")
        paper_layout.addWidget(paper_title)

        self.combo_presets = ComboBox(paper_card)
        self.combo_presets.addItems([preset[0] for preset in self.viewmodel.PRESETS])

        self.spin_width = DoubleSpinBox(paper_card)
        self.spin_width.setRange(50.0, 1000.0)
        self.spin_width.setSingleStep(5.0)
        self.spin_width.setValue(self.viewmodel.paper_width_mm)

        self.spin_height = DoubleSpinBox(paper_card)
        self.spin_height.setRange(50.0, 1000.0)
        self.spin_height.setSingleStep(5.0)
        self.spin_height.setValue(self.viewmodel.paper_height_mm)

        self.spin_paper_margin = DoubleSpinBox(paper_card)
        self.spin_paper_margin.setRange(0.0, 50.0)
        self.spin_paper_margin.setSingleStep(1.0)
        self.spin_paper_margin.setValue(self.viewmodel.paper_margin_mm)

        paper_layout.addWidget(create_setting_row("Paper Format Preset", "Select standard paper sizes (A4, A3, Letter) or Custom", self.combo_presets, paper_card))
        paper_layout.addWidget(create_setting_row("Paper Width (mm)", "Width of printable paper sheet surface in millimeters", self.spin_width, paper_card))
        paper_layout.addWidget(create_setting_row("Paper Height (mm)", "Height of printable paper sheet surface in millimeters", self.spin_height, paper_card))
        paper_layout.addWidget(create_setting_row("Paper Border Margin (mm)", "Outer border margin around paper edge", self.spin_paper_margin, paper_card))

        layout.addWidget(paper_card)

        # 2. AprilTag Fiducial Markers Card
        marker_card = CardWidget(container)
        marker_layout = QVBoxLayout(marker_card)
        marker_layout.setContentsMargins(0, 12, 0, 12)
        marker_layout.setSpacing(0)

        marker_title = StrongBodyLabel("2. AprilTag Fiducial Marker Preferences", marker_card)
        marker_title.setStyleSheet("color: #009FEF; font-size: 14px; font-weight: bold; padding: 0 16px 8px 16px;")
        marker_layout.addWidget(marker_title)

        self.spin_marker_size = DoubleSpinBox(marker_card)
        self.spin_marker_size.setRange(5.0, 100.0)
        self.spin_marker_size.setSingleStep(1.0)
        self.spin_marker_size.setValue(self.viewmodel.marker_size_mm)

        self.spin_marker_gap = DoubleSpinBox(marker_card)
        self.spin_marker_gap.setRange(0.0, 50.0)
        self.spin_marker_gap.setSingleStep(1.0)
        self.spin_marker_gap.setValue(self.viewmodel.marker_min_gap_mm)

        self.switch_outer_markers = SwitchButton(marker_card)
        self.switch_outer_markers.setChecked(self.viewmodel.show_outer_markers)

        marker_layout.addWidget(create_setting_row("AprilTag Marker Size (mm)", "Dimension of AprilTag fiducial square anchors", self.spin_marker_size, marker_card))
        marker_layout.addWidget(create_setting_row("Minimum Marker Spacing / Gap (mm)", "Required clearance gap between markers and elements", self.spin_marker_gap, marker_card))
        marker_layout.addWidget(create_setting_row("Show Outer Perimeter Markers", "Display default corner AprilTag fiducial anchors on canvas", self.switch_outer_markers, marker_card))

        layout.addWidget(marker_card)

        # 3. Key Button Spacing & Geometry Style Card
        button_card = CardWidget(container)
        button_layout = QVBoxLayout(button_card)
        button_layout.setContentsMargins(0, 12, 0, 12)
        button_layout.setSpacing(0)

        button_title = StrongBodyLabel("3. Key Button Spacing & Style Options", button_card)
        button_title.setStyleSheet("color: #009FEF; font-size: 14px; font-weight: bold; padding: 0 16px 8px 16px;")
        button_layout.addWidget(button_title)

        self.spin_button_gap = DoubleSpinBox(button_card)
        self.spin_button_gap.setRange(0.0, 50.0)
        self.spin_button_gap.setSingleStep(1.0)
        self.spin_button_gap.setValue(self.viewmodel.button_min_gap_mm)

        self.spin_stroke_width = DoubleSpinBox(button_card)
        self.spin_stroke_width.setRange(0.2, 5.0)
        self.spin_stroke_width.setSingleStep(0.1)
        self.spin_stroke_width.setValue(self.viewmodel.button_stroke_width_mm)

        self.spin_corner_radius = DoubleSpinBox(button_card)
        self.spin_corner_radius.setRange(0.0, 20.0)
        self.spin_corner_radius.setSingleStep(0.5)
        self.spin_corner_radius.setValue(self.viewmodel.button_corner_radius_mm)

        button_layout.addWidget(create_setting_row("Minimum Button Spacing / Gap (mm)", "Required minimum clearance gap between key buttons", self.spin_button_gap, button_card))
        button_layout.addWidget(create_setting_row("Button Border / Stroke Width (mm)", "Thickness of key button outlines on canvas and exports", self.spin_stroke_width, button_card))
        button_layout.addWidget(create_setting_row("Button Corner Radius (mm)", "Curvature radius of key button corners", self.spin_corner_radius, button_card))

        layout.addWidget(button_card)

        # 4. Grid Snapping & Application Settings Card
        grid_card = CardWidget(container)
        grid_layout = QVBoxLayout(grid_card)
        grid_layout.setContentsMargins(0, 12, 0, 12)
        grid_layout.setSpacing(0)

        grid_title = StrongBodyLabel("4. Grid Snapping & Database Options", grid_card)
        grid_title.setStyleSheet("color: #009FEF; font-size: 14px; font-weight: bold; padding: 0 16px 8px 16px;")
        grid_layout.addWidget(grid_title)

        self.switch_grid = SwitchButton(grid_card)
        self.switch_grid.setChecked(self.viewmodel.grid_snap_enabled)

        self.spin_grid_size = DoubleSpinBox(grid_card)
        self.spin_grid_size.setRange(1.0, 50.0)
        self.spin_grid_size.setValue(self.viewmodel.grid_size_mm)

        self.lbl_db_path = BodyLabel(self.viewmodel.db_path, grid_card)
        self.lbl_db_path.setStyleSheet("color: #CBD5E1;")

        grid_layout.addWidget(create_setting_row("Grid Snapping Enabled", "Snap items to layout grid coordinates during drag", self.switch_grid, grid_card))
        grid_layout.addWidget(create_setting_row("Grid Step Size (mm)", "Grid cell interval spacing in millimeters", self.spin_grid_size, grid_card))
        grid_layout.addWidget(create_setting_row("SQLite Database Location", "File storage path for saved paper layout designs", self.lbl_db_path, grid_card))

        layout.addWidget(grid_card)

        # Save Button Bar
        btn_layout = QHBoxLayout()
        btn_layout.setContentsMargins(0, 6, 0, 0)
        btn_apply = PrimaryPushButton(FluentIcon.SAVE, "Apply Settings", container)
        btn_apply.setFixedHeight(36)
        btn_apply.setFixedWidth(160)
        btn_apply.clicked.connect(self._save_settings)
        btn_layout.addWidget(btn_apply)
        btn_layout.addStretch(1)

        layout.addLayout(btn_layout)
        layout.addStretch(1)

        scroll_area.setWidget(container)
        main_layout.addWidget(scroll_area)

        self.combo_presets.currentIndexChanged.connect(self._on_preset_changed)

    def _bind_viewmodel(self) -> None:
        """Bind ViewModel property signals if needed."""
        pass

    def _on_preset_changed(self, index: int) -> None:
        dims = self.viewmodel.get_preset_dimensions(index)
        if dims is not None:
            w, h = dims
            self.spin_width.setValue(w)
            self.spin_height.setValue(h)

    def sync_from_config(self) -> None:
        """Sync UI inputs with ViewModel property values."""
        self.spin_width.setValue(self.viewmodel.paper_width_mm)
        self.spin_height.setValue(self.viewmodel.paper_height_mm)
        self.spin_paper_margin.setValue(self.viewmodel.paper_margin_mm)
        self.spin_marker_size.setValue(self.viewmodel.marker_size_mm)
        self.spin_marker_gap.setValue(self.viewmodel.marker_min_gap_mm)
        self.switch_outer_markers.setChecked(self.viewmodel.show_outer_markers)
        self.switch_grid.setChecked(self.viewmodel.grid_snap_enabled)
        self.spin_grid_size.setValue(self.viewmodel.grid_size_mm)
        self.spin_button_gap.setValue(self.viewmodel.button_min_gap_mm)
        self.spin_stroke_width.setValue(self.viewmodel.button_stroke_width_mm)
        self.spin_corner_radius.setValue(self.viewmodel.button_corner_radius_mm)

    def _save_settings(self) -> None:
        w = self.spin_width.value()
        h = self.spin_height.value()
        m_size = self.spin_marker_size.value()
        marker_gap = self.spin_marker_gap.value()
        show_outer = self.switch_outer_markers.isChecked()
        grid_snap = self.switch_grid.isChecked()
        grid_size = self.spin_grid_size.value()
        paper_margin = self.spin_paper_margin.value()
        button_gap = self.spin_button_gap.value()
        corner_radius = self.spin_corner_radius.value()
        stroke_width = self.spin_stroke_width.value()

        self.viewmodel.apply_settings(
            w, h, m_size, show_outer, grid_snap, grid_size, paper_margin, button_gap, marker_gap, corner_radius, stroke_width
        )



