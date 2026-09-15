"""
Full-Screen Dark Splash Overlay Component.
Sleek, modern start screen overlay with compact side-by-side action cards.
"""

from pathlib import Path
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)
from qfluentwidgets import (
    BodyLabel,
    CaptionLabel,
    CardWidget,
    ComboBox,
    FluentIcon,
    LineEdit,
    PrimaryPushButton,
    PushButton,
    SingleDirectionScrollArea,
    StrongBodyLabel,
    SubtitleLabel,
)

from config.app_config import AppConfig
from viewmodels.settings_viewmodel import SettingsViewModel


class SplashOverlayWidget(QWidget):
    """
    Full-screen dark theme splash screen overlay widget.
    Features compact, modern side-by-side start cards for New Project and Open Project.
    """

    create_project_requested = Signal(str, float, float, float)  # name, width, height, margin
    open_project_requested = Signal(str)  # filepath

    PRESETS = SettingsViewModel.PRESETS

    def __init__(self, config: AppConfig, parent: QWidget | None = None):
        super().__init__(parent)
        self.setObjectName("splashOverlayWidget")
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.config = config
        self._setup_ui()

    def _setup_ui(self) -> None:
        # Solid dark background matching FluentWindow (#202020)
        self.setStyleSheet("SplashOverlayWidget#splashOverlayWidget { background-color: #202020; color: #F8FAFC; }")

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        scroll_area = SingleDirectionScrollArea(self, orient=Qt.Vertical)
        scroll_area.setWidgetResizable(True)
        scroll_area.setStyleSheet("QScrollArea { border: none; background-color: #202020; }")

        container = QWidget()
        container.setObjectName("startContainer")
        container.setAttribute(Qt.WA_StyledBackground, True)
        container.setStyleSheet("QWidget#startContainer { background-color: #202020; }")

        # Vertical Centering Outer Layout
        outer_layout = QVBoxLayout(container)
        outer_layout.setContentsMargins(32, 40, 32, 40)
        outer_layout.addStretch(1)

        center_hbox = QHBoxLayout()
        center_hbox.addStretch(1)

        content_box = QWidget(container)
        content_box.setMaximumWidth(820)
        content_layout = QVBoxLayout(content_box)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(24)

        # Hero Header Banner
        header_layout = QVBoxLayout()
        header_layout.setSpacing(6)
        header_layout.setAlignment(Qt.AlignCenter)

        title = SubtitleLabel("Paper Virtual Keyboard Designer", content_box)
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("color: #009FEF; font-size: 24px; font-weight: bold;")

        subtitle = CaptionLabel(
            "Select an action below to start a new paper layout or open an existing design project.",
            content_box,
        )
        subtitle.setAlignment(Qt.AlignCenter)
        subtitle.setStyleSheet("color: #94A3B8; font-size: 13px;")

        header_layout.addWidget(title)
        header_layout.addWidget(subtitle)
        content_layout.addLayout(header_layout)

        CARD_STYLE = """
        CardWidget {
            background-color: #1A1A24;
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 12px;
        }
        CardWidget:hover {
            border: 1px solid rgba(0, 159, 239, 0.4);
            background-color: #1E1E2B;
        }
        """

        # 2-Column Modern Side-by-Side Cards
        cards_layout = QHBoxLayout()
        cards_layout.setSpacing(20)

        # Left Card: New Project
        new_card = CardWidget(content_box)
        new_card.setStyleSheet(CARD_STYLE)
        new_layout = QVBoxLayout(new_card)
        new_layout.setContentsMargins(28, 28, 28, 28)
        new_layout.setSpacing(16)

        lbl_new_title = StrongBodyLabel("New Project", new_card)
        lbl_new_title.setStyleSheet("color: #009FEF; font-size: 16px; font-weight: bold;")
        lbl_new_desc = CaptionLabel(
            "Configure layout project name and select a paper size format.",
            new_card,
        )
        lbl_new_desc.setStyleSheet("color: #94A3B8; font-size: 12px;")
        lbl_new_desc.setWordWrap(True)

        new_layout.addWidget(lbl_new_title)
        new_layout.addWidget(lbl_new_desc)

        lbl_name = BodyLabel("Project Name", new_card)
        lbl_name.setStyleSheet("color: #F8FAFC; font-weight: 600; font-size: 12px;")
        self.input_project_name = LineEdit(new_card)
        self.input_project_name.setText("My Paper Keyboard")
        self.input_project_name.setPlaceholderText("Enter project name...")
        self.input_project_name.setFixedHeight(34)

        lbl_preset = BodyLabel("Paper Format", new_card)
        lbl_preset.setStyleSheet("color: #F8FAFC; font-weight: 600; font-size: 12px;")
        self.combo_presets = ComboBox(new_card)
        self.combo_presets.addItems([p[0] for p in self.PRESETS])
        self.combo_presets.setFixedHeight(34)

        new_layout.addWidget(lbl_name)
        new_layout.addWidget(self.input_project_name)
        new_layout.addWidget(lbl_preset)
        new_layout.addWidget(self.combo_presets)

        self.btn_create = PrimaryPushButton(FluentIcon.ADD, "Create Layout", new_card)
        self.btn_create.setFixedHeight(38)
        self.btn_create.clicked.connect(self._on_create_clicked)
        new_layout.addWidget(self.btn_create)

        cards_layout.addWidget(new_card, 1)

        # Right Card: Open Existing Project
        open_card = CardWidget(content_box)
        open_card.setStyleSheet(CARD_STYLE)
        open_layout = QVBoxLayout(open_card)
        open_layout.setContentsMargins(28, 28, 28, 28)
        open_layout.setSpacing(16)

        lbl_open_title = StrongBodyLabel("Open Existing Project", open_card)
        lbl_open_title.setStyleSheet("color: #009FEF; font-size: 16px; font-weight: bold;")
        lbl_open_desc = CaptionLabel(
            "Browse and load a saved XML layout file to resume editing or inspect coordinates.",
            open_card,
        )
        lbl_open_desc.setStyleSheet("color: #94A3B8; font-size: 12px; line-height: 1.5;")
        lbl_open_desc.setWordWrap(True)

        open_layout.addWidget(lbl_open_title)
        open_layout.addWidget(lbl_open_desc)

        open_layout.addStretch(1)

        self.btn_open_file = PushButton(FluentIcon.FOLDER, "Browse XML File...", open_card)
        self.btn_open_file.setFixedHeight(38)
        self.btn_open_file.clicked.connect(self._on_open_file_clicked)
        open_layout.addWidget(self.btn_open_file)

        cards_layout.addWidget(open_card, 1)

        content_layout.addLayout(cards_layout)

        center_hbox.addWidget(content_box)
        center_hbox.addStretch(1)
        outer_layout.addLayout(center_hbox)
        outer_layout.addStretch(1)

        scroll_area.setWidget(container)
        main_layout.addWidget(scroll_area)

    def _on_create_clicked(self) -> None:
        name_text = self.input_project_name.text().strip()
        project_name = name_text if name_text else "My Paper Keyboard"
        index = self.combo_presets.currentIndex()
        if 0 <= index < len(self.PRESETS):
            _, w, h = self.PRESETS[index]
            width_mm = w if w is not None else self.config.paper_width_mm
            height_mm = h if h is not None else self.config.paper_height_mm
        else:
            width_mm = self.config.paper_width_mm
            height_mm = self.config.paper_height_mm
        margin_mm = self.config.paper_margin_mm
        self.create_project_requested.emit(project_name, width_mm, height_mm, margin_mm)

    def _on_open_file_clicked(self) -> None:
        filepath, _ = QFileDialog.getOpenFileName(
            self, "Open Layout Project XML", "", "XML Files (*.xml)"
        )
        if filepath:
            self.open_project_requested.emit(filepath)
