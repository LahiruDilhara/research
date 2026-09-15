"""
About View Component.
Displays research overview and system description.
"""

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QSizePolicy, QVBoxLayout, QWidget
from qfluentwidgets import BodyLabel, CardWidget, SubtitleLabel, TitleLabel

from config.app_config import AppConfig


class AboutView(QWidget):
    def __init__(self, config: AppConfig, parent: QWidget | None = None):
        super().__init__(parent)
        self.setObjectName("aboutView")
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.config = config
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(30, 30, 30, 30)
        layout.setSpacing(20)

        title = TitleLabel("Paper Virtual Keyboard Layout Designer", self)
        title.setStyleSheet("color: #009FEF; font-size: 22px; font-weight: bold;")
        layout.addWidget(title)

        card = CardWidget(self)
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(20, 20, 20, 20)
        card_layout.setSpacing(12)

        subtitle = SubtitleLabel("Research Context & Partitioned Architecture", card)
        subtitle.setStyleSheet("color: #F8FAFC; font-size: 16px; font-weight: bold;")
        card_layout.addWidget(subtitle)

        desc1 = BodyLabel(
            "This application serves as Application 1 (Layout Designer) in the customizable paper virtual keyboard research suite. "
            "It allows researchers and users to visually compose custom key layouts on standard A4 plain paper embedded with AprilTag perimeter anchors.",
            card,
        )
        desc1.setStyleSheet("color: #CBD5E1; line-height: 1.4;")
        desc1.setWordWrap(True)
        card_layout.addWidget(desc1)

        desc2 = BodyLabel(
            "Key features:\n"
            "• Physical Layout Decoupling: Separates printed paper geometry from digital action semantics.\n"
            "• AprilTag Fiducial Anchor Ring: Perimeter markers are automatically computed and printed to maintain robust homography tracking.\n"
            "• Runtime Engine Integration: Exports XML files consumed by Application 2 (Runtime Touch Engine) and physically accurate printable PDFs.",
            card,
        )
        desc2.setStyleSheet("color: #CBD5E1; line-height: 1.4;")
        desc2.setWordWrap(True)
        card_layout.addWidget(desc2)

        layout.addWidget(card)
        layout.addStretch(1)
