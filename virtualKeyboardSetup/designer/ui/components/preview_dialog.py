"""
Print Outcome Preview Dialog.
Displays a rasterized PIL image preview of the physical A4 printable paper.
"""

from PIL import Image
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QDialog, QHBoxLayout, QLabel, QScrollArea, QVBoxLayout, QWidget
from qfluentwidgets import PushButton, SubtitleLabel

from core.models.paper_layout import PaperLayoutModel
from config.app_config import AppConfig
from services.preview_service import PreviewService


class PreviewDialog(QDialog):
    def __init__(self, layout: PaperLayoutModel, config: AppConfig, parent: QWidget | None = None):
        super().__init__(parent)
        self.setWindowTitle("Printable Paper Outcome Preview (A4 Landscape)")
        self.resize(1000, 720)

        main_layout = QVBoxLayout(self)

        title = SubtitleLabel("Print Outcome Preview (A4 Landscape with AprilTag Markers)")
        title.setStyleSheet("color: #009FEF; font-size: 16px; font-weight: bold;")
        main_layout.addWidget(title)

        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        self.img_label = QLabel(scroll)
        self.img_label.setAlignment(Qt.AlignCenter)
        scroll.setWidget(self.img_label)
        main_layout.addWidget(scroll)

        btn_box = QHBoxLayout()
        btn_close = PushButton("Close", self)
        btn_close.clicked.connect(self.accept)
        btn_box.addStretch(1)
        btn_box.addWidget(btn_close)
        main_layout.addLayout(btn_box)

        self._render_preview_image(layout, config)

    def _render_preview_image(self, layout: PaperLayoutModel, config: AppConfig) -> None:
        pil_img = PreviewService.render_preview(layout, config, scale=3.0)
        pil_img = pil_img.convert("RGBA")
        data = pil_img.tobytes("raw", "RGBA")

        qimg = QImage(data, pil_img.width, pil_img.height, QImage.Format_RGBA8888)
        pixmap = QPixmap.fromImage(qimg)
        self.img_label.setPixmap(pixmap)
