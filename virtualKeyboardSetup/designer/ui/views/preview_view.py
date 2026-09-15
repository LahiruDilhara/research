"""
Print Outcome Preview View Component (MVVM View Layer).
Dedicated left navigation tab displaying real-time rendered paper layout output,
complete with AprilTag anchors, key bounds, export PDF and save project actions.
"""

from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QFileDialog, QHBoxLayout, QLabel, QScrollArea, QVBoxLayout, QWidget
from qfluentwidgets import (
    CardWidget,
    FluentIcon,
    PrimaryPushButton,
    PushButton,
    SubtitleLabel,
)

from services.preview_service import PreviewService
from viewmodels.designer_viewmodel import DesignerViewModel


class PreviewView(QWidget):
    def __init__(self, viewmodel: DesignerViewModel, parent: QWidget | None = None):
        super().__init__(parent)
        self.setObjectName("previewView")
        self.viewmodel = viewmodel
        self.config = viewmodel.config
        self._setup_ui()
        self._bind_viewmodel()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)

        # Header Title and Action Toolbar
        header_card = CardWidget(self)
        header_layout = QHBoxLayout(header_card)
        header_layout.setContentsMargins(15, 10, 15, 10)

        title = SubtitleLabel("Print Outcome Preview (Live Rendered Paper Sheet)", header_card)
        title.setStyleSheet("color: #009FEF; font-size: 18px; font-weight: bold;")
        header_layout.addWidget(title)
        header_layout.addStretch(1)

        self.btn_refresh = PushButton(FluentIcon.SYNC, "Refresh", header_card)
        self.btn_export_pdf = PrimaryPushButton(FluentIcon.PRINT, "Export PDF", header_card)
        self.btn_export_png = PushButton(FluentIcon.PHOTO, "Export PNG", header_card)

        header_layout.addWidget(self.btn_refresh)
        header_layout.addWidget(self.btn_export_pdf)
        header_layout.addWidget(self.btn_export_png)
        layout.addWidget(header_card)

        # Scrollable Image Container
        self.scroll_area = QScrollArea(self)
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setStyleSheet("QScrollArea { border: none; background-color: #1e1e23; }")

        self.lbl_image = QLabel(self.scroll_area)
        self.lbl_image.setAlignment(Qt.AlignCenter)
        self.lbl_image.setStyleSheet("padding: 20px; background-color: #1e1e23;")
        self.scroll_area.setWidget(self.lbl_image)

        layout.addWidget(self.scroll_area, stretch=1)

    def _bind_viewmodel(self) -> None:
        self.btn_refresh.clicked.connect(self.update_preview)
        self.btn_export_pdf.clicked.connect(self._on_export_pdf)
        self.btn_export_png.clicked.connect(self._on_export_png)

        self.viewmodel.layout_changed.connect(lambda _: self.update_preview())
        self.viewmodel.button_added.connect(lambda _: self.update_preview())
        self.viewmodel.marker_added.connect(lambda _: self.update_preview())
        self.viewmodel.button_updated.connect(lambda _: self.update_preview())
        self.viewmodel.marker_updated.connect(lambda _: self.update_preview())

    def showEvent(self, event) -> None:
        super().showEvent(event)
        self.update_preview()

    def update_preview(self) -> None:
        """Render high-resolution PIL layout image and display in scroll view."""
        pil_img = PreviewService.render_preview(self.viewmodel.layout, self.config, scale=3.0)
        qimg = QImage(
            pil_img.tobytes(),
            pil_img.width,
            pil_img.height,
            pil_img.width * 3,
            QImage.Format_RGB888,
        )
        pixmap = QPixmap.fromImage(qimg)
        self.lbl_image.setPixmap(pixmap)

    def _on_export_pdf(self) -> None:
        filepath, _ = QFileDialog.getSaveFileName(
            self, "Export Layout PDF", "virtual_keyboard_layout.pdf", "PDF Files (*.pdf)"
        )
        if filepath:
            self.viewmodel.export_pdf(filepath)

    def _on_export_png(self) -> None:
        filepath, _ = QFileDialog.getSaveFileName(
            self, "Export Layout PNG Image", "virtual_keyboard_layout.png", "PNG Images (*.png)"
        )
        if filepath:
            self.viewmodel.export_png(filepath)

