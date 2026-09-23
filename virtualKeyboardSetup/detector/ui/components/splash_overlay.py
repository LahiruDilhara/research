"""
ui/components/splash_overlay.py

Full-screen startup overlay for the detector app.
Clean dark theme without contrasting black patches or un-scoped cascades.

Layout
──────
  Hero header  (title + subtitle)
  ─────────────────────────────
  [Card: Load Layout]  |  [Card: Select Model]
  ─────────────────────────────
  [  Configure Actions  ]  [  Quick Start  ]
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
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
    PrimaryPushButton,
    PushButton,
    SingleDirectionScrollArea,
    StrongBodyLabel,
    SubtitleLabel,
)

from config.constants import UI_ACCENT, UI_BG_CARD, UI_BG_DARK, UI_TEXT_PRI, UI_TEXT_SEC
from core.interfaces.touch_model import ModelEntry
from utils.logger import setup_logger

logger = setup_logger("SplashOverlay")

_CARD_STYLE = (
    f"CardWidget {{ "
    f"  background-color: {UI_BG_CARD}; "
    "  border: 1px solid rgba(255, 255, 255, 0.06); "
    "  border-radius: 12px; "
    "} "
    "QLabel { "
    "  background-color: transparent; "
    "  border: none; "
    "}"
)


class SplashOverlayWidget(QWidget):
    """Startup card overlay, shown until user loads a layout and selects a model."""

    configure_actions_requested = Signal(object, str, str)
    # (layout: LayoutData, xml_path: str, model_name: str)

    quick_start_requested = Signal(object, str, str)
    # (layout: LayoutData, xml_path: str, model_name: str)

    xml_browse_requested = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("splashOverlayWidget")
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self._layout_data = None
        self._xml_path = ""

        self._build_ui()

    # ── UI Construction ────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        self.setStyleSheet(
            f"SplashOverlayWidget#splashOverlayWidget {{ "
            f"background-color: {UI_BG_DARK}; color: {UI_TEXT_PRI}; }}"
        )
        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(0, 0, 0, 0)

        scroll = SingleDirectionScrollArea(self, orient=Qt.Vertical)
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet(f"QScrollArea {{ border: none; background-color: {UI_BG_DARK}; }}")

        container = QWidget()
        container.setObjectName("splashContainer")
        container.setAttribute(Qt.WA_StyledBackground, True)
        container.setStyleSheet(f"QWidget#splashContainer {{ background-color: {UI_BG_DARK}; }}")

        outer = QVBoxLayout(container)
        outer.setContentsMargins(32, 48, 32, 48)
        outer.addStretch(1)

        hbox = QHBoxLayout()
        hbox.addStretch(1)

        content = QWidget()
        content.setMaximumWidth(860)
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(28)

        # ── Hero header ────────────────────────────────────────────────────────
        header = QVBoxLayout()
        header.setSpacing(8)
        header.setAlignment(Qt.AlignCenter)

        title = SubtitleLabel("Paper Virtual Keyboard Detector", content)
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet(f"background: transparent; color: {UI_ACCENT}; font-size: 26px; font-weight: bold;")

        subtitle = CaptionLabel(
            "Load a designer layout, assign key actions, then run real-time touch detection.",
            content,
        )
        subtitle.setAlignment(Qt.AlignCenter)
        subtitle.setStyleSheet(f"background: transparent; color: {UI_TEXT_SEC}; font-size: 13px;")
        subtitle.setWordWrap(True)

        header.addWidget(title)
        header.addWidget(subtitle)
        content_layout.addLayout(header)

        # ── Two cards ─────────────────────────────────────────────────────────
        cards_row = QHBoxLayout()
        cards_row.setSpacing(20)
        cards_row.addWidget(self._build_layout_card(content), 1)
        cards_row.addWidget(self._build_model_card(content), 1)
        content_layout.addLayout(cards_row)

        # ── Action buttons ─────────────────────────────────────────────────────
        btn_row = QHBoxLayout()
        btn_row.setContentsMargins(0, 0, 0, 0)

        self.btn_continue = PrimaryPushButton(FluentIcon.PLAY, "Configure Key Actions", content)
        self.btn_continue.setFixedHeight(44)
        self.btn_continue.setEnabled(False)
        self.btn_continue.clicked.connect(self._on_continue_clicked)

        btn_row.addWidget(self.btn_continue)
        content_layout.addLayout(btn_row)

        hbox.addWidget(content)
        hbox.addStretch(1)
        outer.addLayout(hbox)
        outer.addStretch(1)

        scroll.setWidget(container)
        root_layout.addWidget(scroll)

    def _build_layout_card(self, parent: QWidget) -> CardWidget:
        card = CardWidget(parent)
        card.setObjectName("layoutCard")
        card.setFixedHeight(230)
        card.setStyleSheet(
            f"#layoutCard {{ background-color: {UI_BG_CARD}; border: 1px solid rgba(255, 255, 255, 0.06); border-radius: 12px; }} "
            "QLabel { background-color: transparent; border: none; }"
        )
        layout = QVBoxLayout(card)
        layout.setContentsMargins(24, 22, 24, 22)
        layout.setSpacing(12)

        lbl_title = StrongBodyLabel("Load Layout", card)
        lbl_title.setStyleSheet(f"background: transparent; color: {UI_ACCENT}; font-size: 16px; font-weight: bold; border: none;")
        lbl_desc = CaptionLabel(
            "Browse for a designer-exported XML layout file.", card
        )
        lbl_desc.setStyleSheet(f"background: transparent; color: {UI_TEXT_SEC}; font-size: 12px; border: none;")
        lbl_desc.setWordWrap(True)

        layout.addWidget(lbl_title)
        layout.addWidget(lbl_desc)

        self.lbl_xml_path = BodyLabel("No layout selected", card)
        self.lbl_xml_path.setStyleSheet(
            f"background: transparent; color: {UI_TEXT_SEC}; font-size: 11px; font-weight: 500; border: none;"
        )
        self.lbl_xml_path.setWordWrap(True)
        layout.addWidget(self.lbl_xml_path)
        layout.addStretch(1)

        self.btn_browse = PushButton(FluentIcon.FOLDER, "Browse XML File...", card)
        self.btn_browse.setFixedHeight(38)
        self.btn_browse.clicked.connect(self._on_browse_clicked)
        layout.addWidget(self.btn_browse)

        return card

    def _build_model_card(self, parent: QWidget) -> CardWidget:
        card = CardWidget(parent)
        card.setObjectName("modelCard")
        card.setFixedHeight(230)
        card.setStyleSheet(
            f"#modelCard {{ background-color: {UI_BG_CARD}; border: 1px solid rgba(255, 255, 255, 0.06); border-radius: 12px; }} "
            "QLabel { background-color: transparent; border: none; }"
        )
        layout = QVBoxLayout(card)
        layout.setContentsMargins(24, 22, 24, 22)
        layout.setSpacing(12)

        lbl_title = StrongBodyLabel("Select AI Model", card)
        lbl_title.setStyleSheet(f"background: transparent; color: {UI_ACCENT}; font-size: 16px; font-weight: bold; border: none;")
        lbl_desc = CaptionLabel(
            "Choose the AI model plugin to use for touch detection.", card
        )
        lbl_desc.setStyleSheet(f"background: transparent; color: {UI_TEXT_SEC}; font-size: 12px; border: none;")
        lbl_desc.setWordWrap(True)

        layout.addWidget(lbl_title)
        layout.addWidget(lbl_desc)

        self.combo_model = ComboBox(card)
        self.combo_model.setFixedHeight(36)
        self.combo_model.setPlaceholderText("No AI models found")
        self.combo_model.currentTextChanged.connect(self._on_model_changed)
        layout.addWidget(self.combo_model)

        self.lbl_model_desc = CaptionLabel("", card)
        self.lbl_model_desc.setStyleSheet(f"background: transparent; color: {UI_TEXT_SEC}; font-size: 11px; border: none;")
        self.lbl_model_desc.setWordWrap(True)
        layout.addWidget(self.lbl_model_desc)
        layout.addStretch(1)

        return card

    # ── Public update methods (called by MainWindow) ───────────────────────────

    def set_model_entries(self, entries: list[ModelEntry]) -> None:
        self.combo_model.blockSignals(True)
        self.combo_model.clear()
        self._model_entries = {e.name: e for e in entries}
        for e in entries:
            self.combo_model.addItem(e.name)

        if entries:
            self.combo_model.setCurrentIndex(0)
            self.lbl_model_desc.setText(entries[0].description)
        self.combo_model.blockSignals(False)
        self._update_buttons()

    def set_layout_loaded(self, layout_data, xml_path: str) -> None:
        if layout_data and hasattr(layout_data, "buttons") and len(layout_data.buttons) > 0:
            self._layout_data = layout_data
            self._xml_path = xml_path
            name = Path(xml_path).name
            btn_count = len(layout_data.buttons)
            self.lbl_xml_path.setText(f"{name} ({btn_count} keys)")
            self.lbl_xml_path.setStyleSheet(
                f"background: transparent; color: #00DC64; font-size: 11px; font-weight: 600; border: none;"
            )
        else:
            self._layout_data = None
            self._xml_path = ""
            self.lbl_xml_path.setText("Invalid layout: 0 keys detected")
            self.lbl_xml_path.setStyleSheet(
                f"background: transparent; color: #EF4444; font-size: 11px; font-weight: 600; border: none;"
            )
        self._update_buttons()

    # ── Slots ──────────────────────────────────────────────────────────────────

    def _on_model_changed(self, text: str) -> None:
        entry = self._model_entries.get(text)
        if entry:
            self.lbl_model_desc.setText(entry.description)
        self._update_buttons()

    def _on_browse_clicked(self) -> None:
        self.xml_browse_requested.emit()

    def _on_continue_clicked(self) -> None:
        current_text = self.combo_model.currentText()
        entry = self._model_entries.get(current_text)
        canonical_name = entry.name if entry else current_text

        if self._layout_data and self._xml_path and canonical_name:
            self.configure_actions_requested.emit(
                self._layout_data, self._xml_path, canonical_name
            )

    def _update_buttons(self) -> None:
        has_valid_layout = bool(
            self._layout_data
            and hasattr(self._layout_data, "buttons")
            and len(self._layout_data.buttons) > 0
            and self._xml_path
            and Path(self._xml_path).exists()
        )
        has_valid_model = bool(self.combo_model.currentText())
        ready = has_valid_layout and has_valid_model
        self.btn_continue.setEnabled(ready)

        name = self.combo_model.currentText()
        if hasattr(self, "_model_entries") and name in self._model_entries:
            self.lbl_model_desc.setText(self._model_entries[name].description)
