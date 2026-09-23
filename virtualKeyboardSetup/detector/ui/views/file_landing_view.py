"""
ui/views/file_landing_view.py

Startup landing view. Clean, minimal, single-purpose: load a layout XML file.
Two states: empty (choose file) and loaded (confirm and enter workspace).
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QDragEnterEvent, QDropEvent
from PySide6.QtWidgets import (
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)
from qfluentwidgets import (
    CaptionLabel,
    InfoBar,
    InfoBarPosition,
    StrongBodyLabel,
)

from core.layout.layout_parser import LayoutData
from viewmodels.startup_viewmodel import StartupViewModel


# ── Palette ────────────────────────────────────────────────────────────────────
_BG      = "#18191E"
_CARD    = "#1E1F26"
_BORDER  = "#2C2E38"
_ACCENT  = "#009FEF"
_SUCCESS = "#10B981"
_PRI     = "#F1F3F7"
_SEC     = "#7B8192"
_DROP    = "#1A1B22"
_DROPHOV = "#1C2A38"


# ── Shared helpers ─────────────────────────────────────────────────────────────

def _hr(parent: QWidget) -> QFrame:
    """Thin 1px horizontal rule."""
    line = QFrame(parent)
    line.setFrameShape(QFrame.HLine)
    line.setFixedHeight(1)
    line.setStyleSheet(f"background: {_BORDER}; border: none;")
    return line


def _pill(text: str, parent: QWidget) -> QLabel:
    """Small muted badge."""
    lbl = QLabel(text, parent)
    lbl.setStyleSheet(
        f"color: {_SEC};"
        f"background: #252730;"
        f"border: 1px solid {_BORDER};"
        f"border-radius: 4px;"
        f"padding: 1px 9px;"
        f"font-size: 11px;"
    )
    lbl.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
    return lbl


def _btn(label: str, primary: bool, parent: QWidget) -> QPushButton:
    """
    Plain QPushButton with no FluentIcon so there is no icon-text overlap.
    `primary=True` gives the filled accent style; False gives ghost style.
    """
    btn = QPushButton(label, parent)
    if primary:
        btn.setStyleSheet(
            f"QPushButton {{"
            f"  background: {_ACCENT};"
            f"  color: #ffffff;"
            f"  border: none;"
            f"  border-radius: 8px;"
            f"  font-size: 14px;"
            f"  font-weight: 600;"
            f"  padding: 0 20px;"
            f"}}"
            f"QPushButton:hover {{"
            f"  background: #007fd4;"
            f"}}"
            f"QPushButton:disabled {{"
            f"  background: #2A2D38;"
            f"  color: {_SEC};"
            f"}}"
        )
    else:
        btn.setStyleSheet(
            f"QPushButton {{"
            f"  background: #252730;"
            f"  color: {_PRI};"
            f"  border: 1px solid {_BORDER};"
            f"  border-radius: 7px;"
            f"  font-size: 13px;"
            f"  padding: 0 16px;"
            f"}}"
            f"QPushButton:hover {{"
            f"  background: #2D3040;"
            f"  border-color: #3C4055;"
            f"}}"
        )
    return btn


# ── Drop zone widget ───────────────────────────────────────────────────────────

class _DropTarget(QFrame):
    """Dashed drop zone that lights up on valid drag hover."""

    file_dropped = Signal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setAcceptDrops(True)
        self._hov = False
        self._style()

    def _style(self) -> None:
        bg  = _DROPHOV if self._hov else _DROP
        bdr = _ACCENT  if self._hov else _BORDER
        self.setStyleSheet(
            f"QFrame {{"
            f"  background: {bg};"
            f"  border: 1.5px dashed {bdr};"
            f"  border-radius: 10px;"
            f"}}"
            f"QLabel {{ background: transparent; border: none; }}"
        )

    def dragEnterEvent(self, e: QDragEnterEvent) -> None:
        if e.mimeData().hasUrls():
            for url in e.mimeData().urls():
                if url.toLocalFile().lower().endswith(".xml"):
                    e.acceptProposedAction()
                    self._hov = True
                    self._style()
                    return
        e.ignore()

    def dragLeaveEvent(self, e) -> None:
        self._hov = False
        self._style()

    def dropEvent(self, e: QDropEvent) -> None:
        self._hov = False
        self._style()
        for url in e.mimeData().urls():
            fp = url.toLocalFile()
            if fp.lower().endswith(".xml"):
                self.file_dropped.emit(fp)
                e.acceptProposedAction()
                return


# ── Main view ──────────────────────────────────────────────────────────────────

class FileLandingView(QWidget):
    """Full-window, zero-sidebar startup screen for loading a layout file."""

    workspace_entered = Signal(object, str)   # LayoutData, xml_path

    def __init__(
        self,
        startup_vm: StartupViewModel,
        last_xml_path: str = "",
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._vm       = startup_vm
        self._last_xml = last_xml_path
        self._layout: LayoutData | None = None
        self._xml_path = ""

        self._build()
        self._vm.layout_loaded.connect(self._on_loaded)
        self._vm.error_occurred.connect(self._on_error)

        if self._last_xml and Path(self._last_xml).is_file():
            self._load(self._last_xml)

    # ── Build ──────────────────────────────────────────────────────────────────

    def _build(self) -> None:
        self.setStyleSheet(f"background: {_BG};")

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setAlignment(Qt.AlignCenter)

        # Narrow centered panel
        panel = QFrame(self)
        panel.setFixedWidth(520)
        panel.setStyleSheet(
            f"QFrame {{"
            f"  background: {_CARD};"
            f"  border: 1px solid {_BORDER};"
            f"  border-radius: 14px;"
            f"}}"
            f"QLabel {{ background: transparent; border: none; }}"
        )

        vbox = QVBoxLayout(panel)
        vbox.setContentsMargins(36, 30, 36, 28)
        vbox.setSpacing(0)

        # ── Header row ─────────────────────────────────────────────────────────
        hdr = QHBoxLayout()
        hdr.setSpacing(14)

        icon_frame = QFrame(panel)
        icon_frame.setFixedSize(42, 42)
        icon_frame.setStyleSheet(
            "background: #0D1E2F; border: 1px solid #1A3350; border-radius: 10px;"
        )
        icon_vbox = QVBoxLayout(icon_frame)
        icon_vbox.setContentsMargins(0, 0, 0, 0)
        icon_vbox.setAlignment(Qt.AlignCenter)
        icon_lbl = QLabel("⌨", icon_frame)
        icon_lbl.setStyleSheet(
            f"color: {_ACCENT}; font-size: 21px; font-weight: bold;"
            f"background: transparent; border: none;"
        )
        icon_lbl.setAlignment(Qt.AlignCenter)
        icon_vbox.addWidget(icon_lbl)
        hdr.addWidget(icon_frame)

        title_vbox = QVBoxLayout()
        title_vbox.setSpacing(1)
        title_lbl = StrongBodyLabel("Paper Virtual Keyboard", panel)
        title_lbl.setStyleSheet(f"color: {_PRI}; font-size: 16px; font-weight: 700;")
        sub_lbl   = CaptionLabel("Type on any plain paper with a regular camera", panel)
        sub_lbl.setStyleSheet(f"color: {_SEC}; font-size: 12px;")
        title_vbox.addWidget(title_lbl)
        title_vbox.addWidget(sub_lbl)
        hdr.addLayout(title_vbox, 1)

        vbox.addLayout(hdr)
        vbox.addSpacing(22)
        vbox.addWidget(_hr(panel))
        vbox.addSpacing(22)

        # ── File selection label ───────────────────────────────────────────────
        open_lbl = QLabel("Select a keyboard layout file", panel)
        open_lbl.setStyleSheet(
            f"color: {_PRI}; font-size: 13px; font-weight: 600;"
            f"background: transparent; border: none;"
        )
        vbox.addWidget(open_lbl)
        vbox.addSpacing(14)

        # ── Drop zone ──────────────────────────────────────────────────────────
        self.drop_zone = _DropTarget(panel)
        self.drop_zone.setFixedHeight(100)
        dz_vbox = QVBoxLayout(self.drop_zone)
        dz_vbox.setAlignment(Qt.AlignCenter)
        dz_vbox.setSpacing(3)

        dz_lbl1 = QLabel("Drag and drop your .xml file here", self.drop_zone)
        dz_lbl1.setStyleSheet(
            f"color: {_PRI}; font-size: 13px;"
            f"background: transparent; border: none;"
        )
        dz_lbl1.setAlignment(Qt.AlignCenter)

        dz_lbl2 = QLabel("or", self.drop_zone)
        dz_lbl2.setStyleSheet(
            f"color: {_SEC}; font-size: 11px;"
            f"background: transparent; border: none;"
        )
        dz_lbl2.setAlignment(Qt.AlignCenter)

        dz_vbox.addWidget(dz_lbl1)
        dz_vbox.addWidget(dz_lbl2)
        self.drop_zone.file_dropped.connect(self._load)
        vbox.addWidget(self.drop_zone)
        vbox.addSpacing(10)

        # ── Browse button ──────────────────────────────────────────────────────
        self.btn_browse = _btn("Browse File", primary=False, parent=panel)
        self.btn_browse.setFixedHeight(38)
        self.btn_browse.clicked.connect(self._on_browse)
        vbox.addWidget(self.btn_browse)

        # ── Recent shortcut (shown only if previous session exists) ────────────
        self.recent_bar = QWidget(panel)
        self.recent_bar.setStyleSheet("background: transparent;")
        rb = QHBoxLayout(self.recent_bar)
        rb.setContentsMargins(0, 0, 0, 0)
        rb.setSpacing(6)

        rc_lbl = QLabel("Recent:", self.recent_bar)
        rc_lbl.setStyleSheet(f"color: {_SEC}; font-size: 11px; background: transparent; border: none;")
        self.btn_recent = QPushButton("", self.recent_bar)
        self.btn_recent.setFixedHeight(26)
        self.btn_recent.setStyleSheet(
            f"QPushButton {{"
            f"  background: transparent; border: none;"
            f"  color: {_ACCENT}; font-size: 11px; text-align: left;"
            f"}}"
            f"QPushButton:hover {{ text-decoration: underline; }}"
        )
        self.btn_recent.clicked.connect(self._on_load_recent)
        rb.addWidget(rc_lbl)
        rb.addWidget(self.btn_recent, 1)

        if self._last_xml and Path(self._last_xml).is_file():
            self.btn_recent.setText(Path(self._last_xml).name)
            self.recent_bar.setVisible(True)
        else:
            self.recent_bar.setVisible(False)

        vbox.addSpacing(6)
        vbox.addWidget(self.recent_bar)

        # ── Summary block (hidden until file loaded) ───────────────────────────
        vbox.addSpacing(18)
        vbox.addWidget(_hr(panel))
        vbox.addSpacing(14)

        self.summary_box = QWidget(panel)
        self.summary_box.setStyleSheet("background: transparent;")
        sb = QVBoxLayout(self.summary_box)
        sb.setContentsMargins(0, 0, 0, 0)
        sb.setSpacing(8)

        # File name row
        fname_row = QHBoxLayout()
        fname_row.setSpacing(8)

        dot = QLabel("●", self.summary_box)
        dot.setStyleSheet(f"color: {_SUCCESS}; font-size: 12px; background: transparent; border: none;")
        dot.setFixedWidth(14)

        self.lbl_fname = QLabel("", self.summary_box)
        self.lbl_fname.setStyleSheet(
            f"color: {_PRI}; font-size: 14px; font-weight: 600;"
            f"background: transparent; border: none;"
        )

        self.btn_change = QPushButton("Change", self.summary_box)
        self.btn_change.setFixedHeight(26)
        self.btn_change.setFixedWidth(70)
        self.btn_change.setStyleSheet(
            f"QPushButton {{"
            f"  background: #252730; color: {_SEC};"
            f"  border: 1px solid {_BORDER}; border-radius: 5px;"
            f"  font-size: 11px;"
            f"}}"
            f"QPushButton:hover {{ color: {_PRI}; background: #2D3040; }}"
        )
        self.btn_change.clicked.connect(self._on_browse)

        fname_row.addWidget(dot)
        fname_row.addWidget(self.lbl_fname, 1)
        fname_row.addWidget(self.btn_change)
        sb.addLayout(fname_row)

        # Stats pills
        pills_row = QHBoxLayout()
        pills_row.setSpacing(6)
        pills_row.setContentsMargins(18, 0, 0, 0)
        self.pill_keys = _pill("0 keys",      self.summary_box)
        self.pill_tags = _pill("0 AprilTags", self.summary_box)
        self.pill_dims = _pill("",            self.summary_box)
        pills_row.addWidget(self.pill_keys)
        pills_row.addWidget(self.pill_tags)
        pills_row.addWidget(self.pill_dims)
        pills_row.addStretch(1)
        sb.addLayout(pills_row)

        self.summary_box.setVisible(False)
        vbox.addWidget(self.summary_box)
        vbox.addSpacing(20)

        # ── Primary CTA ────────────────────────────────────────────────────────
        self.btn_open = _btn("Open Workspace", primary=True, parent=panel)
        self.btn_open.setFixedHeight(44)
        self.btn_open.setEnabled(False)
        self.btn_open.clicked.connect(self._on_enter)
        vbox.addWidget(self.btn_open)

        root.addWidget(panel, 0, Qt.AlignCenter)

    # ── Actions ────────────────────────────────────────────────────────────────

    def _on_browse(self) -> None:
        start = str(
            Path(self._xml_path).parent if self._xml_path and Path(self._xml_path).parent.is_dir()
            else Path(self._last_xml).parent if self._last_xml and Path(self._last_xml).parent.is_dir()
            else Path.home()
        )
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Layout File", start, "XML Layout Files (*.xml);;All Files (*)"
        )
        if path:
            self._load(path)

    def _on_load_recent(self) -> None:
        if self._last_xml and Path(self._last_xml).is_file():
            self._load(self._last_xml)

    def _load(self, path: str) -> None:
        resolved = str(Path(path).resolve())
        if not Path(resolved).is_file():
            self._on_error(f"File not found: {resolved}")
            return
        self._xml_path = resolved
        self._vm.load_layout(resolved)

    def _on_loaded(self, layout: LayoutData) -> None:
        self._layout = layout
        self.lbl_fname.setText(Path(self._xml_path).name)
        self.pill_keys.setText(f"{len(layout.buttons)} keys")
        self.pill_tags.setText(f"{len(layout.markers)} AprilTags")
        w = int(round(layout.paper_width_mm))
        h = int(round(layout.paper_height_mm))
        self.pill_dims.setText(f"{w} x {h} mm")

        # Switch to confirmed state
        self.drop_zone.setVisible(False)
        self.btn_browse.setVisible(False)
        self.summary_box.setVisible(True)
        self.btn_open.setEnabled(True)

    def _on_enter(self) -> None:
        if self._layout is None:
            self._on_error("Please select a layout file first.")
            return
        self.workspace_entered.emit(self._layout, self._xml_path)

    def _on_error(self, message: str) -> None:
        InfoBar.error(
            title="Cannot Load Layout",
            content=message,
            position=InfoBarPosition.TOP,
            parent=self,
            duration=4000,
        )
