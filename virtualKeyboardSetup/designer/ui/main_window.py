"""
Main Application Shell (MVVM Architecture).
Subclasses PySide6-Fluent-Widgets FluentWindow with sidebar navigation,
toolbar actions, theme customization, and ViewModel orchestration.
"""

from pathlib import Path
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QFileDialog
from qfluentwidgets import (
    Action,
    FluentIcon,
    FluentWindow,
    InfoBar,
    NavigationItemPosition,
    setTheme,
    Theme,
)

from ui.views.designer_view import DesignerView
from ui.views.settings_view import SettingsView
from ui.views.about_view import AboutView
from ui.views.preview_view import PreviewView
from ui.components.preview_dialog import PreviewDialog
from ui.components.splash_overlay import SplashOverlayWidget
from config.app_config import AppConfig
from viewmodels.designer_viewmodel import DesignerViewModel
from viewmodels.settings_viewmodel import SettingsViewModel


class MainWindow(FluentWindow):
    def __init__(self, config: AppConfig):
        super().__init__()
        self.config = config
        self.setWindowTitle(config.app_title)
        self.resize(1280, 800)
        self.setMinimumSize(900, 600)

        # 1. Instantiate ViewModels
        self.designer_vm = DesignerViewModel(self.config, self)
        self.settings_vm = SettingsViewModel(self.config, self)

        # Apply Theme
        if config.app_theme.lower() == "light":
            setTheme(Theme.LIGHT)
        else:
            setTheme(Theme.DARK)

        self._bind_viewmodel_messages()
        self._setup_views()
        self._setup_actions()

        # Full-Screen Dark Splash Overlay (Positioned below titleBar to keep close buttons visible)
        self.splash_overlay = SplashOverlayWidget(self.config, self)
        self.splash_overlay.create_project_requested.connect(self._on_splash_create_project)
        self.splash_overlay.open_project_requested.connect(self._on_splash_open_project)
        self._update_splash_geometry()
        self.splash_overlay.show()
        self.splash_overlay.raise_()

    def _update_splash_geometry(self) -> None:
        if hasattr(self, "splash_overlay") and self.splash_overlay is not None:
            tb_height = self.titleBar.height() if hasattr(self, "titleBar") and self.titleBar else 36
            self.splash_overlay.setGeometry(0, tb_height, self.width(), max(0, self.height() - tb_height))

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._update_splash_geometry()
        if hasattr(self, "splash_overlay") and self.splash_overlay and self.splash_overlay.isVisible():
            self.splash_overlay.raise_()

    def show_start_window(self) -> None:
        """Show full application dark splash screen overlay on user request."""
        self._update_splash_geometry()
        self.splash_overlay.show()
        self.splash_overlay.raise_()

    def _on_splash_create_project(self, name: str, width_mm: float, height_mm: float, margin_mm: float) -> None:
        """Handle project creation from full app splash screen overlay."""
        self.config.paper_width_mm = width_mm
        self.config.paper_height_mm = height_mm
        self.config.paper_margin_mm = margin_mm
        self.designer_vm.create_new_layout()
        self.designer_vm.update_paper_dimensions(width_mm, height_mm)
        self.designer_view.canvas.update_paper_dimensions(width_mm, height_mm)
        self.settings_view.sync_from_config()
        self.setWindowTitle(f"{self.config.app_title} - [{name}]")
        self.splash_overlay.hide()

    def _on_splash_open_project(self, filepath: str) -> None:
        """Handle project open from full app splash screen overlay."""
        self.designer_vm.load_project_xml(filepath)
        self.settings_view.sync_from_config()
        self.designer_view.canvas.update_paper_dimensions(self.config.paper_width_mm, self.config.paper_height_mm)
        self.setWindowTitle(f"{self.config.app_title} - [{Path(filepath).name}]")
        self.splash_overlay.hide()

    def _bind_viewmodel_messages(self) -> None:
        """Bind ViewModel status and error signals to InfoBar notifications."""
        self.designer_vm.status_message.connect(
            lambda title, content: InfoBar.success(title, content, parent=self)
        )
        self.designer_vm.error_message.connect(
            lambda title, content: InfoBar.error(title, content, parent=self)
        )
        self.settings_vm.status_message.connect(
            lambda title, content: InfoBar.success(title, content, parent=self)
        )
        self.settings_vm.paper_dimensions_changed.connect(self._on_paper_dimensions_changed)

    def _setup_views(self) -> None:
        self.designer_view = DesignerView(self.designer_vm, self)
        self.preview_view = PreviewView(self.designer_vm, self)
        self.settings_view = SettingsView(self.settings_vm, self)
        self.about_view = AboutView(self.config, self)

        self.addSubInterface(
            self.designer_view,
            FluentIcon.EDIT,
            "Layout Canvas",
            NavigationItemPosition.TOP,
        )
        self.addSubInterface(
            self.preview_view,
            FluentIcon.PRINT,
            "Print Preview",
            NavigationItemPosition.TOP,
        )
        self.addSubInterface(
            self.settings_view,
            FluentIcon.SETTING,
            "Settings",
            NavigationItemPosition.BOTTOM,
        )
        self.addSubInterface(
            self.about_view,
            FluentIcon.INFO,
            "About",
            NavigationItemPosition.BOTTOM,
        )

    def _on_paper_dimensions_changed(self, width_mm: float, height_mm: float) -> None:
        self.designer_vm.update_paper_dimensions(width_mm, height_mm)
        self.designer_view.canvas.update_paper_dimensions(width_mm, height_mm)

    def _setup_actions(self) -> None:
        self.titleBar.titleLabel.setContentsMargins(15, 0, 0, 0)

        action_new = Action(FluentIcon.ADD, "New Layout", self)
        action_new.triggered.connect(self.show_start_window)

        action_open = Action(FluentIcon.FOLDER, "Open", self)
        action_open.triggered.connect(self._on_action_open)

        action_save = Action(FluentIcon.SAVE, "Save", self)
        action_save.triggered.connect(self._on_action_save)

        action_db_save = Action(FluentIcon.SYNC, "DB Sync", self)
        action_db_save.triggered.connect(self.designer_vm.sync_to_database)

        action_export_pdf = Action(FluentIcon.PRINT, "Export PDF", self)
        action_export_pdf.triggered.connect(self._on_action_export_pdf)

        action_fit_view = Action(FluentIcon.ZOOM_IN, "Fit to View", self)
        action_fit_view.triggered.connect(self.designer_view.canvas.fit_layout_to_view)

        action_toggle_panel = Action(FluentIcon.MENU, "Side Panel", self)
        action_toggle_panel.triggered.connect(self.designer_view.toggle_side_panel)

        action_preview = Action(FluentIcon.VIEW, "Print Preview", self)
        action_preview.triggered.connect(self._on_action_preview)

    def _on_action_open(self) -> None:
        filepath, _ = QFileDialog.getOpenFileName(
            self, "Open Layout", "", "XML Files (*.xml)"
        )
        if filepath:
            self.designer_vm.load_project_xml(filepath)
            self.settings_view.sync_from_config()

    def _on_action_save(self) -> None:
        filepath, _ = QFileDialog.getSaveFileName(
            self, "Save Layout", "layout.xml", "XML Files (*.xml)"
        )
        if filepath:
            self.designer_vm.save_project_xml(filepath)

    def _on_action_export_pdf(self) -> None:
        filepath, _ = QFileDialog.getSaveFileName(
            self, "Export Printable PDF", "layout.pdf", "PDF Files (*.pdf)"
        )
        if filepath:
            self.designer_vm.export_pdf(filepath)

    def _on_action_preview(self) -> None:
        dlg = PreviewDialog(self.designer_vm.layout, self.config, self)
        dlg.exec()
