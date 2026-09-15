"""
Main Application Shell (MVVM Architecture).
Subclasses PySide6-Fluent-Widgets FluentWindow with sidebar navigation,
toolbar actions, theme customization, and ViewModel orchestration.
"""

import re
from pathlib import Path
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QFileDialog, QDialog
from qfluentwidgets import (
    Action,
    FluentIcon,
    FluentWindow,
    InfoBar,
    MessageBox,
    NavigationItemPosition,
    PushButton,
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

    def _get_default_filename(self) -> str:
        """Derive default XML filename from existing open file path or project name."""
        if self.designer_vm.current_filepath:
            return Path(self.designer_vm.current_filepath).name
        proj_name = self.designer_vm.layout.project_name if self.designer_vm.layout else "My Paper Keyboard"
        clean_name = re.sub(r'[^\w\-]+', '_', proj_name.strip()).strip('_').lower()
        return f"{clean_name if clean_name else 'paper_layout'}.xml"

    def _show_save_confirmation_dialog(self) -> str:
        """Show 3-option confirmation dialog when closing or opening a new file with unsaved changes."""
        title = "Unsaved Changes"
        content = "You have unsaved changes in the layout. Do you want to save changes before closing?"
        box = MessageBox(title, content, self)
        box.yesButton.setText("Save")
        box.cancelButton.setText("Cancel")

        dont_save_btn = PushButton("Don't Save", box)
        box.buttonLayout.insertWidget(1, dont_save_btn)

        buttons = [box.yesButton, dont_save_btn, box.cancelButton]
        max_w = max(110, max(b.sizeHint().width() + 20 for b in buttons))
        for b in buttons:
            b.setFixedWidth(max_w)

        result = "cancel"

        def on_save():
            nonlocal result
            result = "save"
            box.done(QDialog.Accepted)

        def on_dont_save():
            nonlocal result
            result = "dont_save"
            box.done(2)

        def on_cancel():
            nonlocal result
            result = "cancel"
            box.done(QDialog.Rejected)

        box.yesButton.clicked.disconnect()
        box.cancelButton.clicked.disconnect()
        box.yesButton.clicked.connect(on_save)
        dont_save_btn.clicked.connect(on_dont_save)
        box.cancelButton.clicked.connect(on_cancel)

        box.exec()
        return result

    def _save_current_layout(self) -> bool:
        """Save current layout directly if filepath exists, or prompt file dialog using project name."""
        if self.designer_vm.current_filepath:
            self.designer_vm.save_project_xml(self.designer_vm.current_filepath)
            return not self.designer_vm.is_dirty
        else:
            default_name = self._get_default_filename()
            filepath, _ = QFileDialog.getSaveFileName(
                self, "Save Layout", default_name, "XML Files (*.xml)"
            )
            if filepath:
                self.designer_vm.save_project_xml(filepath)
                return not self.designer_vm.is_dirty
            return False

    def _prompt_save_if_dirty(self) -> bool:
        """Check if layout has unsaved changes and prompt user. Returns True if safe to proceed."""
        if not self.designer_vm.is_dirty:
            return True

        choice = self._show_save_confirmation_dialog()
        if choice == "save":
            return self._save_current_layout()
        elif choice == "dont_save":
            return True
        else:  # "cancel" or closed dialog
            return False

    def closeEvent(self, event) -> None:
        """Intercept application close event to prompt for unsaved changes."""
        if self._prompt_save_if_dirty():
            event.accept()
        else:
            event.ignore()

    def show_start_window(self) -> None:
        """Show full application dark splash screen overlay on user request."""
        if not self._prompt_save_if_dirty():
            return
        self._update_splash_geometry()
        self.splash_overlay.show()
        self.splash_overlay.raise_()

    def _on_splash_create_project(self, name: str, width_mm: float, height_mm: float, margin_mm: float) -> None:
        """Handle project creation from full app splash screen overlay."""
        self.config.paper_width_mm = width_mm
        self.config.paper_height_mm = height_mm
        self.config.paper_margin_mm = margin_mm
        self.designer_vm.create_new_layout(name)
        self.designer_vm.update_paper_dimensions(width_mm, height_mm)
        self.designer_view.canvas.update_paper_dimensions(width_mm, height_mm)
        self.settings_view.sync_from_config()
        self._update_window_title()
        self.splash_overlay.hide()

    def _on_splash_open_project(self, filepath: str) -> None:
        """Handle project open from full app splash screen overlay."""
        self.designer_vm.load_project_xml(filepath)
        self.settings_view.sync_from_config()
        self.designer_view.canvas.update_paper_dimensions(self.config.paper_width_mm, self.config.paper_height_mm)
        self._update_window_title()
        self.splash_overlay.hide()

    def _update_window_title(self) -> None:
        proj_name = self.designer_vm.layout.project_name if self.designer_vm.layout else ""
        file_name = Path(self.designer_vm.current_filepath).name if self.designer_vm.current_filepath else proj_name
        dirty_suffix = " *" if self.designer_vm.is_dirty else ""
        if file_name:
            self.setWindowTitle(f"{self.config.app_title} - [{file_name}]{dirty_suffix}")
        else:
            self.setWindowTitle(f"{self.config.app_title}{dirty_suffix}")

    def _bind_viewmodel_messages(self) -> None:
        """Bind ViewModel status and error signals to InfoBar notifications."""
        self.designer_vm.status_message.connect(
            lambda title, content: InfoBar.success(title, content, parent=self)
        )
        self.designer_vm.error_message.connect(
            lambda title, content: InfoBar.error(title, content, parent=self)
        )
        self.designer_vm.dirty_changed.connect(self._update_window_title)
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
        action_save.setShortcut("Ctrl+S")
        action_save.triggered.connect(self._on_action_save)

        action_save_as = Action(FluentIcon.SAVE_AS, "Save As...", self)
        action_save_as.setShortcut("Ctrl+Shift+S")
        action_save_as.triggered.connect(self._on_action_save_as)

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
        if not self._prompt_save_if_dirty():
            return
        default_name = self._get_default_filename()
        filepath, _ = QFileDialog.getOpenFileName(
            self, "Open Layout", "", "XML Files (*.xml)"
        )
        if filepath:
            self.designer_vm.load_project_xml(filepath)
            self.settings_view.sync_from_config()
            self._update_window_title()

    def _on_action_save(self) -> None:
        """Save current layout directly to existing open file path, or invoke Save As if new."""
        if self.designer_vm.current_filepath:
            self.designer_vm.save_project_xml(self.designer_vm.current_filepath)
            self._update_window_title()
        else:
            self._on_action_save_as()

    def _on_action_save_as(self) -> None:
        """Prompt file dialog to save layout under a target file path, defaulting to project_name.xml."""
        default_name = self._get_default_filename()
        filepath, _ = QFileDialog.getSaveFileName(
            self, "Save Layout As", default_name, "XML Files (*.xml)"
        )
        if filepath:
            self.designer_vm.save_project_xml(filepath)
            self._update_window_title()

    def _on_action_export_pdf(self) -> None:
        default_name = f"{Path(self._get_default_filename()).stem}.pdf"
        filepath, _ = QFileDialog.getSaveFileName(
            self, "Export Printable PDF", default_name, "PDF Files (*.pdf)"
        )
        if filepath:
            self.designer_vm.export_pdf(filepath)

    def _on_action_preview(self) -> None:
        dlg = PreviewDialog(self.designer_vm.layout, self.config, self)
        dlg.exec()


