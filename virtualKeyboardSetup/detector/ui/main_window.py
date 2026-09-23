"""
ui/main_window.py

FluentWindow shell for the detector app.

Navigation
──────────
  Home             → hosts a QStackedWidget (Splash → Action Config → Camera Select → Detector).
  Settings         → settings view.

View Routing
────────────
  All screen transitions happen cleanly inside the Home stacked widget without creating
  any extra navigation tabs in the sidebar.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QFileDialog, QStackedWidget, QWidget
from qfluentwidgets import FluentIcon, FluentWindow, NavigationItemPosition

from config.app_config import AppConfig
from config.constants import UI_BG_DARK
from core.action.action_executor import ActionData
from core.interfaces.touch_model import ModelEntry, ModelRegistry
from services.action_config_service import ActionConfigService
from ui.components.splash_overlay import SplashOverlayWidget
from ui.views.action_config_view import ActionConfigView
from ui.views.camera_select_view import CameraSelectView
from ui.views.detector_view import DetectorView
from ui.views.settings_view import SettingsView
from viewmodels.action_config_viewmodel import ActionConfigViewModel
from viewmodels.camera_select_viewmodel import CameraSelectViewModel
from viewmodels.detector_viewmodel import DetectorViewModel
from viewmodels.startup_viewmodel import StartupViewModel
from utils.logger import setup_logger

logger = setup_logger("MainWindow")


class MainWindow(FluentWindow):
    """Application shell, dark FluentWindow with internal stacked view routing."""

    def __init__(self, config: AppConfig) -> None:
        super().__init__()
        self._config = config
        self._current_model_entry: ModelEntry | None = None
        self._active_det_vm: DetectorViewModel | None = None
        self._env_path = str(Path(__file__).resolve().parent.parent / ".env")

        # ViewModels
        plugins_dir = str(Path(__file__).resolve().parent.parent / config.plugins_dir)
        self._startup_vm = StartupViewModel(plugins_dir, parent=self)

        self._setup_window()
        self._build_pages()
        self._connect_startup_vm()

        # Discover plugins
        self._startup_vm.discover_models()

    # ── Window setup ───────────────────────────────────────────────────────────

    def _setup_window(self) -> None:
        self.setWindowTitle(self._config.app_title)
        self.setMinimumSize(1100, 720)
        self.resize(1280, 800)
        self.setStyleSheet(
            f"FluentWindow {{ background-color: {UI_BG_DARK}; }}"
        )
        self.navigationInterface.setAcrylicEnabled(False)

    # ── Pages ──────────────────────────────────────────────────────────────────

    def _build_pages(self) -> None:
        # ── Home stacked container ─────────────────────────────────────────────
        self._home_stack = QStackedWidget(self)
        self._home_stack.setObjectName("homeView")
        self._home_stack.setStyleSheet(f"QStackedWidget {{ background-color: {UI_BG_DARK}; border: none; }}")

        # Startup Splash View
        self._splash = SplashOverlayWidget(self._home_stack)
        self._splash.setObjectName("splashView")
        self._splash.xml_browse_requested.connect(self._on_browse_xml)
        self._splash.configure_actions_requested.connect(self._on_configure_requested)
        self._splash.quick_start_requested.connect(self._on_quick_start)
        self._home_stack.addWidget(self._splash)

        self.addSubInterface(
            self._home_stack,
            FluentIcon.HOME,
            "Home",
            position=NavigationItemPosition.TOP,
        )

        # ── Settings page ──────────────────────────────────────────────────────
        self._settings_view = SettingsView(self._env_path)
        self._settings_view.setObjectName("settingsView")
        self.addSubInterface(
            self._settings_view,
            FluentIcon.SETTING,
            "Settings",
            position=NavigationItemPosition.BOTTOM,
        )

    def _connect_startup_vm(self) -> None:
        self._startup_vm.layout_loaded.connect(self._on_layout_loaded)
        self._startup_vm.models_ready.connect(self._splash.set_model_entries)
        self._startup_vm.error_occurred.connect(self._show_error)

    def _on_layout_loaded(self, layout) -> None:
        xml_path = self._startup_vm.layout_path
        self._splash.set_layout_loaded(layout, xml_path)
        self._config.set_last_xml_path(xml_path)

    # ── Navigation / routing ───────────────────────────────────────────────────

    def _on_browse_xml(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Layout XML", "", "XML Files (*.xml)"
        )
        if path:
            self._startup_vm.load_layout(path)

    def _on_configure_requested(
        self, layout_data, xml_path: str, model_name: str
    ) -> None:
        self._current_model_entry = self._get_model_entry(model_name)
        vm = ActionConfigViewModel(layout_data, xml_path, parent=self)
        view = ActionConfigView(vm, parent=self._home_stack)
        view.setObjectName("actionConfigView")
        view.back_requested.connect(self._go_home)
        view.start_requested.connect(
            lambda: self._go_to_camera_select(layout_data, xml_path, vm.config)
        )
        self._push_view(view)

    def _on_quick_start(
        self, layout_data, xml_path: str, model_name: str
    ) -> None:
        self._current_model_entry = self._get_model_entry(model_name)
        # Load existing actions or empty defaults
        svc = ActionConfigService()
        btn_ids = [b.id for b in layout_data.buttons]
        action_config = svc.load(xml_path, btn_ids)
        self._go_to_camera_select(layout_data, xml_path, action_config)

    def _go_to_camera_select(
        self, layout_data, xml_path: str, action_config: dict[str, ActionData]
    ) -> None:
        cam_vm = CameraSelectViewModel(parent=self)
        cam_view = CameraSelectView(cam_vm, parent=self._home_stack)
        cam_view.setObjectName("cameraSelectView")
        cam_view.back_requested.connect(self._go_home)
        cam_view.start_requested.connect(
            lambda cam_idx: self._start_detector(layout_data, action_config, cam_idx)
        )
        self._push_view(cam_view)

    def _start_detector(
        self,
        layout_data,
        action_config: dict[str, ActionData],
        camera_index: int,
    ) -> None:
        if self._current_model_entry is None:
            self._show_error("No model selected.")
            return

        # Stop previous detector if running
        if self._active_det_vm is not None:
            self._active_det_vm.stop()
            self._active_det_vm = None

        det_vm = DetectorViewModel(
            layout=layout_data,
            action_config=action_config,
            config=self._config,
            camera_index=camera_index,
            parent=self,
        )
        det_vm.set_model(self._current_model_entry)
        self._active_det_vm = det_vm

        det_view = DetectorView(det_vm, self._current_model_entry, parent=self._home_stack)
        det_view.setObjectName("detectorView")
        det_view.set_layout_data(layout_data)
        det_view.stop_requested.connect(self._go_home)

        self._push_view(det_view)
        det_vm.start()

    def _go_home(self) -> None:
        if self._active_det_vm is not None:
            self._active_det_vm.stop()
            self._active_det_vm = None
        self._home_stack.setCurrentWidget(self._splash)
        self.switchTo(self._home_stack)

    def _push_view(self, view: QWidget) -> None:
        """Switch to view inside the Home stacked widget without touching the sidebar."""
        # Clean up existing instance with same objectName in the stack
        for i in range(self._home_stack.count() - 1, -1, -1):
            w = self._home_stack.widget(i)
            if w is not None and w is not self._splash and w.objectName() == view.objectName():
                self._home_stack.removeWidget(w)
                w.deleteLater()
        self._home_stack.addWidget(view)
        self._home_stack.setCurrentWidget(view)
        self.switchTo(self._home_stack)

    def closeEvent(self, event) -> None:
        """Ensure threads and camera workers are cleanly shut down on exit."""
        if self._active_det_vm is not None:
            self._active_det_vm.stop()
            self._active_det_vm = None
        super().closeEvent(event)

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _get_model_entry(self, name: str) -> ModelEntry | None:
        return next((e for e in ModelRegistry.all_entries() if e.name == name), None)

    def _show_error(self, msg: str) -> None:
        from qfluentwidgets import InfoBar, InfoBarPosition
        InfoBar.error(
            title="Error",
            content=msg,
            position=InfoBarPosition.TOP,
            parent=self,
            duration=5000,
        )
