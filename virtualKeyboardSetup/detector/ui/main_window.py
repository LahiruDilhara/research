"""
ui/main_window.py

FluentWindow shell for the detector app.

Navigation
──────────
  Splash (startup) → shown as full-screen overlay over the home view.
  Home             → startup splash view (overlaid).
  Settings         → settings view.

Routing (internal stack widget transitions)
──────────────────────────────────────────
  splash "Configure Actions"  → action_config_view
  splash "Quick Start"        → camera_select_view
  action_config "Back"        → splash
  action_config "Start"       → camera_select_view
  camera_select "Back"        → previous view
  camera_select "Start"       → detector_view
  detector_view "Stop"        → splash (resets state)
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QFileDialog, QStackedWidget, QVBoxLayout, QWidget
from qfluentwidgets import FluentIcon, FluentWindow, NavigationItemPosition

from config.app_config import AppConfig
from config.constants import UI_BG_DARK, UI_ACCENT
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
    """Application shell — dark FluentWindow with stacked view routing."""

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
        self._startup_vm.discover_models()

    # ── Window setup ───────────────────────────────────────────────────────────

    def _setup_window(self) -> None:
        self.setWindowTitle(self._config.app_title)
        self.setMinimumSize(1100, 720)
        self.resize(1280, 800)
        # Dark background on the window
        self.setStyleSheet(
            f"FluentWindow {{ background-color: {UI_BG_DARK}; }}"
        )
        self.navigationInterface.setAcrylicEnabled(False)

    # ── Pages ──────────────────────────────────────────────────────────────────

    def _build_pages(self) -> None:
        # ── Home / Splash page ─────────────────────────────────────────────────
        self._splash = SplashOverlayWidget()
        self._splash.setObjectName("homeView")
        self._splash.xml_browse_requested.connect(self._on_browse_xml)
        self._splash.configure_actions_requested.connect(self._on_configure_requested)
        self._splash.quick_start_requested.connect(self._on_quick_start)

        self.addSubInterface(
            self._splash,
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
        self._startup_vm.layout_loaded.connect(
            lambda layout: self._splash.set_layout_loaded(layout, self._startup_vm.layout_path)
        )
        self._startup_vm.models_ready.connect(self._splash.set_model_entries)
        self._startup_vm.error_occurred.connect(self._show_error)

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
        view = ActionConfigView(vm)
        view.setObjectName(f"actionConfigView")
        view.back_requested.connect(lambda: self._go_home())
        view.start_requested.connect(
            lambda: self._go_to_camera_select(layout_data, xml_path, vm.config)
        )
        self._push_view(view, "Configure Actions", FluentIcon.EDIT)

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
        cam_view = CameraSelectView(cam_vm)
        cam_view.setObjectName("cameraSelectView")
        cam_view.back_requested.connect(lambda: self._go_home())
        cam_view.start_requested.connect(
            lambda cam_idx: self._start_detector(layout_data, action_config, cam_idx)
        )
        self._push_view(cam_view, "Select Camera", FluentIcon.CAMERA)

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

        det_view = DetectorView(det_vm, self._current_model_entry)
        det_view.setObjectName("detectorView")
        det_view.set_layout_data(layout_data)
        det_view.stop_requested.connect(self._go_home)

        self._push_view(det_view, "Detector", FluentIcon.PLAY)
        det_vm.start()

    def _go_home(self) -> None:
        if self._active_det_vm is not None:
            self._active_det_vm.stop()
            self._active_det_vm = None
        self.stackedWidget.setCurrentWidget(self._splash)
        self.navigationInterface.setCurrentItem(self._splash.objectName())

    def _push_view(self, view: QWidget, label: str, icon) -> None:
        """Add a transient sub-interface and navigate to it."""
        # If already added (same objectName), reuse it
        for i in range(self.stackedWidget.count()):
            w = self.stackedWidget.widget(i)
            if w is None:
                continue
            if w.objectName() == view.objectName():
                self.stackedWidget.removeWidget(w)
                w.deleteLater()
        self.addSubInterface(view, icon, label)
        self.switchTo(view)

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

