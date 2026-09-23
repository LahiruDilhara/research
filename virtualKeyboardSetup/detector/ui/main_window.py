"""
ui/main_window.py

FluentWindow shell for the detector application.
Implements clean MVVM and SOLID separation:
1. Startup: Full-window file landing screen with zero sidebars. File-only selection.
2. Workspace: Sidebar navigation with Play Mode (Testing HUD), Run Mode (Headless Production),
   Key Editor, and Settings. Mode, camera, and model selections live inside the workspace views.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, Slot
from PySide6.QtWidgets import QWidget
from qfluentwidgets import (
    FluentIcon,
    FluentWindow,
    InfoBar,
    InfoBarPosition,
    NavigationItemPosition,
)

from config.app_config import AppConfig
from config.constants import UI_BG_DARK
from core.interfaces.touch_model import ModelEntry, ModelRegistry
from core.layout.layout_parser import LayoutData
from services.action_config_service import ActionConfigService
from ui.views.action_config_view import ActionConfigView
from ui.views.file_landing_view import FileLandingView
from ui.views.play_mode_view import PlayModeView
from ui.views.run_mode_view import RunModeView
from ui.views.settings_view import SettingsView
from viewmodels.action_config_viewmodel import ActionConfigViewModel
from viewmodels.detector_viewmodel import DetectorViewModel, ExecutionMode
from viewmodels.settings_viewmodel import SettingsViewModel
from viewmodels.startup_viewmodel import StartupViewModel
from utils.logger import setup_logger

logger = setup_logger("MainWindow")


class MainWindow(FluentWindow):
    """Application shell with full-window startup and sidebar-driven workspace."""

    def __init__(self, config: AppConfig) -> None:
        super().__init__()
        self._config = config
        self._active_det_vm: DetectorViewModel | None = None
        self._action_vm: ActionConfigViewModel | None = None
        self._workspace_built: bool = False
        self._env_path = str(Path(__file__).resolve().parent.parent / ".env")

        # ViewModels
        plugins_dir = str(Path(__file__).resolve().parent.parent / config.plugins_dir)
        self._startup_vm = StartupViewModel(plugins_dir, parent=self)
        self._settings_vm = SettingsViewModel(
            env_path=self._env_path,
            config=self._config,
            parent=self,
        )

        self._setup_window()
        self._setup_landing_page()

        # Discover AI model plugins
        self._startup_vm.discover_models()

    # ── Window Setup ───────────────────────────────────────────────────────────

    def _setup_window(self) -> None:
        self.setWindowTitle(self._config.app_title)
        self.setMinimumSize(1100, 720)
        self.resize(1280, 800)
        self.setStyleSheet(
            f"FluentWindow {{ background-color: {UI_BG_DARK}; }}"
        )
        self.navigationInterface.setAcrylicEnabled(False)

    # ── Startup Landing View (No Sidebars) ────────────────────────────────────

    def _setup_landing_page(self) -> None:
        # Hide sidebar navigation completely on application launch
        self.navigationInterface.hide()

        self._landing_view = FileLandingView(
            self._startup_vm,
            last_xml_path=self._config.last_xml_path,
            parent=self,
        )
        self._landing_view.setObjectName("fileLandingView")
        self._landing_view.workspace_entered.connect(self._on_enter_workspace)

        self.stackedWidget.addWidget(self._landing_view)
        self.stackedWidget.setCurrentWidget(self._landing_view)

    # ── Workspace Entry ────────────────────────────────────────────────────────

    def _on_enter_workspace(
        self,
        layout_data: LayoutData,
        xml_path: str,
    ) -> None:
        # Persist layout path
        self._config.set_last_xml_path(xml_path)
        self._settings_vm.set_layout_xml_path(xml_path)

        # Stop previous detector if running
        if self._active_det_vm is not None:
            self._active_det_vm.stop()
            self._active_det_vm = None

        # Load action configuration
        svc = ActionConfigService()
        btn_ids = [b.id for b in layout_data.buttons]
        action_config = svc.load(xml_path, btn_ids)

        # Choose best default model from registry
        models = ModelRegistry.all_entries()
        default_model = None
        for m in models:
            if "lstm" in m.name.lower() or "best" in m.name.lower():
                default_model = m
                break
        if default_model is None and models:
            default_model = models[0]

        # Resolve best available camera index
        from services.camera_discovery import discover_cameras
        available_cams = discover_cameras(max_index=6)
        active_cam_idx = self._config.camera_index
        available_indices = [c.index for c in available_cams]
        if active_cam_idx not in available_indices and available_cams:
            active_cam_idx = available_cams[0].index
            logger.info(
                "Camera %d not in active devices. Auto-selected Camera %d (%s).",
                self._config.camera_index,
                active_cam_idx,
                available_cams[0].name,
            )
            self._config.set_camera_index(active_cam_idx)

        # Initialize Detector ViewModel in Play Mode
        self._active_det_vm = DetectorViewModel(
            layout=layout_data,
            action_config=action_config,
            config=self._config,
            camera_index=active_cam_idx,
            parent=self,
        )
        if default_model is not None:
            self._active_det_vm.set_model(default_model)
        self._active_det_vm.set_execution_mode(ExecutionMode.PLAY)

        # Build workspace interface views
        if not self._workspace_built:
            self._build_workspace(layout_data, xml_path)
            self._workspace_built = True
        else:
            self._update_workspace_layout(layout_data, xml_path)

        # Reveal sidebar navigation
        self.navigationInterface.show()

        # Switch to Play Mode
        self.switchTo(self._play_view)

        # Start detector engine
        self._active_det_vm.start()

    # ── Workspace Interface Construction ───────────────────────────────────────

    def _build_workspace(self, layout_data: LayoutData, xml_path: str) -> None:
        # 1. Play Mode View
        self._play_view = PlayModeView(self._active_det_vm, parent=self)
        self._play_view.setObjectName("playModeView")
        self._play_view.set_layout_data(layout_data)
        self._play_view.mode_switch_requested.connect(self._on_mode_switch_requested)
        self.addSubInterface(
            self._play_view,
            FluentIcon.PLAY,
            "Play Mode",
            position=NavigationItemPosition.TOP,
        )

        # 2. Run Mode View
        self._run_view = RunModeView(self._active_det_vm, parent=self)
        self._run_view.setObjectName("runModeView")
        self._run_view.set_layout_data(layout_data)
        self._run_view.set_touch_threshold(self._config.touch_threshold)
        self._run_view.mode_switch_requested.connect(self._on_mode_switch_requested)
        self.addSubInterface(
            self._run_view,
            FluentIcon.SPEED_HIGH,
            "Run Mode",
            position=NavigationItemPosition.TOP,
        )

        # 3. Key Editor View
        self._action_vm = ActionConfigViewModel(layout_data, xml_path, parent=self)
        self._key_editor_view = ActionConfigView(self._action_vm, parent=self)
        self._key_editor_view.setObjectName("keyEditorView")
        self._key_editor_view.btn_back.setVisible(False)
        self._key_editor_view.btn_continue.setText("Apply Actions")
        self._key_editor_view.btn_continue.clicked.connect(self._on_apply_actions)
        self._action_vm.action_changed.connect(self._on_action_item_changed)
        self._action_vm.config_saved.connect(self._on_actions_saved)
        self.addSubInterface(
            self._key_editor_view,
            FluentIcon.EDIT,
            "Key Editor",
            position=NavigationItemPosition.TOP,
        )

        # 4. Settings View (Bottom)
        self._settings_view = SettingsView(self._settings_vm, parent=self)
        self._settings_view.setObjectName("settingsView")
        self._settings_view.settings_applied.connect(self._on_settings_applied)
        self.addSubInterface(
            self._settings_view,
            FluentIcon.SETTING,
            "Settings",
            position=NavigationItemPosition.BOTTOM,
        )

        # 5. Change Layout Action (Bottom)
        self.navigationInterface.addItem(
            routeKey="changeLayoutAction",
            icon=FluentIcon.FOLDER,
            text="Change Layout",
            onClick=self._return_to_landing,
            selectable=False,
            position=NavigationItemPosition.BOTTOM,
        )

        # Track interface transitions for execution mode synchronization
        self.stackedWidget.currentChanged.connect(self._on_interface_changed)

    def _update_workspace_layout(self, layout_data: LayoutData, xml_path: str) -> None:
        """Update existing workspace views with a newly opened layout XML."""
        if hasattr(self, "_play_view"):
            self._play_view.set_layout_data(layout_data)
        if hasattr(self, "_run_view"):
            self._run_view.set_layout_data(layout_data)
        if hasattr(self, "_key_editor_view"):
            self._action_vm = ActionConfigViewModel(layout_data, xml_path, parent=self)
            self._action_vm.action_changed.connect(self._on_action_item_changed)
            self._action_vm.config_saved.connect(self._on_actions_saved)

    # ── Mode & Settings Synchronization ────────────────────────────────────────

    def _on_mode_switch_requested(self, target_mode: str) -> None:
        if target_mode == "run":
            self.switchTo(self._run_view)
        else:
            self.switchTo(self._play_view)

    @Slot(int)
    def _on_interface_changed(self, index: int) -> None:
        if self._active_det_vm is None:
            return
        widget = self.stackedWidget.widget(index)
        if widget is self._run_view:
            self._active_det_vm.set_execution_mode(ExecutionMode.RUN)
        elif widget is self._play_view:
            self._active_det_vm.set_execution_mode(ExecutionMode.PLAY)

    def _on_action_item_changed(self, button_id: str, action_type: str, value: str) -> None:
        if self._active_det_vm is not None and self._action_vm is not None:
            self._active_det_vm.set_action_config(self._action_vm.config)

    def _on_apply_actions(self) -> None:
        if self._action_vm is not None and self._active_det_vm is not None:
            self._action_vm.save()
            self._active_det_vm.set_action_config(self._action_vm.config)
            InfoBar.success(
                title="Actions Applied",
                content="Updated key actions are now active in the detector.",
                position=InfoBarPosition.TOP,
                parent=self,
                duration=3000,
            )

    def _on_actions_saved(self, path: str) -> None:
        if self._action_vm is not None and self._active_det_vm is not None:
            self._active_det_vm.set_action_config(self._action_vm.config)

    def _on_settings_applied(self) -> None:
        if self._active_det_vm is not None:
            self._active_det_vm.update_settings(self._config)
        if hasattr(self, "_run_view"):
            self._run_view.set_touch_threshold(self._config.touch_threshold)
        logger.info("Propagated settings update to active detector pipeline.")

    # ── Return to Landing Page ─────────────────────────────────────────────────

    def _return_to_landing(self) -> None:
        if self._active_det_vm is not None:
            self._active_det_vm.stop()
            self._active_det_vm = None
        self.navigationInterface.hide()
        self.switchTo(self._landing_view)

    # ── Close Event ────────────────────────────────────────────────────────────

    def closeEvent(self, event) -> None:
        if self._active_det_vm is not None:
            self._active_det_vm.stop()
            self._active_det_vm = None
        super().closeEvent(event)
