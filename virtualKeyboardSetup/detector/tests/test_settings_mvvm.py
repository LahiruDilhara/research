"""
tests/test_settings_mvvm.py

Unit tests verifying MVVM architecture and SOLID principles for Settings:
- SettingsService isolates file I/O operations (Single Responsibility Principle).
- SettingsViewModel encapsulates configuration state and coordinates updates.
- Input validation supports precise decimals and values such as 0.15 without truncation.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from PySide6.QtWidgets import QApplication
from qfluentwidgets import DoubleSpinBox

from config.app_config import AppConfig
from services.settings_service import SettingsService
from viewmodels.settings_viewmodel import SettingsViewModel
from ui.views.settings_view import SettingsView


def _get_or_create_qapp() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    return app


def test_settings_service_roundtrip():
    """Verify SettingsService reads and writes settings in .env format."""
    with tempfile.NamedTemporaryFile("w+", suffix=".env", delete=False) as f:
        f.write("TARGET_FPS=12.0\nTOUCH_THRESHOLD=0.55\n")
        temp_path = Path(f.name)

    try:
        service = SettingsService(temp_path)
        raw = service.load_raw_entries()
        assert raw.get("TARGET_FPS") == "12.0"
        assert raw.get("TOUCH_THRESHOLD") == "0.55"

        service.save_settings({
            "TARGET_FPS": "24.0",
            "FINGERTIP_VELOCITY_THRESHOLD": "0.1500",
        })

        updated = service.load_raw_entries()
        assert updated.get("TARGET_FPS") == "24.0"
        assert updated.get("TOUCH_THRESHOLD") == "0.55"
        assert updated.get("FINGERTIP_VELOCITY_THRESHOLD") == "0.1500"
    finally:
        temp_path.unlink(missing_ok=True)


def test_settings_viewmodel_save_and_signals():
    """Verify SettingsViewModel emits signals and updates AppConfig."""
    with tempfile.NamedTemporaryFile("w+", suffix=".env", delete=False) as f:
        f.write("FINGERTIP_VELOCITY_THRESHOLD=0.008\n")
        temp_path = Path(f.name)

    try:
        config = AppConfig(env_path=temp_path)
        vm = SettingsViewModel(env_path=temp_path, config=config)

        saved_messages: list[str] = []
        vm.settings_saved.connect(saved_messages.append)

        success = vm.save_settings(
            fps=15.0,
            touch_threshold=0.60,
            velocity_threshold=0.15,
            hand_movement_threshold=0.1800,
            detection_confidence=0.65,
            presence_confidence=0.60,
            tracking_confidence=0.70,
            quality_min_avg=0.75,
            quality_min_frame=0.50,
            quality_max_drop=0.30,
            plugins_dir="custom_plugins",
        )

        assert success is True
        assert len(saved_messages) == 1
        assert "Settings saved" in saved_messages[0]

        # Verify AppConfig was updated in memory
        assert abs(config.fingertip_velocity_threshold - 0.15) < 1e-4
        assert abs(config.hand_movement_threshold - 0.1800) < 1e-4
        assert abs(config.mediapipe_min_detection_confidence - 0.65) < 1e-4
        assert abs(config.quality_min_avg_score - 0.75) < 1e-4

        # Verify file on disk
        service = SettingsService(temp_path)
        entries = service.load_raw_entries()
        assert entries.get("FINGERTIP_VELOCITY_THRESHOLD") == "0.1500"
        assert entries.get("HAND_MOVEMENT_THRESHOLD") == "0.1800"
        assert entries.get("MEDIAPIPE_MIN_DETECTION_CONFIDENCE") == "0.65"
        assert entries.get("QUALITY_MIN_AVG_SCORE") == "0.75"
        assert entries.get("TARGET_FPS") == "15.0"
        assert entries.get("AI_MODEL_PLUGINS_DIR") == "custom_plugins"
    finally:
        temp_path.unlink(missing_ok=True)


def test_ui_input_box_dimensions_and_0_15_acceptance():
    """Verify input boxes are comfortably sized and accept 0.15 and multi-decimal inputs."""
    _get_or_create_qapp()

    with tempfile.NamedTemporaryFile("w+", suffix=".env", delete=False) as f:
        f.write("FINGERTIP_VELOCITY_THRESHOLD=0.008\n")
        temp_path = Path(f.name)

    try:
        config = AppConfig(env_path=temp_path)
        vm = SettingsViewModel(env_path=temp_path, config=config)
        view = SettingsView(vm)

        # Check ergonomic sizing
        assert view.fingertip_vel_spin.width() >= 180
        assert view.fingertip_vel_spin.height() >= 30
        assert view.hand_movement_spin.width() >= 180
        assert view.mp_detection_conf_spin.width() >= 180
        assert view.quality_min_avg_spin.width() >= 180
        assert view.touch_threshold_spin.width() >= 180
        assert view.target_fps_spin.width() >= 180
        assert view.plugins_dir_edit.width() >= 300

        # Verify that 0.15 is valid and accepted
        spin = view.fingertip_vel_spin
        val_result, val_text, _ = spin.validate("0.15", 4)
        # In PySide6, QValidator.State.Acceptable is integer 2
        assert val_result.value == 2 or val_result == 2, f"Expected Acceptable, got {val_result}"

        # Test typing multiple decimals e.g. 0.0085, 0.1500
        val_result_multi, _, _ = spin.validate("0.1500", 6)
        assert val_result_multi.value == 2 or val_result_multi == 2

        # Verify value assignment
        spin.setValue(0.15)
        assert abs(spin.value() - 0.15) < 1e-5
    finally:
        temp_path.unlink(missing_ok=True)


def test_settings_xml_persistence_and_preservation():
    """Verify SettingsService saves and loads <DetectorSettings> in layout XML without corrupting other sections."""
    sample_xml = """<?xml version="1.0" ?>
<PaperLayout paper_width_mm="297.0" paper_height_mm="210.0">
  <DesignerLayout>
    <Buttons count="1">
      <Button id="btn_space" x_mm="30" y_mm="50" width_mm="60" height_mm="15">
        <Text>Space</Text>
      </Button>
    </Buttons>
  </DesignerLayout>
  <DetectorActions>
    <Actions count="1">
      <Action button_id="btn_space" label="Space" type="keystroke" value="space" />
    </Actions>
  </DetectorActions>
</PaperLayout>
"""
    with tempfile.NamedTemporaryFile("w+", suffix=".xml", delete=False) as f_xml:
        f_xml.write(sample_xml)
        xml_path = Path(f_xml.name)

    with tempfile.NamedTemporaryFile("w+", suffix=".env", delete=False) as f_env:
        f_env.write("TARGET_FPS=12.0\n")
        env_path = Path(f_env.name)

    try:
        service = SettingsService(env_path)
        service.save_settings(
            updates={
                "TARGET_FPS": "18.0",
                "TOUCH_THRESHOLD": "0.65",
                "FINGERTIP_VELOCITY_THRESHOLD": "0.2200",
                "HAND_MOVEMENT_THRESHOLD": "0.1600",
                "MEDIAPIPE_MIN_DETECTION_CONFIDENCE": "0.55",
                "QUALITY_MIN_AVG_SCORE": "0.70",
                "AI_MODEL_PLUGINS_DIR": "ai_model_plugins",
            },
            xml_path=xml_path,
        )

        # 1. Verify XML file contents
        xml_text = xml_path.read_text(encoding="utf-8")
        assert "<DetectorSettings" in xml_text
        assert "DesignerLayout" in xml_text
        assert "btn_space" in xml_text

        # 2. Verify loading back from XML
        loaded = service.load_xml_settings(xml_path)
        assert loaded.get("TARGET_FPS") == "18.0"
        assert loaded.get("TOUCH_THRESHOLD") == "0.65"
        assert loaded.get("FINGERTIP_VELOCITY_THRESHOLD") == "0.2200"
        assert loaded.get("HAND_MOVEMENT_THRESHOLD") == "0.1600"
        assert loaded.get("MEDIAPIPE_MIN_DETECTION_CONFIDENCE") == "0.55"

        # 3. Verify SettingsViewModel loads and persists with XML
        config = AppConfig(env_path=env_path)
        vm = SettingsViewModel(env_path=env_path, config=config)
        vm.set_layout_xml_path(str(xml_path))

        assert abs(config.touch_threshold - 0.65) < 1e-4
        assert abs(config.fingertip_velocity_threshold - 0.2200) < 1e-4
        assert abs(config.hand_movement_threshold - 0.1600) < 1e-4

        # Save new values through ViewModel
        vm.save_settings(
            fps=20.0,
            touch_threshold=0.70,
            velocity_threshold=0.35,
            hand_movement_threshold=0.1900,
            detection_confidence=0.60,
            presence_confidence=0.60,
            tracking_confidence=0.60,
            quality_min_avg=0.80,
            quality_min_frame=0.55,
            quality_max_drop=0.25,
            plugins_dir="ai_model_plugins",
        )

        reloaded = service.load_xml_settings(xml_path)
        assert reloaded.get("TARGET_FPS") == "20.0"
        assert reloaded.get("TOUCH_THRESHOLD") == "0.70"
        assert reloaded.get("FINGERTIP_VELOCITY_THRESHOLD") == "0.3500"
        assert reloaded.get("HAND_MOVEMENT_THRESHOLD") == "0.1900"
        assert reloaded.get("QUALITY_MIN_AVG_SCORE") == "0.80"
    finally:
        xml_path.unlink(missing_ok=True)
        env_path.unlink(missing_ok=True)


def test_detector_viewmodel_live_settings_update():
    """Verify live DetectorViewModel updates its TouchPipelineService thresholds immediately."""
    from core.layout.layout_parser import LayoutData

    with tempfile.NamedTemporaryFile("w+", suffix=".env", delete=False) as f_env:
        f_env.write("FINGERTIP_VELOCITY_THRESHOLD=0.008\nTOUCH_THRESHOLD=0.55\n")
        env_path = Path(f_env.name)

    try:
        config = AppConfig(env_path=env_path)
        layout = LayoutData(
            paper_width_mm=297,
            paper_height_mm=210,
            marker_size_mm=15,
            marker_family="tag36h11",
            buttons=[],
            markers=[],
        )

        from viewmodels.detector_viewmodel import DetectorViewModel

        det_vm = DetectorViewModel(
            layout=layout,
            action_config={},
            config=config,
            camera_index=0,
        )

        # Initial thresholds
        assert abs(det_vm._pipeline_service.velocity_threshold - 0.008) < 1e-4
        assert abs(det_vm._pipeline_service.touch_threshold - 0.55) < 1e-4

        # Simulate user saving new settings in UI
        config.set_fingertip_velocity_threshold(0.42)
        config.set_touch_threshold(0.75)

        # Live detector update
        det_vm.update_settings(config)

        # Assert live pipeline updated immediately without restarting
        assert abs(det_vm._pipeline_service.velocity_threshold - 0.42) < 1e-4
        assert abs(det_vm._pipeline_service.touch_threshold - 0.75) < 1e-4
    finally:
        env_path.unlink(missing_ok=True)


def test_settings_restore_defaults():
    """Verify SettingsViewModel and SettingsView restore defaults button resets all parameters."""
    _get_or_create_qapp()

    with tempfile.NamedTemporaryFile("w+", suffix=".env", delete=False) as f_env:
        # Prepopulate with customized/deviated values
        f_env.write(
            "TARGET_FPS=30.0\n"
            "TOUCH_THRESHOLD=0.85\n"
            "FINGERTIP_VELOCITY_THRESHOLD=0.2500\n"
            "HAND_MOVEMENT_THRESHOLD=0.4500\n"
            "QUALITY_MIN_AVG_SCORE=0.90\n"
        )
        env_path = Path(f_env.name)

    try:
        config = AppConfig(env_path=env_path)
        vm = SettingsViewModel(env_path=env_path, config=config)
        view = SettingsView(vm)

        # Confirm non-default initial values
        assert abs(vm.target_fps - 30.0) < 1e-4
        assert abs(vm.touch_threshold - 0.85) < 1e-4
        assert abs(vm.hand_movement_threshold - 0.4500) < 1e-4

        # Trigger restore defaults
        view._on_restore_defaults_clicked()

        # Verify values restored to canonical defaults in ViewModel and AppConfig
        assert abs(vm.target_fps - 12.0) < 1e-4
        assert abs(vm.touch_threshold - 0.55) < 1e-4
        assert abs(vm.hand_movement_threshold - 0.1550) < 1e-4
        assert abs(vm.fingertip_velocity_threshold - 0.0080) < 1e-4
        assert abs(vm.mediapipe_min_detection_confidence - 0.50) < 1e-4
        assert abs(vm.quality_min_avg_score - 0.65) < 1e-4
        assert abs(vm.quality_min_frame_score - 0.45) < 1e-4
        assert abs(vm.quality_max_score_drop - 0.35) < 1e-4

        # Verify view inputs refreshed
        assert abs(view.target_fps_spin.value() - 12.0) < 1e-4
        assert abs(view.touch_threshold_spin.value() - 55) < 1e-4
        assert abs(view.hand_movement_spin.value() - 0.1550) < 1e-4
        assert abs(view.fingertip_vel_spin.value() - 0.0080) < 1e-4

        # Verify persisted to disk
        service = SettingsService(env_path)
        entries = service.load_raw_entries()
        assert entries.get("TARGET_FPS") == "12.0"
        assert entries.get("TOUCH_THRESHOLD") == "0.55"
        assert entries.get("HAND_MOVEMENT_THRESHOLD") == "0.1550"
        assert entries.get("FINGERTIP_VELOCITY_THRESHOLD") == "0.0080"
        assert entries.get("ONE_EURO_ENABLED") == "true"
        assert entries.get("ONE_EURO_MIN_CUTOFF") == "0.02"
    finally:
        env_path.unlink(missing_ok=True)


def test_one_euro_filter_settings_persistence_and_detector_update():
    """Verify One Euro filter settings save, persist to disk, and update detector live."""
    _get_or_create_qapp()

    with tempfile.NamedTemporaryFile("w+", suffix=".env", delete=False) as f_env:
        f_env.write("ONE_EURO_ENABLED=false\nONE_EURO_MIN_CUTOFF=1.0\n")
        env_path = Path(f_env.name)

    try:
        config = AppConfig(env_path=env_path)
        vm = SettingsViewModel(env_path=env_path, config=config)
        view = SettingsView(vm)

        assert vm.one_euro_enabled is False
        assert abs(vm.one_euro_min_cutoff - 1.0) < 1e-4

        # Save new One Euro parameters
        success = vm.save_settings(
            fps=12.0,
            touch_threshold=0.55,
            velocity_threshold=0.008,
            hand_movement_threshold=0.155,
            detection_confidence=0.50,
            presence_confidence=0.50,
            tracking_confidence=0.50,
            quality_min_avg=0.65,
            quality_min_frame=0.45,
            quality_max_drop=0.35,
            plugins_dir="ai_model_plugins",
            one_euro_enabled=True,
            one_euro_min_cutoff=0.85,
            one_euro_beta=2.50,
            one_euro_d_cutoff=1.20,
        )
        assert success is True
        assert vm.one_euro_enabled is True
        assert abs(vm.one_euro_min_cutoff - 0.85) < 1e-4
        assert abs(vm.one_euro_beta - 2.50) < 1e-4
        assert abs(vm.one_euro_d_cutoff - 1.20) < 1e-4

        # Verify disk persistence
        service = SettingsService(env_path)
        raw = service.load_raw_entries()
        assert raw.get("ONE_EURO_ENABLED") == "true"
        assert raw.get("ONE_EURO_MIN_CUTOFF") == "0.85"
        assert raw.get("ONE_EURO_BETA") == "2.50"
        assert raw.get("ONE_EURO_D_CUTOFF") == "1.20"

        # Verify live detector pipeline synchronization
        from core.layout.layout_parser import LayoutData
        from viewmodels.detector_viewmodel import DetectorViewModel

        layout = LayoutData(
            paper_width_mm=297,
            paper_height_mm=210,
            marker_size_mm=15,
            marker_family="tag36h11",
            buttons=[],
            markers=[],
        )
        det_vm = DetectorViewModel(layout=layout, action_config={}, config=config, camera_index=0)
        pipe = det_vm._pipeline_service
        assert pipe.one_euro_filter.enabled is True
        assert abs(pipe.one_euro_filter.min_cutoff - 0.85) < 1e-4
        assert abs(pipe.one_euro_filter.beta - 2.50) < 1e-4
        assert abs(pipe.one_euro_filter.d_cutoff - 1.20) < 1e-4

        # Modify config and call update_settings
        config.set_one_euro_enabled(False)
        config.set_one_euro_min_cutoff(1.50)
        det_vm.update_settings(config)
        assert pipe.one_euro_filter.enabled is False
        assert abs(pipe.one_euro_filter.min_cutoff - 1.50) < 1e-4

        # Test view fields refresh
        view._refresh_fields()
        assert view.one_euro_enabled_switch.isChecked() is False
        assert abs(view.one_euro_min_cutoff_spin.value() - 1.50) < 1e-4
    finally:
        env_path.unlink(missing_ok=True)


if __name__ == "__main__":
    test_settings_service_roundtrip()
    test_settings_viewmodel_save_and_signals()
    test_ui_input_box_dimensions_and_0_15_acceptance()
    test_settings_xml_persistence_and_preservation()
    test_detector_viewmodel_live_settings_update()
    test_settings_restore_defaults()
    test_one_euro_filter_settings_persistence_and_detector_update()
    print("\nAll 7 Settings MVVM, XML persistence, One Euro filter, live update, and restore defaults tests passed successfully!")
