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
            plugins_dir="custom_plugins",
        )

        assert success is True
        assert len(saved_messages) == 1
        assert "Settings saved" in saved_messages[0]

        # Verify AppConfig was updated in memory
        assert abs(config.fingertip_velocity_threshold - 0.15) < 1e-4

        # Verify file on disk
        service = SettingsService(temp_path)
        entries = service.load_raw_entries()
        assert entries.get("FINGERTIP_VELOCITY_THRESHOLD") == "0.1500"
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

        # 3. Verify SettingsViewModel loads and persists with XML
        config = AppConfig(env_path=env_path)
        vm = SettingsViewModel(env_path=env_path, config=config)
        vm.set_layout_xml_path(str(xml_path))

        assert abs(config.touch_threshold - 0.65) < 1e-4
        assert abs(config.fingertip_velocity_threshold - 0.2200) < 1e-4

        # Save new values through ViewModel
        vm.save_settings(
            fps=20.0,
            touch_threshold=0.70,
            velocity_threshold=0.35,
            plugins_dir="ai_model_plugins",
        )

        reloaded = service.load_xml_settings(xml_path)
        assert reloaded.get("TARGET_FPS") == "20.0"
        assert reloaded.get("TOUCH_THRESHOLD") == "0.70"
        assert reloaded.get("FINGERTIP_VELOCITY_THRESHOLD") == "0.3500"
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


if __name__ == "__main__":
    test_settings_service_roundtrip()
    test_settings_viewmodel_save_and_signals()
    test_ui_input_box_dimensions_and_0_15_acceptance()
    test_settings_xml_persistence_and_preservation()
    test_detector_viewmodel_live_settings_update()
    print("\nAll 5 Settings MVVM, XML persistence, and live update tests passed successfully!")
