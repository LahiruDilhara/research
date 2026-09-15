"""
Unit Tests for MVVM ViewModels.
Tests DesignerViewModel and SettingsViewModel signal emission, state management,
button/marker CRUD, XML project persistence, and settings updates.
"""

import unittest
import tempfile
from pathlib import Path
from PySide6.QtWidgets import QApplication

from config.app_config import AppConfig
from viewmodels.designer_viewmodel import DesignerViewModel
from viewmodels.settings_viewmodel import SettingsViewModel

# Ensure single QCoreApplication / QApplication instance for PySide6 signal tests
app = QApplication.instance() or QApplication([])


class TestDesignerViewModel(unittest.TestCase):
    def setUp(self):
        self.config = AppConfig()
        self.vm = DesignerViewModel(self.config)

    def test_add_button_emits_signal(self):
        added_buttons = []
        self.vm.button_added.connect(added_buttons.append)

        self.vm.add_button()
        self.assertEqual(len(added_buttons), 1)
        self.assertEqual(len(self.vm.layout.buttons), 1)
        self.assertEqual(self.vm.layout.buttons[0].id, "btn_1")

    def test_add_custom_marker_emits_signal(self):
        added_markers = []
        self.vm.marker_added.connect(added_markers.append)

        self.vm.add_custom_marker()
        self.assertEqual(len(added_markers), 1)
        self.assertEqual(len(self.vm.layout.custom_markers), 1)
        self.assertTrue(self.vm.layout.use_custom_markers)

    def test_save_and_load_xml_via_viewmodel(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            xml_path = str(Path(tmpdir) / "vm_test_layout.xml")
            self.vm.add_button()
            self.vm.add_custom_marker()

            self.vm.save_project_xml(xml_path)
            self.assertTrue(Path(xml_path).exists())

            # Load into fresh ViewModel
            vm_loaded = DesignerViewModel(self.config)
            vm_loaded.load_project_xml(xml_path)

            self.assertEqual(len(vm_loaded.layout.buttons), 1)
            self.assertEqual(len(vm_loaded.layout.custom_markers), 1)

    def test_save_xml_contains_all_constants_and_details(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            xml_path = str(Path(tmpdir) / "full_details_layout.xml")
            self.config.button_stroke_width_mm = 2.5
            self.config.button_corner_radius_mm = 3.0
            self.config.button_min_gap_mm = 12.0
            self.config.grid_size_mm = 10.0

            self.vm.add_button()
            self.vm.save_project_xml(xml_path)

            xml_content = Path(xml_path).read_text(encoding="utf-8")
            self.assertIn("<SystemConfig", xml_content)
            self.assertIn("<PaperMargin", xml_content)
            self.assertIn("<MarkerRingZone", xml_content)
            self.assertIn("<InteriorRegion", xml_content)
            self.assertIn("<ButtonStylingRules", xml_content)
            self.assertIn("<GridConfig", xml_content)
            self.assertIn("<RuntimePipelineSpecs", xml_content)
            self.assertIn('target_camera_fps="12"', xml_content)
            self.assertIn('touch_model_architecture="LSTM"', xml_content)

            # Load into fresh ViewModel and verify settings restored
            vm_loaded = DesignerViewModel(AppConfig())
            vm_loaded.load_project_xml(xml_path)
            self.assertEqual(vm_loaded.config.button_stroke_width_mm, 2.5)
            self.assertEqual(vm_loaded.config.button_corner_radius_mm, 3.0)
            self.assertEqual(vm_loaded.config.button_min_gap_mm, 12.0)
            self.assertEqual(vm_loaded.config.grid_size_mm, 10.0)


class TestSettingsViewModel(unittest.TestCase):
    def setUp(self):
        self.config = AppConfig()
        self.vm = SettingsViewModel(self.config)

    def test_apply_settings_emits_signal(self):
        dim_events = []
        self.vm.paper_dimensions_changed.connect(lambda w, h: dim_events.append((w, h)))

        self.vm.apply_settings(420.0, 297.0, 20.0, True, True, 10.0)
        self.assertEqual(len(dim_events), 1)
        self.assertEqual(dim_events[0], (420.0, 297.0))
        self.assertEqual(self.config.paper_width_mm, 420.0)
        self.assertEqual(self.config.paper_height_mm, 297.0)
        self.assertEqual(self.config.marker_size_mm, 20.0)
        self.assertTrue(self.config.show_outer_markers)


if __name__ == "__main__":
    unittest.main()
