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

    def test_unique_marker_ids_allocation(self):
        # Adding multiple custom markers guarantees all assigned Tag IDs are unique
        self.vm.add_custom_marker()
        self.vm.add_custom_marker()
        self.vm.add_custom_marker()

        assigned_ids = [m.id for m in self.vm.layout.custom_markers]
        self.assertEqual(len(assigned_ids), len(set(assigned_ids)))

        # Also verify no conflict with outer perimeter marker IDs
        from core.geometry.marker_generator import generate_marker_layout
        outer_ids = {m.id for m in generate_marker_layout(self.config)}
        for m_id in assigned_ids:
            self.assertNotIn(m_id, outer_ids)


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

    def test_load_xml_restores_all_configured_settings(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            xml_path = str(Path(tmpdir) / "configured_settings_test.xml")
            self.config.button_stroke_width_mm = 2.8
            self.config.button_corner_radius_mm = 4.5
            self.config.button_min_gap_mm = 12.0
            self.config.grid_size_mm = 8.0
            self.config.paper_margin_mm = 15.0
            self.config.marker_size_mm = 20.0
            self.config.show_outer_markers = False

            self.vm.add_button()
            self.vm.save_project_xml(xml_path)

            # Load into fresh ViewModel with default AppConfig
            fresh_config = AppConfig()
            vm_loaded = DesignerViewModel(fresh_config)
            vm_loaded.load_project_xml(xml_path)

            # Verify that fresh_config was fully updated with loaded settings
            self.assertAlmostEqual(fresh_config.button_stroke_width_mm, 2.8)
            self.assertAlmostEqual(fresh_config.button_corner_radius_mm, 4.5)
            self.assertAlmostEqual(fresh_config.button_min_gap_mm, 12.0)
            self.assertAlmostEqual(fresh_config.grid_size_mm, 8.0)
            self.assertAlmostEqual(fresh_config.paper_margin_mm, 15.0)
            self.assertAlmostEqual(fresh_config.marker_size_mm, 20.0)
            self.assertFalse(fresh_config.show_outer_markers)

    def test_xml_all_button_locations_and_coordinates_persisted(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            xml_path = str(Path(tmpdir) / "coordinates_test.xml")
            self.vm.update_project_name("DAW Studio Controller")
            self.vm.add_button()
            self.vm.add_custom_marker()

            btn = self.vm.layout.buttons[0]
            btn.x_mm = 55.5
            btn.y_mm = 65.5
            btn.width_mm = 40.0
            btn.height_mm = 30.0
            btn.text = "Play Pause"

            m = self.vm.layout.custom_markers[0]
            m.x_mm = 120.0
            m.y_mm = 140.0

            self.vm.save_project_xml(xml_path)

            # Load into fresh ViewModel
            vm_loaded = DesignerViewModel(AppConfig())
            vm_loaded.load_project_xml(xml_path)

            self.assertEqual(vm_loaded.layout.project_name, "DAW Studio Controller")
            self.assertEqual(len(vm_loaded.layout.buttons), 1)
            loaded_btn = vm_loaded.layout.buttons[0]
            self.assertAlmostEqual(loaded_btn.x_mm, 55.5)
            self.assertAlmostEqual(loaded_btn.y_mm, 65.5)
            self.assertAlmostEqual(loaded_btn.width_mm, 40.0)
            self.assertAlmostEqual(loaded_btn.height_mm, 30.0)
            self.assertEqual(loaded_btn.text, "Play Pause")

            self.assertEqual(len(vm_loaded.layout.custom_markers), 1)
            loaded_m = vm_loaded.layout.custom_markers[0]
            self.assertAlmostEqual(loaded_m.x_mm, 120.0)
            self.assertAlmostEqual(loaded_m.y_mm, 140.0)

    def test_add_button_on_loaded_project_creates_new_unique_button(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            xml_path = str(Path(tmpdir) / "counter_test.xml")
            # Create a layout with 3 buttons
            self.vm.add_button()  # btn_1
            self.vm.add_button()  # btn_2
            self.vm.add_button()  # btn_3
            self.vm.save_project_xml(xml_path)

            # Load into fresh ViewModel
            vm_loaded = DesignerViewModel(AppConfig())
            vm_loaded.load_project_xml(xml_path)

            self.assertEqual(len(vm_loaded.layout.buttons), 3)

            # Add a 4th button
            vm_loaded.add_button()
            self.assertEqual(len(vm_loaded.layout.buttons), 4)
            added_btn = vm_loaded.layout.buttons[3]
            self.assertEqual(added_btn.id, "btn_4")

            # Verify existing buttons were not overwritten or moved
            self.assertEqual(vm_loaded.layout.buttons[0].id, "btn_1")
            self.assertEqual(vm_loaded.layout.buttons[1].id, "btn_2")
            self.assertEqual(vm_loaded.layout.buttons[2].id, "btn_3")

    def test_custom_marker_overlap_and_gap_validation(self):
        from core.geometry.layout_geometry import validate_layout_geometry
        from core.models.marker_model import MarkerModel

        # Create two overlapping custom markers
        self.vm.layout.use_custom_markers = True
        m1 = MarkerModel(id=20, x_mm=50.0, y_mm=50.0, size_mm=15.0)
        m2 = MarkerModel(id=21, x_mm=55.0, y_mm=50.0, size_mm=15.0)  # Overlaps m1
        self.vm.layout.custom_markers = [m1, m2]

        is_valid, msg = validate_layout_geometry(self.vm.layout, self.config)
        self.assertFalse(is_valid)
        self.assertIn("overlaps or is too close", msg)

        # Separate markers with proper gap (> 2.0 mm gap)
        m2.x_mm = 70.0  # m1 right edge = 57.5, m2 left edge = 62.5 -> gap = 5.0 mm
        is_valid, msg = validate_layout_geometry(self.vm.layout, self.config)
        self.assertTrue(is_valid)
        self.assertEqual(msg, "")

    def test_button_overlap_and_gap_validation(self):
        from core.geometry.layout_geometry import validate_layout_geometry
        from core.models.button_model import ButtonModel

        # Create two buttons touching on border (b1 right edge = 60.0, b2 left edge = 60.0 -> gap = 0 mm)
        b1 = ButtonModel(id="btn_1", x_mm=35.0, y_mm=35.0, width_mm=25.0, height_mm=18.0)
        b2 = ButtonModel(id="btn_2", x_mm=60.0, y_mm=35.0, width_mm=25.0, height_mm=18.0)
        self.vm.layout.buttons = [b1, b2]

        is_valid, msg = validate_layout_geometry(self.vm.layout, self.config)
        self.assertFalse(is_valid)
        self.assertIn("overlaps or is too close", msg)

        # Separate buttons with required gap (gap >= config.button_min_gap_mm)
        b2.x_mm = 72.0  # gap = 12.0 mm >= 10.0 mm
        is_valid, msg = validate_layout_geometry(self.vm.layout, self.config)
        self.assertTrue(is_valid)
        self.assertEqual(msg, "")




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
