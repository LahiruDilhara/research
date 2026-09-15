"""
XML Project Repository Service.
Implements IProjectRepository for unified XML project file persistence and runtime export.
Saves complete project details, system constants, grid config, styling rules, runtime specs,
and marker/button geometry into structured XML.
"""

import xml.etree.ElementTree as ET
from xml.dom import minidom
from pathlib import Path

from config.app_config import AppConfig
from config.constants import (
    DEFAULT_PAPER_WIDTH_MM,
    DEFAULT_PAPER_HEIGHT_MM,
    PAPER_MARGIN_MM,
    MARKER_SIZE_MM,
    MARKER_SPACING_MM,
    DEFAULT_MARKER_MIN_GAP_RATIO,
    MARKER_MIN_GAP_MM,
    MARKER_FAMILY,
    BUTTON_STROKE_WIDTH_MM,
    BUTTON_CORNER_RADIUS_MM,
    BUTTON_MIN_WIDTH_MM,
    BUTTON_MIN_HEIGHT_MM,
    BUTTON_MIN_GAP_MM,
    DEFAULT_FONT_SIZE_PT,
    INTERIOR_BUFFER_MM,
    DEFAULT_GRID_SIZE_MM,
)
from core.interfaces.repository_interface import IProjectRepository
from core.models.paper_layout import PaperLayoutModel
from core.models.button_model import ButtonModel
from core.models.marker_model import MarkerModel
from core.geometry.marker_generator import generate_marker_layout


class XmlRepository(IProjectRepository):
    def __init__(self, config: AppConfig | None = None):
        self.config = config or AppConfig()

    def save(self, filepath_or_id: str | Path, layout: PaperLayoutModel) -> None:
        path = Path(filepath_or_id)

        # Active markers: either custom placed markers or auto perimeter markers
        if layout.use_custom_markers and layout.custom_markers:
            markers = layout.custom_markers
            use_custom = 1
        else:
            markers = generate_marker_layout(self.config)
            use_custom = 0

        # Derived ring and interior calculations
        p_margin = layout.paper_margin_mm
        m_size = layout.marker_size_mm
        buf = layout.interior_buffer_mm

        outer_x_min = p_margin
        outer_y_min = p_margin
        outer_x_max = layout.paper_width_mm - p_margin
        outer_y_max = layout.paper_height_mm - p_margin

        inner_x_min = p_margin + m_size
        inner_y_min = p_margin + m_size
        inner_x_max = layout.paper_width_mm - inner_x_min
        inner_y_max = layout.paper_height_mm - inner_y_min

        int_x_min = inner_x_min + buf
        int_y_min = inner_y_min + buf
        int_x_max = layout.paper_width_mm - int_x_min
        int_y_max = layout.paper_height_mm - int_y_min
        int_w = max(0.0, int_x_max - int_x_min)
        int_h = max(0.0, int_y_max - int_y_min)

        root = ET.Element("PaperLayout")
        root.set("project_name", layout.project_name)
        root.set("paper_width_mm", f"{layout.paper_width_mm:.3f}")
        root.set("paper_height_mm", f"{layout.paper_height_mm:.3f}")
        root.set("paper_margin_mm", f"{layout.paper_margin_mm:.3f}")
        root.set("marker_size_mm", f"{layout.marker_size_mm:.3f}")
        root.set("marker_margin_mm", f"{layout.paper_margin_mm:.3f}")
        root.set("marker_spacing_mm", f"{layout.marker_spacing_mm:.3f}")
        root.set("marker_family", layout.marker_family)
        root.set("marker_min_gap_ratio", f"{layout.marker_min_gap_ratio:.3f}")
        root.set("marker_min_gap_mm", f"{layout.marker_min_gap_mm:.3f}")
        root.set("use_custom_markers", str(use_custom))
        root.set("show_outer_markers", str(1 if layout.show_outer_markers else 0))
        root.set("coordinate_origin", "top_left_paper_corner (x_right, y_down)")
        root.set("units", "millimeters")

        # 1. System Config metadata
        sys_config_el = ET.SubElement(root, "SystemConfig")
        sys_config_el.set("project_name", layout.project_name)
        sys_config_el.set("app_title", "Virtual Keyboard Paper Layout Designer")
        sys_config_el.set("version", "1.0.0")
        sys_config_el.set("schema_version", "2.0")
        sys_config_el.set("research_suite", "Application 1 Layout Designer & Application 2 Runtime Engine")

        # 2. Paper Margin
        margin_el = ET.SubElement(root, "PaperMargin")
        margin_el.set("margin_mm", f"{p_margin:.3f}")
        margin_el.set("top_mm", f"{p_margin:.3f}")
        margin_el.set("bottom_mm", f"{p_margin:.3f}")
        margin_el.set("left_mm", f"{p_margin:.3f}")
        margin_el.set("right_mm", f"{p_margin:.3f}")

        # 3. Marker Ring Zone
        marker_zone_el = ET.SubElement(root, "MarkerRingZone")
        marker_zone_el.set("outer_x_min_mm", f"{outer_x_min:.3f}")
        marker_zone_el.set("outer_y_min_mm", f"{outer_y_min:.3f}")
        marker_zone_el.set("outer_x_max_mm", f"{outer_x_max:.3f}")
        marker_zone_el.set("outer_y_max_mm", f"{outer_y_max:.3f}")
        marker_zone_el.set("inner_x_min_mm", f"{inner_x_min:.3f}")
        marker_zone_el.set("inner_y_min_mm", f"{inner_y_min:.3f}")
        marker_zone_el.set("inner_x_max_mm", f"{inner_x_max:.3f}")
        marker_zone_el.set("inner_y_max_mm", f"{inner_y_max:.3f}")
        marker_zone_el.set("marker_size_mm", f"{m_size:.3f}")
        marker_zone_el.set("marker_family", layout.marker_family)
        marker_zone_el.set("marker_spacing_mm", f"{layout.marker_spacing_mm:.3f}")
        marker_zone_el.set("marker_min_gap_ratio", f"{layout.marker_min_gap_ratio:.3f}")
        marker_zone_el.set("marker_min_gap_mm", f"{layout.marker_min_gap_mm:.3f}")

        # 4. Interior Active Region
        interior_el = ET.SubElement(root, "InteriorRegion")
        interior_el.set("x_min_mm", f"{int_x_min:.3f}")
        interior_el.set("y_min_mm", f"{int_y_min:.3f}")
        interior_el.set("x_max_mm", f"{int_x_max:.3f}")
        interior_el.set("y_max_mm", f"{int_y_max:.3f}")
        interior_el.set("width_mm", f"{int_w:.3f}")
        interior_el.set("height_mm", f"{int_h:.3f}")
        interior_el.set("interior_buffer_mm", f"{buf:.3f}")

        # 5. Button Styling Rules & System Constants
        style_rules_el = ET.SubElement(root, "ButtonStylingRules")
        style_rules_el.set("button_stroke_width_mm", f"{layout.button_stroke_width_mm:.3f}")
        style_rules_el.set("button_corner_radius_mm", f"{layout.button_corner_radius_mm:.3f}")
        style_rules_el.set("button_min_width_mm", f"{layout.button_min_width_mm:.3f}")
        style_rules_el.set("button_min_height_mm", f"{layout.button_min_height_mm:.3f}")
        style_rules_el.set("button_min_gap_mm", f"{layout.button_min_gap_mm:.3f}")
        style_rules_el.set("default_font_size_pt", str(layout.default_font_size_pt))

        # 6. Grid Alignment Config
        grid_el = ET.SubElement(root, "GridConfig")
        grid_el.set("grid_snap_enabled", str(1 if layout.grid_snap_enabled else 0))
        grid_el.set("grid_size_mm", f"{layout.grid_size_mm:.3f}")

        # 7. Runtime Touch Detection Specs
        runtime_el = ET.SubElement(root, "RuntimePipelineSpecs")
        runtime_el.set("target_camera_fps", "12")
        runtime_el.set("temporal_window_frames", "5")
        runtime_el.set("temporal_window_stride", "3")
        runtime_el.set("mediapipe_num_hands", "1")
        runtime_el.set("mediapipe_landmarks_count", "21")
        runtime_el.set("feature_vector_size", "84")
        runtime_el.set("touch_model_architecture", "LSTM")
        runtime_el.set("end_to_end_latency_ms", "29.09")

        # 8. Markers Section
        markers_el = ET.SubElement(root, "Markers", count=str(len(markers)), use_custom_markers=str(use_custom))
        for m in markers:
            me = ET.SubElement(markers_el, "Marker")
            me.set("id", str(m.id))
            me.set("center_x_mm", f"{m.x_mm:.3f}")
            me.set("center_y_mm", f"{m.y_mm:.3f}")
            me.set("size_mm", f"{m.size_mm:.3f}")
            me.set("family", layout.marker_family)

            corners_el = ET.SubElement(me, "Corners")
            corner_names = ["TopLeft", "TopRight", "BottomRight", "BottomLeft"]
            for c_name, (cx, cy) in zip(corner_names, m.corners_mm):
                ce = ET.SubElement(corners_el, c_name)
                ce.set("x_mm", f"{cx:.3f}")
                ce.set("y_mm", f"{cy:.3f}")

        # 9. Buttons Section
        buttons_el = ET.SubElement(root, "Buttons", count=str(len(layout.buttons)))
        for b in layout.buttons:
            be = ET.SubElement(buttons_el, "Button")
            be.set("id", str(b.id))
            be.set("x_mm", f"{b.x_mm:.3f}")
            be.set("y_mm", f"{b.y_mm:.3f}")
            be.set("width_mm", f"{b.width_mm:.3f}")
            be.set("height_mm", f"{b.height_mm:.3f}")
            be.set("x_max_mm", f"{b.x_max_mm:.3f}")
            be.set("y_max_mm", f"{b.y_max_mm:.3f}")
            be.set("center_x_mm", f"{b.center_x_mm:.3f}")
            be.set("center_y_mm", f"{b.center_y_mm:.3f}")

            style_el = ET.SubElement(be, "Style")
            style_el.set("font_size_pt", str(b.font_size_pt))
            style_el.set("stroke_width_mm", f"{layout.button_stroke_width_mm:.3f}")
            style_el.set("corner_radius_mm", f"{layout.button_corner_radius_mm:.3f}")

            text_el = ET.SubElement(be, "Text")
            text_el.text = b.text

        rough_string = ET.tostring(root, "utf-8")
        pretty = minidom.parseString(rough_string).toprettyxml(indent="  ")

        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(pretty)

    def load(self, filepath_or_id: str | Path) -> PaperLayoutModel:
        path = Path(filepath_or_id)
        tree = ET.parse(path)
        root = tree.getroot()

        project_name = root.attrib.get("project_name", "My Paper Keyboard")
        paper_width = float(root.attrib.get("paper_width_mm", DEFAULT_PAPER_WIDTH_MM))
        paper_height = float(root.attrib.get("paper_height_mm", DEFAULT_PAPER_HEIGHT_MM))
        paper_margin = float(root.attrib.get("paper_margin_mm", root.attrib.get("marker_margin_mm", PAPER_MARGIN_MM)))
        marker_size = float(root.attrib.get("marker_size_mm", MARKER_SIZE_MM))
        marker_spacing = float(root.attrib.get("marker_spacing_mm", MARKER_SPACING_MM))
        marker_family = root.attrib.get("marker_family", MARKER_FAMILY)
        marker_min_gap_ratio = float(root.attrib.get("marker_min_gap_ratio", DEFAULT_MARKER_MIN_GAP_RATIO))
        marker_min_gap = float(root.attrib.get("marker_min_gap_mm", MARKER_MIN_GAP_MM))
        use_custom = bool(int(root.attrib.get("use_custom_markers", "0")))
        show_outer = bool(int(root.attrib.get("show_outer_markers", "1")))

        # Check MarkerRingZone for marker_min_gap_mm override
        mz_el = root.find("MarkerRingZone")
        if mz_el is not None and "marker_min_gap_mm" in mz_el.attrib:
            marker_min_gap = float(mz_el.attrib["marker_min_gap_mm"])

        # Parse ButtonStylingRules
        button_stroke = BUTTON_STROKE_WIDTH_MM
        button_radius = BUTTON_CORNER_RADIUS_MM
        button_min_w = BUTTON_MIN_WIDTH_MM
        button_min_h = BUTTON_MIN_HEIGHT_MM
        button_min_gap = BUTTON_MIN_GAP_MM
        default_font_size = DEFAULT_FONT_SIZE_PT

        rules_el = root.find("ButtonStylingRules")
        if rules_el is not None:
            button_stroke = float(rules_el.attrib.get("button_stroke_width_mm", BUTTON_STROKE_WIDTH_MM))
            button_radius = float(rules_el.attrib.get("button_corner_radius_mm", BUTTON_CORNER_RADIUS_MM))
            button_min_w = float(rules_el.attrib.get("button_min_width_mm", BUTTON_MIN_WIDTH_MM))
            button_min_h = float(rules_el.attrib.get("button_min_height_mm", BUTTON_MIN_HEIGHT_MM))
            button_min_gap = float(rules_el.attrib.get("button_min_gap_mm", BUTTON_MIN_GAP_MM))
            default_font_size = int(rules_el.attrib.get("default_font_size_pt", DEFAULT_FONT_SIZE_PT))

        # Parse GridConfig
        grid_snap = True
        grid_size = DEFAULT_GRID_SIZE_MM
        grid_el = root.find("GridConfig")
        if grid_el is not None:
            grid_snap = bool(int(grid_el.attrib.get("grid_snap_enabled", "1")))
            grid_size = float(grid_el.attrib.get("grid_size_mm", DEFAULT_GRID_SIZE_MM))

        # Parse InteriorRegion
        interior_buf = INTERIOR_BUFFER_MM
        int_el = root.find("InteriorRegion")
        if int_el is not None and "interior_buffer_mm" in int_el.attrib:
            interior_buf = float(int_el.attrib["interior_buffer_mm"])

        # Parse Buttons
        buttons: list[ButtonModel] = []
        buttons_el = root.find("Buttons")
        if buttons_el is not None:
            for be in buttons_el.findall("Button"):
                btn_id = be.attrib["id"]
                x_mm = float(be.attrib["x_mm"])
                y_mm = float(be.attrib["y_mm"])
                w_mm = float(be.attrib["width_mm"])
                h_mm = float(be.attrib["height_mm"])

                style_el = be.find("Style")
                font_size = int(style_el.attrib.get("font_size_pt", default_font_size)) if style_el is not None else default_font_size

                text_el = be.find("Text")
                text = text_el.text if (text_el is not None and text_el.text) else ""

                buttons.append(
                    ButtonModel(
                        id=btn_id,
                        x_mm=x_mm,
                        y_mm=y_mm,
                        width_mm=w_mm,
                        height_mm=h_mm,
                        text=text,
                        font_size_pt=font_size,
                    )
                )

        # Parse Custom Markers
        custom_markers: list[MarkerModel] = []
        markers_el = root.find("Markers")
        if markers_el is not None:
            marker_use_custom = bool(int(markers_el.attrib.get("use_custom_markers", str(int(use_custom)))))
            if marker_use_custom or use_custom:
                for me in markers_el.findall("Marker"):
                    m_id = int(me.attrib["id"])
                    cx_mm = float(me.attrib.get("center_x_mm", me.attrib.get("x_mm", 0.0)))
                    cy_mm = float(me.attrib.get("center_y_mm", me.attrib.get("y_mm", 0.0)))
                    m_sz = float(me.attrib.get("size_mm", marker_size))
                    custom_markers.append(
                        MarkerModel(id=m_id, x_mm=cx_mm, y_mm=cy_mm, size_mm=m_sz)
                    )

        return PaperLayoutModel(
            project_name=project_name,
            paper_width_mm=paper_width,
            paper_height_mm=paper_height,
            paper_margin_mm=paper_margin,
            marker_size_mm=marker_size,
            marker_spacing_mm=marker_spacing,
            marker_family=marker_family,
            marker_min_gap_ratio=marker_min_gap_ratio,
            marker_min_gap_mm=marker_min_gap,
            show_outer_markers=show_outer,
            use_custom_markers=bool(custom_markers) or use_custom,
            button_stroke_width_mm=button_stroke,
            button_corner_radius_mm=button_radius,
            button_min_width_mm=button_min_w,
            button_min_height_mm=button_min_h,
            button_min_gap_mm=button_min_gap,
            default_font_size_pt=default_font_size,
            interior_buffer_mm=interior_buf,
            grid_snap_enabled=grid_snap,
            grid_size_mm=grid_size,
            buttons=buttons,
            custom_markers=custom_markers,
        )
