"""
xml_parser.py

Parses layout XML files exported by designer_app.py / project_io.py.
Extracts paper metadata, AprilTag markers (ID, center, 4 paper corners),
and button bounding boxes for runtime homography and touch detection.
"""

import xml.etree.ElementTree as ET
import os


class LayoutData:
    def __init__(self, paper_width_mm, paper_height_mm, marker_size_mm, marker_family, markers, buttons):
        self.paper_width_mm = paper_width_mm
        self.paper_height_mm = paper_height_mm
        self.marker_size_mm = marker_size_mm
        self.marker_family = marker_family
        self.markers = markers  # Dict mapping marker_id (int) -> marker info dict
        self.buttons = buttons  # List of button info dicts

    def find_button_at(self, x_mm, y_mm):
        """Returns the button dict if (x_mm, y_mm) falls inside its bounding box, else None."""
        for btn in self.buttons:
            if (btn["x_mm"] <= x_mm <= btn["x_max_mm"]) and (btn["y_mm"] <= y_mm <= btn["y_max_mm"]):
                return btn
        return None


def parse_layout_xml(xml_path):
    """
    Parses a PaperLayout XML file.

    Returns:
        LayoutData instance.
    """
    if not os.path.exists(xml_path):
        raise FileNotFoundError(f"XML file not found: {xml_path}")

    tree = ET.parse(xml_path)
    root = tree.getroot()

    if root.tag != "PaperLayout":
        raise ValueError(f"Invalid XML layout file. Expected root tag 'PaperLayout', got '{root.tag}'")

    paper_width_mm = float(root.get("paper_width_mm", 297.0))
    paper_height_mm = float(root.get("paper_height_mm", 210.0))
    marker_size_mm = float(root.get("marker_size_mm", 15.0))
    marker_family = root.get("marker_family", "DICT_APRILTAG_36h11")

    # Parse Markers
    markers_dict = {}
    markers_el = root.find("Markers")
    if markers_el is not None:
        for m_el in markers_el.findall("Marker"):
            m_id = int(m_el.get("id"))
            center_x = float(m_el.get("center_x_mm"))
            center_y = float(m_el.get("center_y_mm"))
            size = float(m_el.get("size_mm", marker_size_mm))

            corners_el = m_el.find("Corners")
            corners_list = []
            if corners_el is not None:
                for corner_name in ["TopLeft", "TopRight", "BottomRight", "BottomLeft"]:
                    c_el = corners_el.find(corner_name)
                    if c_el is not None:
                        cx = float(c_el.get("x_mm"))
                        cy = float(c_el.get("y_mm"))
                        corners_list.append((cx, cy))

            # Fallback if corners missing in XML
            if len(corners_list) != 4:
                h = size / 2.0
                corners_list = [
                    (center_x - h, center_y - h),  # TopLeft
                    (center_x + h, center_y - h),  # TopRight
                    (center_x + h, center_y + h),  # BottomRight
                    (center_x - h, center_y + h),  # BottomLeft
                ]

            markers_dict[m_id] = {
                "id": m_id,
                "center_x_mm": center_x,
                "center_y_mm": center_y,
                "size_mm": size,
                "corners_mm": corners_list  # [TL, TR, BR, BL]
            }

    # Parse Buttons
    buttons_list = []
    buttons_el = root.find("Buttons")
    if buttons_el is not None:
        for b_el in buttons_el.findall("Button"):
            b_id = b_el.get("id")
            x_mm = float(b_el.get("x_mm"))
            y_mm = float(b_el.get("y_mm"))
            width_mm = float(b_el.get("width_mm"))
            height_mm = float(b_el.get("height_mm"))
            x_max_mm = float(b_el.get("x_max_mm", x_mm + width_mm))
            y_max_mm = float(b_el.get("y_max_mm", y_mm + height_mm))

            text_el = b_el.find("Text")
            text = text_el.text if (text_el is not None and text_el.text) else ""

            buttons_list.append({
                "id": b_id,
                "x_mm": x_mm,
                "y_mm": y_mm,
                "width_mm": width_mm,
                "height_mm": height_mm,
                "x_max_mm": x_max_mm,
                "y_max_mm": y_max_mm,
                "center_x_mm": x_mm + width_mm / 2.0,
                "center_y_mm": y_mm + height_mm / 2.0,
                "text": text,
            })

    return LayoutData(
        paper_width_mm=paper_width_mm,
        paper_height_mm=paper_height_mm,
        marker_size_mm=marker_size_mm,
        marker_family=marker_family,
        markers=markers_dict,
        buttons=buttons_list
    )
