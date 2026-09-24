"""
core/layout/layout_parser.py

Parses the designer-exported XML layout file into clean domain objects.
No business logic lives here, pure data extraction only.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path


# ── Domain objects ─────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class MarkerData:
    """One AprilTag fiducial anchor as defined in the XML."""
    id: int
    center_x_mm: float
    center_y_mm: float
    size_mm: float
    # 4 corner positions in mm-space (top_left, top_right, bottom_right, bottom_left)
    top_left: tuple[float, float]
    top_right: tuple[float, float]
    bottom_right: tuple[float, float]
    bottom_left: tuple[float, float]

    @property
    def corners_mm(self) -> list[tuple[float, float]]:
        """All 4 corners in canonical clockwise order:
        top-left, top-right, bottom-right, bottom-left."""
        return [self.top_left, self.top_right, self.bottom_right, self.bottom_left]

    def __getitem__(self, key: str):
        return getattr(self, key)


class MarkerList(list):
    """List of MarkerData that also provides dict-like items() and ID lookup for analyzer compatibility."""
    def items(self):
        return [(m.id, m) for m in self]

    def __contains__(self, key):
        if super().__contains__(key):
            return True
        if isinstance(key, int):
            return any(m.id == key for m in self)
        return False

    def __getitem__(self, key):
        if isinstance(key, int):
            if key < 0 or key >= len(self):
                for m in self:
                    if m.id == key:
                        return m
        return super().__getitem__(key)


@dataclass
class ButtonData:
    """One key button as defined in the XML."""
    id: str
    label: str
    x_mm: float
    y_mm: float
    x_max_mm: float
    y_max_mm: float
    width_mm: float
    height_mm: float
    center_x_mm: float
    center_y_mm: float

    def contains_mm(self, x: float, y: float) -> bool:
        """True if the mm-space point (x, y) falls inside this button's bounding box."""
        return self.x_mm <= x <= self.x_max_mm and self.y_mm <= y <= self.y_max_mm

    def __getitem__(self, key: str):
        if key == "text":
            return self.label
        return getattr(self, key)


@dataclass
class LayoutData:
    """Complete parsed layout: paper dimensions, markers, and buttons."""
    paper_width_mm: float
    paper_height_mm: float
    marker_size_mm: float
    marker_family: str
    project_name: str = "Paper Virtual Keyboard"
    markers: list[MarkerData] = field(default_factory=MarkerList)
    buttons: list[ButtonData] = field(default_factory=list)
    source_path: str = ""

    @property
    def marker_by_id(self) -> dict[int, MarkerData]:
        return {m.id: m for m in self.markers}

    def find_button_at(self, x_mm: float, y_mm: float) -> ButtonData | None:
        """Returns the button if (x_mm, y_mm) falls inside its bounding box, else None."""
        for btn in self.buttons:
            if btn.contains_mm(x_mm, y_mm):
                return btn
        return None


# ── Parser ─────────────────────────────────────────────────────────────────────

class LayoutParser:
    """Parses a designer XML file into a LayoutData object."""

    def parse(self, xml_path: str | Path) -> LayoutData:
        """
        Raises
        ------
        FileNotFoundError  : if xml_path does not exist.
        ValueError         : if the XML is malformed or missing required attributes.
        """
        path = Path(xml_path)
        if not path.exists():
            raise FileNotFoundError(f"Layout XML not found: {path}")

        try:
            tree = ET.parse(path)
        except ET.ParseError as exc:
            raise ValueError(f"Malformed XML: {exc}") from exc

        root = tree.getroot()
        if root.tag != "PaperLayout":
            raise ValueError(f"Expected root <PaperLayout>, got <{root.tag}>")

        proj_name = root.attrib.get("project_name", "")
        if not proj_name:
            sys_cfg = root.find(".//SystemConfig")
            if sys_cfg is not None and "project_name" in sys_cfg.attrib:
                proj_name = sys_cfg.attrib["project_name"]
        if not proj_name:
            proj_name = path.stem.replace("_", " ").title()

        marker_size_mm = float(root.attrib.get("marker_size_mm", "15.0"))
        layout = LayoutData(
            paper_width_mm  = float(root.attrib["paper_width_mm"]),
            paper_height_mm = float(root.attrib["paper_height_mm"]),
            marker_size_mm  = marker_size_mm,
            marker_family   = root.attrib.get("marker_family", "DICT_APRILTAG_36h11"),
            project_name    = proj_name,
            source_path     = str(path),
        )

        designer_el = root.find("DesignerLayout")
        if designer_el is None:
            designer_el = root

        # ── Parse markers ─────────────────────────────────────────────────────
        markers_el = designer_el.find("Markers")
        if markers_el is not None:
            for m_el in markers_el.findall("Marker"):
                m_id = int(m_el.attrib["id"])
                cx = float(m_el.attrib["center_x_mm"])
                cy = float(m_el.attrib["center_y_mm"])
                size = float(m_el.attrib.get("size_mm", marker_size_mm))
                corners_el = m_el.find("Corners")

                if corners_el is not None:
                    try:
                        tl = self._corner(corners_el, "TopLeft")
                        tr = self._corner(corners_el, "TopRight")
                        br = self._corner(corners_el, "BottomRight")
                        bl = self._corner(corners_el, "BottomLeft")
                    except Exception:
                        h = size / 2.0
                        tl, tr, br, bl = (cx - h, cy - h), (cx + h, cy - h), (cx + h, cy + h), (cx - h, cy + h)
                else:
                    h = size / 2.0
                    tl, tr, br, bl = (cx - h, cy - h), (cx + h, cy - h), (cx + h, cy + h), (cx - h, cy + h)

                layout.markers.append(MarkerData(
                    id            = m_id,
                    center_x_mm   = cx,
                    center_y_mm   = cy,
                    size_mm       = size,
                    top_left      = tl,
                    top_right     = tr,
                    bottom_right  = br,
                    bottom_left   = bl,
                ))

        # ── Parse buttons ─────────────────────────────────────────────────────
        buttons_el = designer_el.find("Buttons")
        if buttons_el is not None:
            for b_el in buttons_el.findall("Button"):
                text_el = b_el.find("Text")
                label = text_el.text.strip() if text_el is not None and text_el.text else b_el.attrib["id"]
                x_mm = float(b_el.attrib["x_mm"])
                y_mm = float(b_el.attrib["y_mm"])
                width_mm = float(b_el.attrib["width_mm"])
                height_mm = float(b_el.attrib["height_mm"])
                x_max_mm = float(b_el.attrib.get("x_max_mm", x_mm + width_mm))
                y_max_mm = float(b_el.attrib.get("y_max_mm", y_mm + height_mm))
                center_x_mm = float(b_el.attrib.get("center_x_mm", x_mm + width_mm / 2.0))
                center_y_mm = float(b_el.attrib.get("center_y_mm", y_mm + height_mm / 2.0))

                layout.buttons.append(ButtonData(
                    id          = b_el.attrib["id"],
                    label       = label,
                    x_mm        = x_mm,
                    y_mm        = y_mm,
                    x_max_mm    = x_max_mm,
                    y_max_mm    = y_max_mm,
                    width_mm    = width_mm,
                    height_mm   = height_mm,
                    center_x_mm = center_x_mm,
                    center_y_mm = center_y_mm,
                ))

        return layout

    @staticmethod
    def _corner(corners_el: ET.Element, tag: str) -> tuple[float, float]:
        el = corners_el.find(tag)
        if el is None:
            raise ValueError(f"Missing <{tag}> inside <Corners>")
        return (float(el.attrib["x_mm"]), float(el.attrib["y_mm"]))
