"""
project_io.py

Save/load the in-progress design, and export the final XML (for the
runtime touch-detection app) and PDF (for printing).
"""

import json
import xml.etree.ElementTree as ET
from xml.dom import minidom
import numpy as np
import cv2
from reportlab.lib.units import mm
from reportlab.pdfgen import canvas as pdf_canvas
from reportlab.lib.utils import ImageReader
from PIL import Image
import io

from layout_constants import (
    PAPER_WIDTH_MM, PAPER_HEIGHT_MM,
    PAPER_MARGIN_MM, MARKER_SIZE_MM, MARKER_MARGIN_MM, MARKER_SPACING_MM, MARKER_FAMILY,
    MARKER_RING_OUTER_MIN, MARKER_RING_OUTER_MAX_X, MARKER_RING_OUTER_MAX_Y,
    MARKER_RING_INNER_MIN, MARKER_RING_INNER_MAX_X, MARKER_RING_INNER_MAX_Y,
    BUTTON_STROKE_WIDTH_MM, BUTTON_CORNER_RADIUS_MM,
    INTERIOR_X_MIN, INTERIOR_Y_MIN, INTERIOR_X_MAX, INTERIOR_Y_MAX,
    generate_marker_layout, marker_corners_mm,
)


# =============================================================
# PROJECT SAVE / LOAD  (in-progress design, editable later)
# =============================================================

def save_project(filepath, buttons):
    """
    buttons: list of dicts, each:
        {"id": str, "x_mm": float, "y_mm": float, "width_mm": float,
         "height_mm": float, "text": str, "font_size_pt": int}
    """
    data = {
        "paper_width_mm": PAPER_WIDTH_MM,
        "paper_height_mm": PAPER_HEIGHT_MM,
        "marker_size_mm": MARKER_SIZE_MM,
        "buttons": buttons,
    }
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)


def load_project(filepath):
    with open(filepath, "r") as f:
        data = json.load(f)
    return data.get("buttons", [])


# =============================================================
# XML EXPORT  (consumed by the runtime touch-detection app)
# =============================================================

def export_xml(filepath, buttons):
    markers = generate_marker_layout()

    root = ET.Element("PaperLayout")
    root.set("paper_width_mm", f"{PAPER_WIDTH_MM}")
    root.set("paper_height_mm", f"{PAPER_HEIGHT_MM}")
    root.set("marker_size_mm", f"{MARKER_SIZE_MM}")
    root.set("marker_margin_mm", f"{MARKER_MARGIN_MM}")
    root.set("marker_spacing_mm", f"{MARKER_SPACING_MM}")
    root.set("marker_family", MARKER_FAMILY)
    root.set("coordinate_origin", "top_left_paper_corner (x_right, y_down)")

    # 1. Outer Paper Margin (Distance from physical paper edge to start of marker ring)
    margin_el = ET.SubElement(root, "PaperMargin")
    margin_el.set("margin_mm", f"{PAPER_MARGIN_MM:.3f}")
    margin_el.set("top_mm", f"{PAPER_MARGIN_MM:.3f}")
    margin_el.set("bottom_mm", f"{PAPER_MARGIN_MM:.3f}")
    margin_el.set("left_mm", f"{PAPER_MARGIN_MM:.3f}")
    margin_el.set("right_mm", f"{PAPER_MARGIN_MM:.3f}")

    # 2. Marker Ring Zone (Area occupied by AprilTag markers)
    marker_zone_el = ET.SubElement(root, "MarkerRingZone")
    marker_zone_el.set("outer_x_min_mm", f"{MARKER_RING_OUTER_MIN:.3f}")
    marker_zone_el.set("outer_y_min_mm", f"{MARKER_RING_OUTER_MIN:.3f}")
    marker_zone_el.set("outer_x_max_mm", f"{MARKER_RING_OUTER_MAX_X:.3f}")
    marker_zone_el.set("outer_y_max_mm", f"{MARKER_RING_OUTER_MAX_Y:.3f}")
    marker_zone_el.set("inner_x_min_mm", f"{MARKER_RING_INNER_MIN:.3f}")
    marker_zone_el.set("inner_y_min_mm", f"{MARKER_RING_INNER_MIN:.3f}")
    marker_zone_el.set("inner_x_max_mm", f"{MARKER_RING_INNER_MAX_X:.3f}")
    marker_zone_el.set("inner_y_max_mm", f"{MARKER_RING_INNER_MAX_Y:.3f}")

    # 3. Interior active button region limits
    interior_el = ET.SubElement(root, "InteriorRegion")
    interior_el.set("x_min_mm", f"{INTERIOR_X_MIN:.3f}")
    interior_el.set("y_min_mm", f"{INTERIOR_Y_MIN:.3f}")
    interior_el.set("x_max_mm", f"{INTERIOR_X_MAX:.3f}")
    interior_el.set("y_max_mm", f"{INTERIOR_Y_MAX:.3f}")
    interior_el.set("width_mm", f"{(INTERIOR_X_MAX - INTERIOR_X_MIN):.3f}")
    interior_el.set("height_mm", f"{(INTERIOR_Y_MAX - INTERIOR_Y_MIN):.3f}")

    # Markers section with full corner details
    markers_el = ET.SubElement(root, "Markers", count=str(len(markers)))
    for m in markers:
        me = ET.SubElement(markers_el, "Marker")
        me.set("id", str(m["id"]))
        me.set("center_x_mm", f"{m['x_mm']:.3f}")
        me.set("center_y_mm", f"{m['y_mm']:.3f}")
        me.set("size_mm", f"{MARKER_SIZE_MM:.3f}")

        # Add exact 4 corner coordinates (TL, TR, BR, BL)
        corners = marker_corners_mm(m["x_mm"], m["y_mm"], MARKER_SIZE_MM)
        corners_el = ET.SubElement(me, "Corners")
        corner_names = ["TopLeft", "TopRight", "BottomRight", "BottomLeft"]
        for c_name, (cx, cy) in zip(corner_names, corners):
            ce = ET.SubElement(corners_el, c_name)
            ce.set("x_mm", f"{cx:.3f}")
            ce.set("y_mm", f"{cy:.3f}")

    # Buttons section with full geometry and bounding box details
    buttons_el = ET.SubElement(root, "Buttons", count=str(len(buttons)))
    for b in buttons:
        be = ET.SubElement(buttons_el, "Button")
        be.set("id", str(b["id"]))
        be.set("x_mm", f"{b['x_mm']:.3f}")
        be.set("y_mm", f"{b['y_mm']:.3f}")
        be.set("width_mm", f"{b['width_mm']:.3f}")
        be.set("height_mm", f"{b['height_mm']:.3f}")
        be.set("x_max_mm", f"{(b['x_mm'] + b['width_mm']):.3f}")
        be.set("y_max_mm", f"{(b['y_mm'] + b['height_mm']):.3f}")
        be.set("center_x_mm", f"{(b['x_mm'] + b['width_mm'] / 2.0):.3f}")
        be.set("center_y_mm", f"{(b['y_mm'] + b['height_mm'] / 2.0):.3f}")

        style_el = ET.SubElement(be, "Style")
        style_el.set("font_size_pt", str(b.get("font_size_pt", 14)))
        style_el.set("stroke_width_mm", f"{BUTTON_STROKE_WIDTH_MM}")
        style_el.set("corner_radius_mm", f"{BUTTON_CORNER_RADIUS_MM}")

        text_el = ET.SubElement(be, "Text")
        text_el.text = b.get("text", "")

    rough_string = ET.tostring(root, "utf-8")
    pretty = minidom.parseString(rough_string).toprettyxml(indent="  ")
    with open(filepath, "w") as f:
        f.write(pretty)


# =============================================================
# PDF EXPORT  (physically accurate, for printing)
# =============================================================

def _generate_marker_image_array(tag_id, pixel_size=400):
    """Generate a single AprilTag (tag36h11) marker as a numpy image array."""
    tag_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    img = cv2.aruco.generateImageMarker(tag_dict, tag_id, pixel_size)
    return img


def _marker_image_reader(tag_id, pixel_size=400):
    """Wrap a generated marker image as a reportlab ImageReader."""
    arr = _generate_marker_image_array(tag_id, pixel_size)
    pil_img = Image.fromarray(arr).convert("L")
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    buf.seek(0)
    return ImageReader(buf)


def _wrap_text_to_width(text, max_w, string_width_fn):
    """Utility to break string into lines fitting within max_w and truncate long words with ellipsis."""
    words = text.split(" ")
    lines = []
    current_line = []
    
    for word in words:
        # Check if single word itself is wider than max_w
        if string_width_fn(word) > max_w:
            # Truncate word with ellipsis
            trunc_word = word
            while len(trunc_word) > 1 and string_width_fn(trunc_word + "…") > max_w:
                trunc_word = trunc_word[:-1]
            word = trunc_word + "…" if len(trunc_word) < len(word) else word

        test_line = " ".join(current_line + [word])
        if string_width_fn(test_line) <= max_w:
            current_line.append(word)
        else:
            if current_line:
                lines.append(" ".join(current_line))
                current_line = [word]
            else:
                lines.append(word)
                current_line = []
    if current_line:
        lines.append(" ".join(current_line))
    return lines if lines else [text]


def export_pdf(filepath, buttons):
    """
    Render the final printable A4-landscape PDF: marker ring + buttons
    with text, all placed using exact mm coordinates and word-wrapped text inside button bounds.
    """
    markers = generate_marker_layout()

    page_width = PAPER_WIDTH_MM * mm
    page_height = PAPER_HEIGHT_MM * mm

    c = pdf_canvas.Canvas(filepath, pagesize=(page_width, page_height))

    # --- Draw markers ---
    for m in markers:
        size_pt = MARKER_SIZE_MM * mm
        x_mm = m["x_mm"] - MARKER_SIZE_MM / 2.0
        y_mm_top = m["y_mm"] - MARKER_SIZE_MM / 2.0
        pdf_x = x_mm * mm
        pdf_y = (PAPER_HEIGHT_MM - y_mm_top - MARKER_SIZE_MM) * mm

        img_reader = _marker_image_reader(m["id"], pixel_size=400)
        c.drawImage(img_reader, pdf_x, pdf_y, width=size_pt, height=size_pt)

    # --- Draw buttons ---
    stroke_pt = BUTTON_STROKE_WIDTH_MM * mm
    radius_pt = BUTTON_CORNER_RADIUS_MM * mm

    c.setLineWidth(stroke_pt)
    c.setStrokeColorRGB(0.15, 0.15, 0.15)

    for b in buttons:
        x_mm, y_mm, w_mm, h_mm = b["x_mm"], b["y_mm"], b["width_mm"], b["height_mm"]
        pdf_x = x_mm * mm
        pdf_y = (PAPER_HEIGHT_MM - y_mm - h_mm) * mm  # flip y
        pdf_w = w_mm * mm
        pdf_h = h_mm * mm

        c.roundRect(pdf_x, pdf_y, pdf_w, pdf_h, radius_pt, stroke=1, fill=0)

        font_size = b.get("font_size_pt", 14)
        c.setFont("Helvetica", font_size)
        text = b.get("text", "")
        if text:
            # pdf_w is in points (1mm = 2.8346pt). 4mm margin on sides = 8mm total margin (in pt: 8 * mm)
            max_w = max(10, pdf_w - (4 * mm))
            max_h = max(10, pdf_h - (3 * mm))

            lines = _wrap_text_to_width(text, max_w, lambda s: c.stringWidth(s, "Helvetica", font_size))
            
            line_height = font_size * 1.2
            total_text_h = len(lines) * line_height
            
            # Clip number of lines if height exceeds button box
            if total_text_h > max_h:
                max_lines = max(1, int(max_h / line_height))
                lines = lines[:max_lines]
                if not lines[-1].endswith("…"):
                    lines[-1] = lines[-1][:max(1, len(lines[-1]) - 1)] + "…"
                total_text_h = len(lines) * line_height

            c.setFillColorRGB(0.1, 0.1, 0.1)
            # Center block vertically
            start_y = pdf_y + (pdf_h / 2.0) + (total_text_h / 2.0) - (font_size * 0.8)
            for i, line in enumerate(lines):
                line_w = c.stringWidth(line, "Helvetica", font_size)
                line_x = pdf_x + (pdf_w - line_w) / 2.0
                line_y = start_y - (i * line_height)
                c.drawString(line_x, line_y, line)

    c.showPage()
    c.save()


# =============================================================
# PRINT OUTCOME PREVIEW GENERATOR
# =============================================================

def generate_preview_image(buttons, scale=3.0):
    """
    Renders the exact printable paper outcome (A4 landscape) as a high-resolution PIL Image.
    Uses identical word wrapping and bounding box constraints so text never overflows preview or PDF.
    """
    from PIL import ImageDraw, ImageFont

    width_px = int(PAPER_WIDTH_MM * scale)
    height_px = int(PAPER_HEIGHT_MM * scale)

    img = Image.new("RGB", (width_px, height_px), "#FFFFFF")
    draw = ImageDraw.Draw(img)

    # 1. Draw AprilTag markers
    markers = generate_marker_layout()
    marker_size_px = int(MARKER_SIZE_MM * scale)

    for m in markers:
        arr = _generate_marker_image_array(m["id"], pixel_size=marker_size_px)
        marker_pil = Image.fromarray(arr).convert("RGB")

        half_mm = MARKER_SIZE_MM / 2.0
        x0 = int((m["x_mm"] - half_mm) * scale)
        y0 = int((m["y_mm"] - half_mm) * scale)
        img.paste(marker_pil, (x0, y0))

    # 2. Draw Buttons with wrapped text
    stroke_px = max(1, int(BUTTON_STROKE_WIDTH_MM * scale))
    radius_px = int(BUTTON_CORNER_RADIUS_MM * scale)

    for b in buttons:
        x0 = int(b["x_mm"] * scale)
        y0 = int(b["y_mm"] * scale)
        w_px = int(b["width_mm"] * scale)
        h_px = int(b["height_mm"] * scale)
        x1 = x0 + w_px
        y1 = y0 + h_px

        draw.rounded_rectangle([x0, y0, x1, y1], radius=radius_px, fill="#FFFFFF", outline="#262626", width=stroke_px)

        font_size_pt = b.get("font_size_pt", 14)
        font_px = max(8, int(font_size_pt * (25.4 / 72.0) * scale))

        font = None
        for font_name in ["Helvetica.ttf", "LiberationSans-Regular.ttf", "DejaVuSans.ttf", "FreeSans.ttf", "Arial.ttf"]:
            try:
                font = ImageFont.truetype(font_name, font_px)
                break
            except Exception:
                continue

        if font is None:
            font = ImageFont.load_default()

        text = b.get("text", "")
        if text:
            # Horizontal margin in pixels (scale px per mm)
            max_w = max(10, w_px - int(4.0 * scale))
            max_h = max(10, h_px - int(3.0 * scale))

            lines = _wrap_text_to_width(text, max_w, lambda s: draw.textbbox((0, 0), s, font=font)[2] - draw.textbbox((0, 0), s, font=font)[0])
            
            line_height = font_px * 1.2
            total_h = len(lines) * line_height

            if total_h > max_h:
                max_lines = max(1, int(max_h / line_height))
                lines = lines[:max_lines]
                if not lines[-1].endswith("…"):
                    lines[-1] = lines[-1][:max(1, len(lines[-1]) - 1)] + "…"
                total_h = len(lines) * line_height

            start_y = y0 + (h_px - total_h) / 2.0
            for i, line in enumerate(lines):
                bbox = draw.textbbox((0, 0), line, font=font)
                lw = bbox[2] - bbox[0]
                lx = x0 + (w_px - lw) / 2.0
                ly = start_y + (i * line_height)
                draw.text((lx, ly), line, fill="#1A1A1A", font=font)

    return img
