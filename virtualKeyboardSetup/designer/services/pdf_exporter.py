"""
PDF Exporter Service.
Generates physically accurate A4 landscape printable PDF layout using ReportLab.
"""

import io
import cv2
from PIL import Image
from pathlib import Path
from reportlab.lib.units import mm
from reportlab.pdfgen import canvas as pdf_canvas
from reportlab.lib.utils import ImageReader

from core.interfaces.export_interface import IExportService
from core.models.paper_layout import PaperLayoutModel
from core.geometry.marker_generator import generate_marker_layout
from config.app_config import AppConfig


class PdfExporter(IExportService):
    @staticmethod
    def _generate_marker_image_reader(tag_id: int, pixel_size: int = 400) -> ImageReader:
        tag_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
        img_arr = cv2.aruco.generateImageMarker(tag_dict, tag_id, pixel_size)
        pil_img = Image.fromarray(img_arr).convert("L")
        buf = io.BytesIO()
        pil_img.save(buf, format="PNG")
        buf.seek(0)
        return ImageReader(buf)

    @staticmethod
    def _wrap_text_to_width(text: str, max_w: float, string_width_fn) -> list[str]:
        words = text.split(" ")
        lines = []
        current_line = []

        for word in words:
            if string_width_fn(word) > max_w:
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

    def export(self, filepath: str | Path, layout: PaperLayoutModel, config: AppConfig) -> None:
        if layout.use_custom_markers and layout.custom_markers:
            markers = layout.custom_markers
        else:
            markers = generate_marker_layout(config)
        path = Path(filepath)

        page_width = config.paper_width_mm * mm
        page_height = config.paper_height_mm * mm

        path.parent.mkdir(parents=True, exist_ok=True)
        c = pdf_canvas.Canvas(str(path), pagesize=(page_width, page_height))

        # 1. Draw AprilTag Markers
        marker_size_pt = config.marker_size_mm * mm
        for m in markers:
            x_mm = m.x_mm - config.marker_size_mm / 2.0
            y_mm_top = m.y_mm - config.marker_size_mm / 2.0
            pdf_x = x_mm * mm
            pdf_y = (config.paper_height_mm - y_mm_top - config.marker_size_mm) * mm

            reader = self._generate_marker_image_reader(m.id)
            c.drawImage(reader, pdf_x, pdf_y, width=marker_size_pt, height=marker_size_pt)

        # 2. Draw Buttons
        stroke_pt = config.button_stroke_width_mm * mm
        radius_pt = config.button_corner_radius_mm * mm

        c.setLineWidth(stroke_pt)
        c.setStrokeColorRGB(0.15, 0.15, 0.15)

        for b in layout.buttons:
            pdf_x = b.x_mm * mm
            pdf_y = (config.paper_height_mm - b.y_mm - b.height_mm) * mm
            pdf_w = b.width_mm * mm
            pdf_h = b.height_mm * mm

            c.roundRect(pdf_x, pdf_y, pdf_w, pdf_h, radius_pt, stroke=1, fill=0)

            if b.text:
                font_size = b.font_size_pt
                c.setFont("Helvetica", font_size)
                max_w = max(10, pdf_w - (4 * mm))
                max_h = max(10, pdf_h - (3 * mm))

                lines = self._wrap_text_to_width(
                    b.text, max_w, lambda s: c.stringWidth(s, "Helvetica", font_size)
                )

                line_height = font_size * 1.2
                total_text_h = len(lines) * line_height

                if total_text_h > max_h:
                    max_lines = max(1, int(max_h / line_height))
                    lines = lines[:max_lines]
                    if not lines[-1].endswith("…"):
                        lines[-1] = lines[-1][: max(1, len(lines[-1]) - 1)] + "…"
                    total_text_h = len(lines) * line_height

                c.setFillColorRGB(0.1, 0.1, 0.1)
                start_y = pdf_y + (pdf_h / 2.0) + (total_text_h / 2.0) - (font_size * 0.8)
                for idx, line in enumerate(lines):
                    line_w = c.stringWidth(line, "Helvetica", font_size)
                    line_x = pdf_x + (pdf_w - line_w) / 2.0
                    line_y = start_y - (idx * line_height)
                    c.drawString(line_x, line_y, line)

        # 3. Draw Project Name Title on Top Outer Margin Ring (Outside tags with gap)
        if getattr(layout, "project_name", None):
            c.setFont("Helvetica-Bold", 5.5)
            c.setFillColorRGB(0.3, 0.3, 0.3)
            pdf_title_x = (config.paper_width_mm / 2.0) * mm
            pdf_title_y = (config.paper_height_mm - (config.paper_margin_mm / 3.0)) * mm
            c.drawCentredString(pdf_title_x, pdf_title_y, f"PROJECT: {layout.project_name.upper()}")

        c.showPage()
        c.save()
