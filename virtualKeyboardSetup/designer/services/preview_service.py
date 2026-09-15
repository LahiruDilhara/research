"""
Print Outcome Preview Service.
Renders high-resolution rasterized PIL image previews of the exact printed A4 paper sheet.
"""

import cv2
from PIL import Image, ImageDraw, ImageFont

from core.models.paper_layout import PaperLayoutModel
from core.geometry.marker_generator import generate_marker_layout
from config.app_config import AppConfig


class PreviewService:
    @staticmethod
    def render_preview(layout: PaperLayoutModel, config: AppConfig, scale: float = 3.0) -> Image.Image:
        width_px = int(config.paper_width_mm * scale)
        height_px = int(config.paper_height_mm * scale)

        img = Image.new("RGB", (width_px, height_px), "#FFFFFF")
        draw = ImageDraw.Draw(img)

        # 1. Draw AprilTag Markers
        if layout.use_custom_markers and layout.custom_markers:
            markers = layout.custom_markers
        else:
            markers = generate_marker_layout(config)
        marker_size_px = int(config.marker_size_mm * scale)
        tag_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)

        for m in markers:
            arr = cv2.aruco.generateImageMarker(tag_dict, m.id, marker_size_px)
            marker_pil = Image.fromarray(arr).convert("RGB")

            half_mm = config.marker_size_mm / 2.0
            x0 = int((m.x_mm - half_mm) * scale)
            y0 = int((m.y_mm - half_mm) * scale)
            img.paste(marker_pil, (x0, y0))

        # 2. Draw Buttons
        stroke_px = max(1, int(config.button_stroke_width_mm * scale))
        radius_px = int(config.button_corner_radius_mm * scale)

        for b in layout.buttons:
            x0 = int(b.x_mm * scale)
            y0 = int(b.y_mm * scale)
            w_px = int(b.width_mm * scale)
            h_px = int(b.height_mm * scale)
            x1 = x0 + w_px
            y1 = y0 + h_px

            draw.rounded_rectangle(
                [x0, y0, x1, y1],
                radius=radius_px,
                fill="#FFFFFF",
                outline="#262626",
                width=stroke_px,
            )

            if b.text:
                font_px = max(8, int(b.font_size_pt * (25.4 / 72.0) * scale))
                font = None
                for font_name in [
                    "Helvetica.ttf",
                    "LiberationSans-Regular.ttf",
                    "DejaVuSans.ttf",
                    "FreeSans.ttf",
                    "Arial.ttf",
                ]:
                    try:
                        font = ImageFont.truetype(font_name, font_px)
                        break
                    except Exception:
                        continue
                if font is None:
                    font = ImageFont.load_default()

                max_w = max(10, w_px - int(4.0 * scale))
                max_h = max(10, h_px - int(3.0 * scale))

                lines = PreviewService._wrap_text(
                    b.text,
                    max_w,
                    lambda s: draw.textbbox((0, 0), s, font=font)[2]
                    - draw.textbbox((0, 0), s, font=font)[0],
                )

                line_height = font_px * 1.2
                total_h = len(lines) * line_height

                if total_h > max_h:
                    max_lines = max(1, int(max_h / line_height))
                    lines = lines[:max_lines]
                    if not lines[-1].endswith("…"):
                        lines[-1] = lines[-1][: max(1, len(lines[-1]) - 1)] + "…"
                    total_h = len(lines) * line_height

                start_y = y0 + (h_px - total_h) / 2.0
                for idx, line in enumerate(lines):
                    bbox = draw.textbbox((0, 0), line, font=font)
                    lw = bbox[2] - bbox[0]
                    lx = x0 + (w_px - lw) / 2.0
                    ly = start_y + (idx * line_height)
                    draw.text((lx, ly), line, fill="#1A1A1A", font=font)

        # 3. Draw Project Name on Top Outer Margin Ring
        proj_name = getattr(layout, "project_name", "")
        if proj_name:
            font_size_px = max(7, int(4.5 * scale))
            font = None
            for font_name in ["Helvetica-Bold.ttf", "LiberationSans-Bold.ttf", "FreeSansBold.ttf", "Arial.ttf"]:
                try:
                    font = ImageFont.truetype(font_name, font_size_px)
                    break
                except Exception:
                    continue
            if font is None:
                font = ImageFont.load_default()
            title_text = f"PROJECT: {proj_name.upper()}"
            bbox = draw.textbbox((0, 0), title_text, font=font)
            tw = bbox[2] - bbox[0]
            tx = (width_px - tw) / 2.0
            ty = int((config.paper_margin_mm / 3.0) * scale)
            draw.text((tx, ty), title_text, fill="#333333", font=font)

        return img

    @staticmethod
    def _wrap_text(text: str, max_w: float, width_fn) -> list[str]:
        words = text.split(" ")
        lines = []
        current_line = []

        for word in words:
            if width_fn(word) > max_w:
                trunc_word = word
                while len(trunc_word) > 1 and width_fn(trunc_word + "…") > max_w:
                    trunc_word = trunc_word[:-1]
                word = trunc_word + "…" if len(trunc_word) < len(word) else word

            test_line = " ".join(current_line + [word])
            if width_fn(test_line) <= max_w:
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
