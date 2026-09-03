import sys, os, io
from typing import Optional
sys.path.insert(0, os.path.abspath("."))
from PIL import Image, ImageDraw, ImageFont
import pymupdf as fitz
from pdf_processor.renderer import WordImageRenderer


def render_word_image_crisp(
    renderer: WordImageRenderer,
    word_text: str,
    bbox_width: float,
    bbox_height: float,
    font_name: str,
    font_size: float,
    is_bold: bool = False,
    is_italic: bool = False,
    baseline_offset: float = 0.0,
    text_color: int = 0,
    custom_font_file: Optional[str] = None
) -> bytes:
    dpi_scale = renderer.dpi_scale
    canvas_w = max(1, int(bbox_width * dpi_scale))
    canvas_h = max(1, int(bbox_height * dpi_scale))

    # 1. Create a solid WHITE RGB image
    bg_image = Image.new("RGB", (canvas_w, canvas_h), (255, 255, 255))
    draw = ImageDraw.Draw(bg_image)

    font = renderer._select_font(font_name, font_size, is_bold, is_italic, custom_font_file)
    rgb_color = renderer.int_color_to_rgb(text_color)

    baseline_y = baseline_offset * dpi_scale if baseline_offset > 0 else canvas_h * 0.78
    draw_pt = (0, baseline_y)

    # Draw text on solid white background
    draw.text(draw_pt, word_text, font=font, fill=rgb_color, anchor="ls")

    # 2. Convert to RGBA and make white background transparent while preserving crisp glyph anti-aliasing
    rgba_image = bg_image.convert("RGBA")
    data = rgba_image.getdata()

    new_data = []
    r_target, g_target, b_target = rgb_color
    for r, g, b, a in data:
        if r > 250 and g > 250 and b > 250:
            # Fully white -> fully transparent
            new_data.append((255, 255, 255, 0))
        else:
            # Calculate alpha based on luminance distance from white
            lum = (0.299 * r + 0.587 * g + 0.114 * b)
            alpha = int(255 * (1.0 - (lum / 255.0)))
            new_data.append((r_target, g_target, b_target, alpha))

    rgba_image.putdata(new_data)

    buffer = io.BytesIO()
    rgba_image.save(buffer, format="PNG")
    buffer.seek(0)
    return buffer.getvalue()


def test_rendering_quality():
    renderer = WordImageRenderer(dpi_scale=3.0)
    font_file = renderer.CMU_SERIF_REGULAR
    
    # Test rendering a sample word
    img_bytes = render_word_image_crisp(
        renderer=renderer,
        word_text="Fingertip",
        bbox_width=50.0,
        bbox_height=12.0,
        font_name="CMR12",
        font_size=11.96,
        is_bold=False,
        is_italic=False,
        baseline_offset=9.0,
        text_color=0,
        custom_font_file=font_file
    )
    
    print(f"Crisp PNG rendered! Buffer size: {len(img_bytes)} bytes.")


if __name__ == "__main__":
    test_rendering_quality()
