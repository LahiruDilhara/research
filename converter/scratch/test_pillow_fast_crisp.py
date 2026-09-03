import sys, os, io, time
sys.path.insert(0, os.path.abspath("."))
from PIL import Image, ImageDraw
import pymupdf as fitz
from pdf_processor.renderer import WordImageRenderer


def render_word_image_crisp_fast(
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
    custom_font_file: str = None
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

    # 2. Extract grayscale 'L' channel: white (255) -> 0 alpha, black (0) -> 255 alpha
    gray_L = bg_image.convert("L")
    alpha_channel = Image.eval(gray_L, lambda lum: int(255 * (1.0 - (lum / 255.0))))

    # 3. Create target solid color RGBA image and attach exact alpha_channel
    color_image = Image.new("RGBA", (canvas_w, canvas_h), (rgb_color[0], rgb_color[1], rgb_color[2], 255))
    color_image.putalpha(alpha_channel)

    buffer = io.BytesIO()
    color_image.save(buffer, format="PNG")
    buffer.seek(0)
    return buffer.getvalue()


def benchmark():
    renderer = WordImageRenderer(dpi_scale=3.0)
    font_file = renderer.CMU_SERIF_REGULAR
    
    t0 = time.time()
    for _ in range(1000):
        render_word_image_crisp_fast(
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
    t1 = time.time() - t0
    print(f"Rendered 1000 crisp word images in {t1:.3f} seconds ({t1/1000*1000:.3f} ms per image)!")


if __name__ == "__main__":
    benchmark()
