import io
import os
from typing import Tuple, Optional
from PIL import Image, ImageDraw, ImageFont


class WordImageRenderer:
    """
    Renders individual words as transparent PNG image byte streams using Pillow.
    Matches text metrics, font family (serif/sans/mono), weight (regular/bold),
    style (italic), color, font size, and baseline position.
    """

    def __init__(self, font_path: Optional[str] = None, dpi_scale: float = 3.0):
        self.font_path = font_path
        self.dpi_scale = dpi_scale
        self._font_cache = {}

    def _select_font_file(self, font_name: str, is_bold: bool, is_italic: bool) -> str:
        """
        Maps PDF font attributes to installed system TrueType font files.
        """
        if self.font_path and os.path.isfile(self.font_path):
            return self.font_path

        clean_name = (font_name or "").lower()

        # Check Serif fonts (e.g. Times, Roman, Serif, Minion, Georgia)
        if any(keyword in clean_name for keyword in ["times", "serif", "roman", "georgia", "minion"]):
            if is_bold and is_italic:
                return "/usr/share/fonts/truetype/liberation/LiberationSerif-BoldItalic.ttf"
            elif is_bold:
                return "/usr/share/fonts/truetype/liberation/LiberationSerif-Bold.ttf"
            elif is_italic:
                return "/usr/share/fonts/truetype/liberation/LiberationSerif-Italic.ttf"
            else:
                return "/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf"

        # Check Monospace fonts (e.g. Courier, Mono, Monospace)
        if any(keyword in clean_name for keyword in ["mono", "courier", "console"]):
            if is_bold and is_italic:
                return "/usr/share/fonts/truetype/liberation/LiberationMono-BoldItalic.ttf"
            elif is_bold:
                return "/usr/share/fonts/truetype/liberation/LiberationMono-Bold.ttf"
            elif is_italic:
                return "/usr/share/fonts/truetype/liberation/LiberationMono-Italic.ttf"
            else:
                return "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf"

        # Default Sans-Serif fonts (e.g. Helvetica, Arial, Sans, Liberation)
        if is_bold and is_italic:
            return "/usr/share/fonts/truetype/liberation/LiberationSans-BoldItalic.ttf"
        elif is_bold:
            return "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"
        elif is_italic:
            return "/usr/share/fonts/truetype/liberation/LiberationSans-Italic.ttf"
        else:
            return "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"

    def _get_font(self, font_name: str, size: float, is_bold: bool, is_italic: bool) -> ImageFont.ImageFont:
        """
        Retrieves or caches a TrueType font matching target point size and style.
        """
        font_file = self._select_font_file(font_name, is_bold, is_italic)
        scaled_size = max(8, int(size * self.dpi_scale))
        cache_key = (font_file, scaled_size)

        if cache_key in self._font_cache:
            return self._font_cache[cache_key]

        font = None
        if os.path.isfile(font_file):
            try:
                font = ImageFont.truetype(font_file, scaled_size)
            except Exception:
                font = None

        if font is None:
            font = ImageFont.load_default()

        self._font_cache[cache_key] = font
        return font

    @staticmethod
    def int_color_to_rgb(color_int: int) -> Tuple[int, int, int]:
        """
        Converts PyMuPDF 24-bit integer color (0xRRGGBB) to (R, G, B) tuple.
        """
        if color_int is None or color_int < 0:
            return (0, 0, 0)
        r = (color_int >> 16) & 0xFF
        g = (color_int >> 8) & 0xFF
        b = color_int & 0xFF
        return (r, g, b)

    def render_word_image(
        self,
        word_text: str,
        bbox_width: float,
        bbox_height: float,
        font_name: str,
        font_size: float,
        is_bold: bool = False,
        is_italic: bool = False,
        baseline_offset: Optional[float] = None,
        text_color: int = 0
    ) -> bytes:
        """
        Renders word_text into a transparent RGBA PNG image matching exact target bounding metrics.
        """
        scaled_w = max(1, int(bbox_width * self.dpi_scale))
        scaled_h = max(1, int(bbox_height * self.dpi_scale))

        # Create transparent image canvas
        image = Image.new("RGBA", (scaled_w, scaled_h), (255, 255, 255, 0))
        draw = ImageDraw.Draw(image)

        font = self._get_font(font_name, font_size, is_bold, is_italic)

        # Measure drawn text width to adjust font size if needed (prevents horizontal stretching)
        try:
            bbox = font.getbbox(word_text, anchor="ls")
            drawn_w = bbox[2] - bbox[0]
            if drawn_w > 0 and abs(drawn_w - scaled_w) > (2 * self.dpi_scale):
                ratio = scaled_w / drawn_w
                adjusted_size = max(6.0, font_size * ratio)
                font_file = self._select_font_file(font_name, is_bold, is_italic)
                font = ImageFont.truetype(font_file, max(8, int(adjusted_size * self.dpi_scale)))
        except Exception:
            pass

        # Calculate baseline position
        if baseline_offset is not None and baseline_offset > 0:
            pil_baseline_y = int(baseline_offset * self.dpi_scale)
        else:
            pil_baseline_y = int(scaled_h * 0.78)

        rgb_color = self.int_color_to_rgb(text_color)
        rgba_fill = (rgb_color[0], rgb_color[1], rgb_color[2], 255)

        # Draw text at exact baseline using Left-Baseline anchor 'ls'
        draw.text((0, pil_baseline_y), word_text, fill=rgba_fill, font=font, anchor="ls")

        buffer = io.BytesIO()
        image.save(buffer, format="PNG", dpi=(int(72 * self.dpi_scale), int(72 * self.dpi_scale)))
        buffer.seek(0)
        return buffer.getvalue()
