import os
import io
from typing import Tuple, Optional
from PIL import Image, ImageDraw, ImageFont
from pdf_processor.classifier import TextClassifier


class WordImageRenderer:
    """
    Renders text words into transparent PNG image buffers matching the exact font properties
    (font family, size, color, baseline alignment, and style) of the original PDF text span.
    """

    # Default TrueType font fallbacks
    SERIF_REGULAR = "/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf"
    SERIF_BOLD = "/usr/share/fonts/truetype/liberation/LiberationSerif-Bold.ttf"
    SERIF_ITALIC = "/usr/share/fonts/truetype/liberation/LiberationSerif-Italic.ttf"
    SERIF_BOLD_ITALIC = "/usr/share/fonts/truetype/liberation/LiberationSerif-BoldItalic.ttf"

    SANS_REGULAR = "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"
    SANS_BOLD = "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"
    SANS_ITALIC = "/usr/share/fonts/truetype/liberation/LiberationSans-Italic.ttf"
    SANS_BOLD_ITALIC = "/usr/share/fonts/truetype/liberation/LiberationSans-BoldItalic.ttf"

    MONO_REGULAR = "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf"

    def __init__(self, font_path: Optional[str] = None, dpi_scale: float = 3.0):
        self.font_path = font_path
        self.dpi_scale = max(1.0, dpi_scale)

    def _select_font_file(self, font_name: str, is_bold: bool, is_italic: bool) -> str:
        """
        Selects the appropriate system TrueType font file path based on font family and style attributes.
        """
        if self.font_path and os.path.exists(self.font_path):
            return self.font_path

        name_lower = font_name.lower()

        # Check for Serif / Roman fonts (CMR, LMRoman, Times, Georgia, Serif)
        if TextClassifier.is_serif_font(name_lower):
            if is_bold and is_italic:
                return self.SERIF_BOLD_ITALIC
            elif is_bold:
                return self.SERIF_BOLD
            elif is_italic:
                return self.SERIF_ITALIC
            else:
                return self.SERIF_REGULAR

        # Check for Monospace fonts
        if TextClassifier.is_mono_font(name_lower):
            return self.MONO_REGULAR

        # Default to Sans-Serif
        if is_bold and is_italic:
            return self.SANS_BOLD_ITALIC
        elif is_bold:
            return self.SANS_BOLD
        elif is_italic:
            return self.SANS_ITALIC
        else:
            return self.SANS_REGULAR

    def _select_font(self, font_name: str, font_size: float, is_bold: bool, is_italic: bool, custom_font_file: Optional[str] = None) -> ImageFont.FreeTypeFont:
        font_file = custom_font_file if (custom_font_file and os.path.exists(custom_font_file)) else self._select_font_file(font_name, is_bold, is_italic)
        try:
            return ImageFont.truetype(font_file, int(font_size * self.dpi_scale))
        except Exception:
            return ImageFont.load_default()

    @staticmethod
    def int_color_to_rgb(color_int: int) -> Tuple[int, int, int]:
        """
        Converts sRGB integer color value to RGB tuple.
        """
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
        baseline_offset: float = 0.0,
        text_color: int = 0,
        custom_font_file: Optional[str] = None
    ) -> bytes:
        """
        Renders a word string into a high-DPI transparent PNG byte buffer matching the exact font file.
        """
        canvas_w = max(1, int(bbox_width * self.dpi_scale))
        canvas_h = max(1, int(bbox_height * self.dpi_scale))

        # Create transparent RGBA canvas
        image = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))
        draw = ImageDraw.Draw(image)

        font = self._select_font(font_name, font_size, is_bold, is_italic, custom_font_file)
        rgb_color = self.int_color_to_rgb(text_color)
        fill_color = (rgb_color[0], rgb_color[1], rgb_color[2], 255)

        # Baseline point calculation
        baseline_y = baseline_offset * self.dpi_scale if baseline_offset > 0 else canvas_h * 0.78
        draw_pt = (0, baseline_y)

        # Render text onto transparent image canvas
        draw.text(draw_pt, word_text, font=font, fill=fill_color, anchor="ls")

        # Export image buffer as PNG
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        buffer.seek(0)
        return buffer.getvalue()
