import pymupdf as fitz
from PIL import Image, ImageDraw, ImageFont
import io


def test_updated_rendering():
    doc = fitz.open("sample_test.pdf")
    page = doc[0]
    words = page.get_text("words")
    page_dict = page.get_text("dict")

    # Pick words from body text
    body_words = [w for w in words if w[5] == 1 and len(w[4].strip()) >= 4]
    
    dpi = 3.0

    print(f"Testing {len(body_words)} words...")

    replacements = []

    for w in body_words[:10]:
        x0, y0, x1, y1, text, b_no, l_no, w_no = w[:8]
        word_rect = fitz.Rect(x0, y0, x1, y1)
        clean_text = text.strip()

        # Find font, size, color, baseline origin
        span_font = "helvetica"
        font_size = 10.0
        text_color = 0
        baseline_offset = word_rect.height * 0.78
        is_bold = False
        is_italic = False

        for block in page_dict.get("blocks", []):
            if block.get("type", 0) != 0:
                continue
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    sx0, sy0, sx1, sy1 = span.get("bbox", (0, 0, 0, 0))
                    if not (x1 < sx0 or x0 > sx1 or y1 < sy0 or y0 > sy1):
                        font_size = span.get("size", 10.0)
                        text_color = span.get("color", 0)
                        span_font = span.get("font", "").lower()
                        flags = span.get("flags", 0)
                        is_bold = bool(flags & 16) or ("bold" in span_font)
                        is_italic = bool(flags & 2) or ("italic" in span_font or "oblique" in span_font)
                        
                        origin = span.get("origin")
                        if origin:
                            baseline_offset = max(1.0, origin[1] - y0)
                        break

        # Font resolution path
        if "times" in span_font or "serif" in span_font or "roman" in span_font:
            if is_bold and is_italic:
                ttf_path = "/usr/share/fonts/truetype/liberation/LiberationSerif-BoldItalic.ttf"
            elif is_bold:
                ttf_path = "/usr/share/fonts/truetype/liberation/LiberationSerif-Bold.ttf"
            elif is_italic:
                ttf_path = "/usr/share/fonts/truetype/liberation/LiberationSerif-Italic.ttf"
            else:
                ttf_path = "/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf"
        elif "mono" in span_font or "courier" in span_font:
            ttf_path = "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf"
        else:
            if is_bold and is_italic:
                ttf_path = "/usr/share/fonts/truetype/liberation/LiberationSans-BoldItalic.ttf"
            elif is_bold:
                ttf_path = "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"
            elif is_italic:
                ttf_path = "/usr/share/fonts/truetype/liberation/LiberationSans-Italic.ttf"
            else:
                ttf_path = "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"

        # Precise canvas sizing
        scaled_w = int(word_rect.width * dpi)
        scaled_h = int(word_rect.height * dpi)

        img = Image.new("RGBA", (scaled_w, scaled_h), (255, 255, 255, 0))
        draw = ImageDraw.Draw(img)

        pil_font_size = int(font_size * dpi)
        font = ImageFont.truetype(ttf_path, pil_font_size)

        # Baseline position in canvas
        pil_baseline_y = int(baseline_offset * dpi)

        # Extract RGB color
        r = (text_color >> 16) & 0xFF
        g = (text_color >> 8) & 0xFF
        b = text_color & 0xFF

        draw.text((0, pil_baseline_y), clean_text, fill=(r, g, b, 255), font=font, anchor="ls")

        buffer = io.BytesIO()
        img.save(buffer, format="PNG", dpi=(int(72 * dpi), int(72 * dpi)))
        replacements.append((word_rect, buffer.getvalue()))

    # Apply redactions
    for word_rect, img_bytes in replacements:
        page.add_redact_annot(word_rect, fill=(1, 1, 1))
    page.apply_redactions()

    # Insert transparent PNGs
    for word_rect, img_bytes in replacements:
        page.insert_image(word_rect, stream=img_bytes)

    doc.save("test_perfect_aligned.pdf")
    doc.close()
    print("Saved test_perfect_aligned.pdf")


if __name__ == "__main__":
    test_updated_rendering()
