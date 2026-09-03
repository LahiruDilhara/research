import sys, os
sys.path.insert(0, os.path.abspath("."))
import pymupdf as fitz
from pdf_processor.classifier import TextClassifier


def find_span_metrics_exact(word_bbox, page_dict, default_size=10.0):
    word_rect = fitz.Rect(word_bbox)
    best_span = None
    best_overlap = 0.0

    for block in page_dict.get("blocks", []):
        if block.get("type", 0) != 0:
            continue
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                span_rect = fitz.Rect(span.get("bbox", (0, 0, 0, 0)))
                intersection = word_rect & span_rect
                overlap_area = intersection.width * intersection.height

                if overlap_area > best_overlap:
                    best_overlap = overlap_area
                    best_span = span

    if best_span:
        font_name = best_span.get("font", "")
        font_size = best_span.get("size", default_size)
        color = best_span.get("color", 0)
        flags = best_span.get("flags", 0)

        is_bold = TextClassifier.is_bold_font(font_name, flags)
        is_italic = TextClassifier.is_italic_font(font_name, flags)

        origin = best_span.get("origin")
        if origin:
            baseline_offset = max(1.0, origin[1] - word_bbox[1])
            baseline_y = origin[1]
        else:
            baseline_offset = (word_bbox[3] - word_bbox[1]) * 0.78
            baseline_y = word_bbox[1] + baseline_offset

        return {
            "font_name": font_name,
            "font_size": font_size,
            "color": color,
            "is_bold": is_bold,
            "is_italic": is_italic,
            "baseline_offset": baseline_offset,
            "baseline_y": baseline_y
        }

    return {
        "font_name": "",
        "font_size": default_size,
        "color": 0,
        "is_bold": False,
        "is_italic": False,
        "baseline_offset": (word_bbox[3] - word_bbox[1]) * 0.78,
        "baseline_y": word_bbox[1] + (word_bbox[3] - word_bbox[1]) * 0.78
    }


def test_page14_metrics():
    doc = fitz.open("main.pdf")
    page = doc[13]  # Page 14
    page_dict = page.get_text("dict")
    words = page.get_text("words")

    print("Checking words on Page 14:")
    for w in words[:35]:
        wx0, wy0, wx1, wy1, word_text = w[:5]
        metrics = find_span_metrics_exact((wx0, wy0, wx1, wy1), page_dict)
        print(f"Word: {word_text:20s} | Font: {metrics['font_name']:12s} | Bold: {metrics['is_bold']}")

    doc.close()


if __name__ == "__main__":
    test_page14_metrics()
