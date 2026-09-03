import sys, os
sys.path.insert(0, os.path.abspath("."))
import pymupdf as fitz
from PIL import ImageFont
from pdf_processor.renderer import WordImageRenderer
from pdf_processor.classifier import TextClassifier


def find_metrics(word_bbox, page_dict, renderer, classifier):
    word_rect = fitz.Rect(word_bbox)
    best_span = None
    best_overlap = 0.0
    for block in page_dict.get("blocks", []):
        if block.get("type", 0) != 0: continue
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                srect = fitz.Rect(span.get("bbox", (0, 0, 0, 0)))
                intersection = word_rect & srect
                area = intersection.width * intersection.height
                if area > best_overlap:
                    best_overlap = area
                    best_span = span

    if best_span:
        fname = best_span.get("font", "")
        fsize = best_span.get("size", classifier.median_font_size)
        color = best_span.get("color", 0)
        flags = best_span.get("flags", 0)
        is_bold = classifier.is_bold_font(fname, flags)
        is_italic = classifier.is_italic_font(fname, flags)
        ffile = renderer._select_font_file(fname, is_bold, is_italic)
        origin = best_span.get("origin")
        baseline_y = origin[1] if origin else (word_bbox[1] + (word_bbox[3]-word_bbox[1])*0.78)
        baseline_offset = max(1.0, baseline_y - word_bbox[1])
        return {"font_name": fname, "font_file": ffile, "font_size": fsize, "color": color, "is_bold": is_bold, "is_italic": is_italic, "baseline_offset": baseline_offset, "baseline_y": baseline_y}
    return {"font_name": "", "font_file": renderer.SERIF_REGULAR, "font_size": classifier.median_font_size, "color": 0, "is_bold": False, "is_italic": False, "baseline_offset": (word_bbox[3]-word_bbox[1])*0.78, "baseline_y": word_bbox[1]+(word_bbox[3]-word_bbox[1])*0.78}


def inspect_body_paragraph():
    doc = fitz.open("main.pdf")
    page = doc[77]  # Page 78
    dict_data = page.get_text("dict")
    words = page.get_text("words")

    renderer = WordImageRenderer(dpi_scale=3.0)
    font_stats = TextClassifier.calculate_document_font_stats([dict_data])
    classifier = TextClassifier(doc_font_stats=font_stats)

    for w in words[100:130]:
        x0, y0, x1, y1, word_text = w[:5]
        clean_word = word_text.strip()
        if len(clean_word) < 3:
            continue

        w_rect = fitz.Rect(x0, y0, x1, y1)
        metrics = find_metrics((x0, y0, x1, y1), dict_data, renderer, classifier)

        print(f"Word: '{clean_word:15s}' | Font Name: {metrics['font_name']:10s} | is_bold: {metrics['is_bold']} | is_italic: {metrics['is_italic']} | Font File: {os.path.basename(metrics['font_file'])}")

    doc.close()


if __name__ == "__main__":
    inspect_body_paragraph()
