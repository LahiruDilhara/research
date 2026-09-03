import sys, os
sys.path.insert(0, os.path.abspath("."))
import pymupdf as fitz
from pdf_processor.classifier import TextClassifier


def is_pure_code_block(font_names: set, block_text: str) -> bool:
    """
    Detects pure multi-line code blocks where all text spans use typewriter font.
    Allows narrative paragraphs containing brief inline code snippets (e.g. Ctrl + C).
    """
    if not font_names:
        return False
    if all(TextClassifier.is_mono_font(fn) for fn in font_names):
        return True
    return False


def test_page36_classification():
    doc = fitz.open("main.pdf")
    pages_dict = [page.get_text("dict") for page in doc]
    font_stats = TextClassifier.calculate_document_font_stats(pages_dict)
    classifier = TextClassifier(doc_font_stats=font_stats)

    page36 = pages_dict[35]
    print("--- Page 36 Blocks ---")
    for block in page36.get("blocks", []):
        if block.get("type", 0) != 0:
            continue
        
        full_text = ""
        font_names = set()
        span_sizes = []
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                full_text += span.get("text", "") + " "
                font_names.add(span.get("font", ""))
                span_sizes.append(span.get("size", 10.0))

        avg_size = (sum(span_sizes) / len(span_sizes)) if span_sizes else 10.0
        
        is_title = classifier.is_title_or_heading(avg_size, 0, font_names, full_text)
        is_pure_code = is_pure_code_block(font_names, full_text)
        
        # Test refined body condition
        is_body = (
            len(full_text.split()) >= 8
            and not is_title
            and not is_pure_code
            and not classifier.is_standalone_equation_block(full_text)
            and not classifier.is_reference_heading_or_block(full_text)
            and not classifier.is_toc_line(full_text)
            and not classifier.is_caption(full_text)
        )

        if "Inability" in full_text:
            print(f"TARGET PARAGRAPH: Size {avg_size:.2f}pt | Title: {is_title} | Pure Code: {is_pure_code} | Body: {is_body}")
            print(f"Fonts: {font_names}")
            print(f"Text: '{full_text.strip()[:100]}'")

    doc.close()


if __name__ == "__main__":
    test_page36_classification()
