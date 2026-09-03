import sys, os
sys.path.insert(0, os.path.abspath("."))
import pymupdf as fitz
from pdf_processor.classifier import TextClassifier


def test_why_false():
    doc = fitz.open("main.pdf")
    page = doc[7]  # Page 8 (index 7)
    dict_data = page.get_text("dict")
    
    font_stats = TextClassifier.calculate_document_font_stats([dict_data])
    classifier = TextClassifier(doc_font_stats=font_stats)

    for block_no, block in enumerate(dict_data.get("blocks", [])):
        if block.get("type", 0) != 0:
            continue
        
        full_text = ""
        fonts = set()
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                full_text += span.get("text", "") + " "
                fonts.add(span.get("font", ""))

        full_text = full_text.strip()
        word_count = len(full_text.split())

        print(f"\n--- Block #{block_no} ({word_count} words) ---")
        print(f"Text snippet: '{full_text[:80]}...'")
        print(f"Fonts: {fonts}")
        print(f"is_math_font: {any(classifier.is_math_font(fn) for fn in fonts)}")
        print(f"is_math_or_formula_text: {classifier.is_math_or_formula_text(full_text)}")
        print(f"is_mono_font: {any(classifier.is_mono_font(fn) for fn in fonts)}")
        print(f"is_reference_heading_or_block: {classifier.is_reference_heading_or_block(full_text)}")
        print(f"is_toc_line: {classifier.is_toc_line(full_text)}")
        print(f"is_caption: {classifier.is_caption(full_text)}")

    doc.close()


if __name__ == "__main__":
    test_why_false()
