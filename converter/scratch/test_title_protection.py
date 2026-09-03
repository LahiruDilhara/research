import sys, os
sys.path.insert(0, os.path.abspath("."))
import pymupdf as fitz
from pdf_processor.classifier import TextClassifier


def test_full_doc_classification():
    doc = fitz.open("main.pdf")
    pages_dict = [page.get_text("dict") for page in doc]
    
    font_stats = TextClassifier.calculate_document_font_stats(pages_dict)
    classifier = TextClassifier(doc_font_stats=font_stats)

    print(f"Full Document Median Font Size: {classifier.median_font_size:.2f} pt")

    # Check Page 1 (Title Page)
    page1 = doc[0]
    p1_dict = pages_dict[0]
    print("\n--- PAGE 1 (Title Page) Blocks ---")
    for block in p1_dict.get("blocks", []):
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
        
        is_title = classifier.is_title_or_heading(avg_size, 0, list(font_names)[0] if font_names else "", full_text)
        is_body = classifier.is_body_paragraph(full_text, font_names, avg_size)

        print(f"Block {block.get('number')}: Size: {avg_size:.2f}pt | Title: {is_title} | Body: {is_body}")
        print(f"  Text: '{full_text.strip()[:70]}'")

    doc.close()


if __name__ == "__main__":
    test_full_doc_classification()
