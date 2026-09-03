import sys, os
sys.path.insert(0, os.path.abspath("."))
import pymupdf as fitz
from pdf_processor.classifier import TextClassifier


def test_page78_blocks():
    doc = fitz.open("main.pdf")
    pages_dict = [page.get_text("dict") for page in doc]
    
    font_stats = TextClassifier.calculate_document_font_stats(pages_dict)
    classifier = TextClassifier(doc_font_stats=font_stats)

    page = doc[77]  # Page 78
    dict_data = pages_dict[77]

    print("--- Page 78 Blocks Classification ---")
    for block in dict_data.get("blocks", []):
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
        
        is_body = classifier.is_body_paragraph(full_text, font_names, avg_size)

        if "Fingertip" in full_text or "unitless" in full_text or "Landmark" in full_text:
            print(f"\nTARGET BLOCK: Size {avg_size:.2f}pt | is_body_paragraph: {is_body}")
            print(f"Font Names in Block: {font_names}")
            print(f"Text Snippet: '{full_text.strip()[:100]}'")

    doc.close()


if __name__ == "__main__":
    test_page78_blocks()
