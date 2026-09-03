import sys, os
sys.path.insert(0, os.path.abspath("."))
import pymupdf as fitz
from pdf_processor.classifier import TextClassifier
from pdf_processor.processor import PDFPostProcessor


def test_page78_word_processing():
    processor = PDFPostProcessor("main.pdf", "tmp_out.pdf")
    doc = fitz.open("main.pdf")
    pages_dict = [page.get_text("dict") for page in doc]
    
    font_stats = TextClassifier.calculate_document_font_stats(pages_dict)
    classifier = TextClassifier(doc_font_stats=font_stats)

    page = doc[77]  # Page 78
    page_dict = pages_dict[77]
    words = page.get_text("words")

    print("--- Word-Level Math Filter Check on Page 78 ---")

    for w in words:
        x0, y0, x1, y1, word_text, block_no, line_no, word_no = w[:8]
        clean_word = word_text.strip()
        
        stripped_word = clean_word.strip("(),.-[]\"':;!?")
        if len(stripped_word) < 3 or not any(c.isalpha() for c in stripped_word):
            continue

        metrics = processor._find_span_metrics((x0, y0, x1, y1), page_dict, 11.96)
        font_name = metrics["font_name"]

        is_math = classifier.is_math_font(font_name) or classifier.is_math_or_formula_text(clean_word)

        if any(kw in clean_word for kw in ["Fingertip", "webcam", "Lhand", "Landmark", "distance", "camera", "perspective", "focal"]):
            print(f"Word: {clean_word:20s} | Font: {font_name:10s} | Math Filter: {is_math}")

    doc.close()


if __name__ == "__main__":
    test_page78_word_processing()
