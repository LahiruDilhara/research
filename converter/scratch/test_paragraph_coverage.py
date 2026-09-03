import sys, os
sys.path.insert(0, os.path.abspath("."))
import pymupdf as fitz
from pdf_processor.classifier import TextClassifier
from pdf_processor.processor import PDFPostProcessor


def inspect_paragraph_coverage():
    doc = fitz.open("main.pdf")
    pages_dict = [page.get_text("dict") for page in doc]
    font_stats = TextClassifier.calculate_document_font_stats(pages_dict)
    classifier = TextClassifier(doc_font_stats=font_stats)
    print(f"Median font size: {classifier.median_font_size}")

    # Inspect pages 1 to 10
    for page_idx in range(10):
        page = doc[page_idx]
        page_dict = pages_dict[page_idx]
        page_height = page.rect.height
        
        print(f"\n================ PAGE {page_idx+1} ================")
        blocks = page_dict.get("blocks", [])
        
        for block_no, block in enumerate(blocks):
            if block.get("type", 0) != 0:
                continue
                
            full_text = ""
            fonts = set()
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    full_text += span.get("text", "") + " "
                    fonts.add(span.get("font", ""))
            
            full_text = full_text.strip()
            bbox = block.get("bbox", (0, 0, 0, 0))
            word_count = len(full_text.split())

            is_hf = classifier.is_header_or_footer(bbox, page_height, full_text)
            is_body = classifier.is_body_paragraph(full_text, fonts, classifier.median_font_size)

            print(f"Block #{block_no} ({word_count} w) | H/F: {is_hf} | Body: {is_body} | Text: '{full_text[:60]}...'")

    doc.close()


if __name__ == "__main__":
    inspect_paragraph_coverage()
