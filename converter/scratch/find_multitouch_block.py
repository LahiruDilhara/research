import sys, os
sys.path.insert(0, os.path.abspath("."))
import pymupdf as fitz
from pdf_processor.classifier import TextClassifier


def find_paragraph():
    doc = fitz.open("main.pdf")
    pages_dict = [page.get_text("dict") for page in doc]
    font_stats = TextClassifier.calculate_document_font_stats(pages_dict)
    classifier = TextClassifier(doc_font_stats=font_stats)

    for page_idx in range(len(doc)):
        page = doc[page_idx]
        text = page.get_text()
        if "Inability to Detect Simultaneous" in text or "Traditional monocular camera keyboards" in text:
            print(f"Found paragraph on Page {page_idx+1}!")
            pdict = pages_dict[page_idx]
            for block in pdict.get("blocks", []):
                btext = ""
                font_names = set()
                span_sizes = []
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        btext += span.get("text", "") + " "
                        font_names.add(span.get("font", ""))
                        span_sizes.append(span.get("size", 10.0))

                if "Inability to Detect" in btext or "Traditional monocular" in btext:
                    avg_size = (sum(span_sizes) / len(span_sizes)) if span_sizes else 10.0
                    is_title = classifier.is_title_or_heading(avg_size, 0, font_names, btext)
                    is_body = classifier.is_body_paragraph(btext, font_names, avg_size)

                    print(f"Block Number: {block.get('number')}")
                    print(f"Avg Size: {avg_size:.2f}pt | Median Size: {classifier.median_font_size:.2f}pt")
                    print(f"is_title_or_heading: {is_title}")
                    print(f"is_body_paragraph: {is_body}")
                    print(f"Font Names: {font_names}")
                    print(f"Full Text:\n'{btext.strip()}'\n")

    doc.close()


if __name__ == "__main__":
    find_paragraph()
