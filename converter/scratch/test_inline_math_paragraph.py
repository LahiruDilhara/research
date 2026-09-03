import sys, os
sys.path.insert(0, os.path.abspath("."))
import pymupdf as fitz
from pdf_processor.classifier import TextClassifier


def find_target_paragraph():
    doc = fitz.open("main.pdf")
    target_page = None
    target_block = None

    for page_idx in range(len(doc)):
        page = doc[page_idx]
        text = page.get_text()
        if "Fingertip pixel movement" in text or "unitless hand scale normalization" in text:
            target_page = page_idx
            print(f"Found target paragraph on Page {page_idx+1}!")
            dict_data = page.get_text("dict")
            for block in dict_data.get("blocks", []):
                btext = ""
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        btext += span.get("text", "") + " "
                if "Fingertip pixel movement" in btext or "unitless hand scale" in btext:
                    target_block = block
                    print(f"Block Text:\n{btext.strip()}\n")
            break

    doc.close()
    return target_page, target_block


if __name__ == "__main__":
    find_target_paragraph()
