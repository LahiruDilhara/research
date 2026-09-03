import sys, os
sys.path.insert(0, os.path.abspath("."))
import pymupdf as fitz
from pdf_processor.classifier import TextClassifier


def inspect_bold_detection():
    doc = fitz.open("main.pdf")

    # Check pages 8 to 15
    for page_idx in range(7, 15):
        page = doc[page_idx]
        page_dict = page.get_text("dict")
        
        print(f"\n================ PAGE {page_idx+1} ================")
        for block in page_dict.get("blocks", []):
            if block.get("type", 0) != 0:
                continue
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    font_name = span.get("font", "")
                    flags = span.get("flags", 0)
                    text = span.get("text", "").strip()

                    if not text:
                        continue

                    is_bold = TextClassifier.is_bold_font(font_name, flags)
                    fn_lower = font_name.lower()
                    has_bold_in_name = any(b in fn_lower for b in ["bold", "cmb", "bx", "bld"])

                    if is_bold and not has_bold_in_name:
                        print(f"  [MISMATCH?] Font: {font_name:20s} | Flags: {flags:3d} | Text: '{text[:35]}'")
                    elif is_bold:
                        print(f"  [TRUE BOLD ] Font: {font_name:20s} | Flags: {flags:3d} | Text: '{text[:35]}'")

    doc.close()


if __name__ == "__main__":
    inspect_bold_detection()
