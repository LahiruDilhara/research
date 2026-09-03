import sys, os
sys.path.insert(0, os.path.abspath("."))
import pymupdf as fitz
from pdf_processor.classifier import TextClassifier


def is_bold_font_strict(font_name: str, flags: int) -> bool:
    fn = font_name.lower()
    # Explicit bold keywords in font name
    if any(b in fn for b in ["bold", "cmb", "bx", "bld"]):
        return True
    return False


def is_italic_font_strict(font_name: str, flags: int) -> bool:
    fn = font_name.lower()
    if any(i in fn for i in ["italic", "oblique", "cmti", "lmti", "mi", "it"]):
        return True
    return bool(flags & 2)


def test_strict_font_matching():
    doc = fitz.open("main.pdf")
    page = doc[7]  # Page 8
    dict_data = page.get_text("dict")

    bold_count = 0
    regular_count = 0

    for block in dict_data.get("blocks", []):
        if block.get("type", 0) != 0:
            continue
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                font_name = span.get("font", "")
                flags = span.get("flags", 0)
                text = span.get("text", "").strip()

                if not text:
                    continue

                b = is_bold_font_strict(font_name, flags)
                if b:
                    bold_count += 1
                else:
                    regular_count += 1

    print(f"Page 8 strict bold spans: {bold_count} | regular spans: {regular_count}")
    doc.close()


if __name__ == "__main__":
    test_strict_font_matching()
