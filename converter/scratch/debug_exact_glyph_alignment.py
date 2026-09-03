import sys, os
sys.path.insert(0, os.path.abspath("."))
import pymupdf as fitz
from PIL import ImageFont, Image, ImageDraw
from pdf_processor.renderer import WordImageRenderer
from pdf_processor.classifier import TextClassifier


def inspect_line_spans():
    doc = fitz.open("main.pdf")
    page = doc[5]  # Page 6 (Introduction page from user screenshot)
    pdict = page.get_text("dict")

    print("--- Page 6 Text Spans & Word Metrics ---")

    for block in pdict.get("blocks", []):
        if block.get("type", 0) != 0:
            continue
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                stext = span.get("text", "").strip()
                if "Physical computer keyboards" in stext or "drawbacks" in stext or "tactile" in stext:
                    print(f"Span Text: '{span.get('text')}'")
                    print(f"  Font: {span.get('font')} | Size: {span.get('size'):.2f}pt | Flags: {span.get('flags')}")
                    print(f"  BBox: {span.get('bbox')}")
                    print(f"  Origin: {span.get('origin')}\n")

    doc.close()


if __name__ == "__main__":
    inspect_line_spans()
