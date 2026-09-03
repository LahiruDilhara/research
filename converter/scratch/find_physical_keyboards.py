import sys, os
sys.path.insert(0, os.path.abspath("."))
import pymupdf as fitz

doc = fitz.open("main.pdf")
for page_idx in range(len(doc)):
    text = doc[page_idx].get_text()
    if "Physical computer keyboards" in text or "fifty years" in text:
        print(f"Found on Page {page_idx + 1}!")
        pdict = doc[page_idx].get_text("dict")
        for block in pdict.get("blocks", []):
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    stext = span.get("text", "")
                    if "Physical" in stext or "keyboards" in stext or "fifty" in stext:
                        print(f"  Span: '{stext}' | Font: {span.get('font')} | Size: {span.get('size'):.2f}pt | BBox: {span.get('bbox')} | Origin: {span.get('origin')}")
        break
doc.close()
