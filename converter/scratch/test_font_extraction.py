import sys, os
sys.path.insert(0, os.path.abspath("."))
import pymupdf as fitz


def test_fonts_in_pdf():
    doc = fitz.open("main.pdf")
    page = doc[7]  # Page 8 (Chapter 1)
    
    font_list = page.get_fonts()
    print("Page 8 embedded fonts:")
    for f in font_list:
        xref, ext, type_str, font_name, key, encoding = f[:6]
        print(f"  xref: {xref:4d} | Ext: {ext:5s} | Type: {type_str:5s} | Name: {font_name:20s}")
        
        # Try extracting font buffer
        if xref > 0:
            font_info = doc.extract_font(xref)
            name, ext_found, subtype, buffer = font_info
            print(f"    Extracted: {name} | Ext: {ext_found} | Size: {len(buffer)} bytes")

    doc.close()


if __name__ == "__main__":
    test_fonts_in_pdf()
