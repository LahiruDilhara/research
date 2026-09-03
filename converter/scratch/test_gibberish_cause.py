import sys, os
sys.path.insert(0, os.path.abspath("."))
import pymupdf as fitz


def test_rendering_comparison():
    doc = fitz.open("main.pdf")
    page = doc[7]
    
    # Extract CMR12
    font_info = doc.extract_font(691)
    name, ext, subtype, buffer = font_info
    pfa_path = f"/tmp/{name}.{ext}"
    with open(pfa_path, "wb") as f:
        f.write(buffer)

    ttf_serif = "/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf"

    test_doc = fitz.open()
    tpage = test_doc.new_page(width=595, height=842)

    # Line 1 with PFA (TeX font)
    tpage.insert_text(fitz.Point(50, 100), "Human-Computer Interaction (HCI)", fontsize=12, fontfile=pfa_path)

    # Line 2 with TTF (LiberationSerif)
    tpage.insert_text(fitz.Point(50, 150), "Human-Computer Interaction (HCI)", fontsize=12, fontfile=ttf_serif)

    test_doc.save("gibberish_compare.pdf")
    test_doc.close()
    doc.close()

    # Extract text from both
    test_doc = fitz.open("gibberish_compare.pdf")
    print("Page text from compare PDF:")
    print(repr(test_doc[0].get_text()))
    test_doc.close()


if __name__ == "__main__":
    test_rendering_comparison()
