import sys, os
sys.path.insert(0, os.path.abspath("."))
import pymupdf as fitz


def test_extracted_rendering():
    doc = fitz.open("main.pdf")
    page = doc[7]  # Page 8 (Chapter 1)

    # Extract CMR12 font xref 691
    font_info = doc.extract_font(691)
    name, ext, subtype, buffer = font_info
    
    # Write to temp font file
    temp_font_path = f"/tmp/{name}.{ext}"
    with open(temp_font_path, "wb") as f:
        f.write(buffer)

    print(f"Saved extracted font to {temp_font_path} ({len(buffer)} bytes)")

    # Test PyMuPDF insert_text with extracted font
    test_doc = fitz.open()
    test_page = test_doc.new_page(width=595, height=842)
    
    test_page.insert_text(
        fitz.Point(50, 100),
        "Human-Computer Interaction (HCI) presents a customizable, paper-based virtual keyboard system.",
        fontsize=12,
        fontname="cmr_font",
        fontfile=temp_font_path,
        color=(0, 0, 0)
    )

    test_doc.save("test_extracted_render.pdf")
    test_doc.close()
    doc.close()

    # Render PNG
    test_doc = fitz.open("test_extracted_render.pdf")
    pix = test_doc[0].get_pixmap(dpi=150)
    pix.save("test_extracted_render.png")
    test_doc.close()

    print("Rendered test_extracted_render.png successfully!")


if __name__ == "__main__":
    test_extracted_rendering()
