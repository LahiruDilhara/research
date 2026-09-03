import pymupdf as fitz


def render_page(pdf_path="output_refined.pdf", out_png="out_page1.png"):
    doc = fitz.open(pdf_path)
    page = doc[0]
    pix = page.get_pixmap(dpi=150)
    pix.save(out_png)
    doc.close()
    print(f"Saved rendered page to {out_png}")


if __name__ == "__main__":
    render_page()
