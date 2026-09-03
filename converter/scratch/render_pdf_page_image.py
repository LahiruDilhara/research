import sys, os
sys.path.insert(0, os.path.abspath("."))
import pymupdf as fitz

def render_page_preview():
    doc = fitz.open("output_sample.pdf")
    page = doc[0]
    pix = page.get_pixmap(dpi=150)
    out_img = "/home/lahirukasunidilhara/.gemini/antigravity-ide/brain/207ff07b-7d48-4451-a4f9-abb3697a3234/.tempmediaStorage/p07_crisp_preview.png"
    os.makedirs(os.path.dirname(out_img), exist_ok=True)
    pix.save(out_img)
    print(f"Page 1 preview saved to '{out_img}'")
    doc.close()

if __name__ == "__main__":
    render_page_preview()
