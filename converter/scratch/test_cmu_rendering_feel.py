import sys, os
sys.path.insert(0, os.path.abspath("."))
import pymupdf as fitz
from pdf_processor.processor import PDFPostProcessor


def test_chapter1_cmu_rendering():
    # Process only page 8 (Chapter 1) with seed 42
    processor = PDFPostProcessor(
        input_path="main.pdf",
        output_path="test_cmu_out.pdf",
        probability=0.25,
        homo_probability=0.15,
        zw_probability=0.15,
        disrupt_probability=0.15,
        seed=42,
        verbose=True
    )
    
    # Process doc
    doc = fitz.open("main.pdf")
    page8_doc = fitz.open()
    page8_doc.insert_pdf(doc, from_page=7, to_page=7)
    page8_doc.save("test_page8_input.pdf")
    page8_doc.close()
    doc.close()

    proc_p8 = PDFPostProcessor(
        input_path="test_page8_input.pdf",
        output_path="test_page8_output.pdf",
        probability=0.25,
        homo_probability=0.15,
        zw_probability=0.15,
        disrupt_probability=0.15,
        seed=42,
        verbose=True
    )
    proc_p8.process()

    # Render PNG of page 8 output
    doc_out = fitz.open("test_page8_output.pdf")
    pix = doc_out[0].get_pixmap(dpi=150)
    pix.save("test_page8_output.png")
    doc_out.close()

    print("Rendered test_page8_output.png successfully!")


if __name__ == "__main__":
    test_chapter1_cmu_rendering()
