import sys, os
sys.path.insert(0, os.path.abspath("."))
import pymupdf as fitz
from pdf_processor.processor import PDFPostProcessor


def run_test_p07():
    input_path = "main.pdf"
    output_path = "output_p07_test.pdf"

    print("Running PDFPostProcessor with -p 0.7...")
    processor = PDFPostProcessor(
        input_path=input_path,
        output_path=output_path,
        probability=0.7,
        homo_probability=0.0,
        zw_probability=0.0,
        disrupt_probability=0.0,
        workers=4,
        seed=42,
        verbose=True
    )
    summary = processor.process()
    print(f"Done! Processed {summary['total_image_replacements']} images.")

if __name__ == "__main__":
    run_test_p07()
