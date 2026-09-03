import sys, os, io, time
sys.path.insert(0, os.path.abspath("."))
import pymupdf as fitz
from PIL import Image, ImageDraw, ImageFont
from pdf_processor.renderer import WordImageRenderer
from pdf_processor.classifier import TextClassifier
from pdf_processor.processor import PDFPostProcessor


def test_compression():
    print("--- Testing Image Caching & PDF Compression Optimization ---")
    start_time = time.time()

    # Run processor with DPI scale 2.5 and measure output size
    processor = PDFPostProcessor(
        input_path="main.pdf",
        output_path="output_compressed.pdf",
        probability=0.7,
        homo_probability=0.0,
        zw_probability=0.0,
        disrupt_probability=0.0,
        seed=42,
        dpi_scale=2.5,
        workers=8,
        verbose=False
    )

    summary = processor.process()
    elapsed = time.time() - start_time

    out_size_mb = os.path.getsize("output_compressed.pdf") / (1024 * 1024)
    orig_size_mb = os.path.getsize("main.pdf") / (1024 * 1024)

    print(f"Processed {summary['total_pages']} pages ({summary['total_image_replacements']} images) in {elapsed:.2f}s")
    print(f"Original PDF Size: {orig_size_mb:.2f} MB")
    print(f"Output PDF Size:   {out_size_mb:.2f} MB")

    if os.path.exists("output_compressed.pdf"):
        os.remove("output_compressed.pdf")


if __name__ == "__main__":
    test_compression()
