import pymupdf as fitz


def verify_processed_pdf(pdf_path="output_test.pdf"):
    doc = fitz.open(pdf_path)
    print(f"\n--- Verifying Output PDF: {pdf_path} ---")
    print(f"Page count: {len(doc)}")

    for idx, page in enumerate(doc):
        image_list = page.get_images()
        text = page.get_text()
        print(f"\n[Page {idx + 1}] Embedded Images Count: {len(image_list)}")
        print(f"Sample Text snippet:\n{text[:200]}...")

    doc.close()


if __name__ == "__main__":
    verify_processed_pdf()
