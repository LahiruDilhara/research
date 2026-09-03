import pymupdf as fitz


def verify_links_preserved(pdf_path="output_links_test.pdf"):
    doc = fitz.open(pdf_path)
    page1 = doc[0]
    links = page1.get_links()
    print(f"\n--- Verifying {pdf_path} ---")
    print(f"Total interactive links found on Page 1: {len(links)}")
    for i, l in enumerate(links):
        print(f"  Link {i+1}: kind={l.get('kind')}, rect={l.get('from')}, page={l.get('page')}")

    doc.close()


if __name__ == "__main__":
    verify_links_preserved()
