import pymupdf as fitz


def create_pdf_with_links(filename="sample_links.pdf"):
    doc = fitz.open()

    page1 = doc.new_page(width=595, height=842)
    page1.insert_text((50, 60), "Interactive PDF Test Document", fontsize=16)
    page1.insert_text((50, 115), "1. Section One", fontsize=14)

    para1 = "This is a body paragraph containing an interactive citation [1] that links to page 2."
    page1.insert_textbox(fitz.Rect(50, 130, 545, 180), para1, fontsize=10)

    page1.insert_text((50, 200), "Table of Contents", fontsize=12)
    page1.insert_text((50, 235), "1. Section One ...................... 1", fontsize=10)

    page2 = doc.new_page(width=595, height=842)
    page2.insert_text((50, 60), "References", fontsize=14)
    page2.insert_text((50, 115), "[1] Sample Reference entry on page 2.", fontsize=10)

    doc.save(filename)
    doc.close()

    # Re-open saved PDF to add links
    doc = fitz.open(filename)
    page1 = doc[0]
    
    # Add link from TOC to Section 1
    page1.insert_link({
        "kind": fitz.LINK_GOTO,
        "from": fitz.Rect(50, 220, 250, 240),
        "page": 0,
        "to": fitz.Point(50, 100)
    })

    # Add link from citation [1] to Reference on page 2
    page1.insert_link({
        "kind": fitz.LINK_GOTO,
        "from": fitz.Rect(380, 135, 400, 150),
        "page": 1,
        "to": fitz.Point(50, 100)
    })

    doc.save(filename, incremental=True, encryption=fitz.PDF_ENCRYPT_KEEP)
    doc.close()
    print(f"Created {filename} with interactive TOC & citation links.")


def test_link_preservation():
    create_pdf_with_links()

    doc = fitz.open("sample_links.pdf")
    page1 = doc[0]

    links_before = page1.get_links()
    print(f"Links before redaction: {len(links_before)}")
    for l in links_before:
        print(f"  Link: kind={l.get('kind')}, from={l.get('from')}, page={l.get('page')}")

    # Redact body word (excluding citation box)
    word_rect = fitz.Rect(50, 135, 100, 150)
    page1.add_redact_annot(word_rect, fill=(1, 1, 1))
    page1.apply_redactions()

    # Restore saved links
    for l in links_before:
        page1.insert_link(l)

    links_after = page1.get_links()
    print(f"\nLinks after redaction & restoration: {len(links_after)}")
    for l in links_after:
        print(f"  Link: kind={l.get('kind')}, from={l.get('from')}, page={l.get('page')}")

    doc.save("sample_links_processed.pdf")
    doc.close()
    print("Saved sample_links_processed.pdf")


if __name__ == "__main__":
    test_link_preservation()
