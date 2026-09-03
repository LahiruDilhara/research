import pymupdf as fitz


def inspect_baseline():
    doc = fitz.open("sample_test.pdf")
    page = doc[0]
    page_dict = page.get_text("dict")

    for block in page_dict.get("blocks", []):
        if block.get("type", 0) != 0:
            continue
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                text = span.get("text", "").strip()
                if text and len(text) > 10:
                    bbox = span.get("bbox")
                    origin = span.get("origin")
                    font = span.get("font")
                    size = span.get("size")
                    flags = span.get("flags")
                    
                    baseline_offset = origin[1] - bbox[1]
                    total_height = bbox[3] - bbox[1]
                    ratio = baseline_offset / total_height if total_height > 0 else 0
                    
                    print(f"Text: '{text[:25]}...' | Font: {font} | Size: {size} | Flags: {flags}")
                    print(f"  bbox: {bbox}")
                    print(f"  origin: {origin}")
                    print(f"  baseline_offset: {baseline_offset:.2f} pt / total_height: {total_height:.2f} pt (ratio: {ratio:.3f})\n")

    doc.close()


if __name__ == "__main__":
    inspect_baseline()
