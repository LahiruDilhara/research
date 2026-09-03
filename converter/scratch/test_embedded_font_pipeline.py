import sys, os
sys.path.insert(0, os.path.abspath("."))
import pymupdf as fitz


def build_font_cache(doc, cache_dir="/tmp/pdf_fonts"):
    os.makedirs(cache_dir, exist_ok=True)
    cache = {}

    for page_idx in range(len(doc)):
        page = doc[page_idx]
        for font_info in page.get_fonts():
            xref, ext, type_str, font_name = font_info[0], font_info[1], font_info[2], font_info[3]
            if font_name in cache or xref <= 0:
                continue

            try:
                name_extracted, ext_extracted, subtype, buffer = doc.extract_font(xref)
                if buffer and len(buffer) > 100:
                    clean_name = font_name.replace("+", "_").replace(" ", "_")
                    font_file_path = os.path.join(cache_dir, f"{clean_name}.{ext_extracted}")
                    with open(font_file_path, "wb") as f:
                        f.write(buffer)
                    cache[font_name] = font_file_path
                    # Also map prefix-stripped name e.g. CMR12
                    base_name = font_name.split("+")[-1]
                    cache[base_name] = font_file_path
            except Exception as e:
                pass

    return cache


def test_font_cache_on_main():
    doc = fitz.open("main.pdf")
    cache = build_font_cache(doc)
    print(f"Extracted and cached {len(cache)} font files:")
    for k, v in cache.items():
        print(f"  Font Key: {k:25s} -> {v}")

    doc.close()


if __name__ == "__main__":
    test_font_cache_on_main()
