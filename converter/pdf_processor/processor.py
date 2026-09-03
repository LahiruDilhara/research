import pymupdf as fitz  # PyMuPDF
import random
from typing import Optional, List, Tuple, Dict, Any
from pdf_processor.classifier import TextClassifier
from pdf_processor.renderer import WordImageRenderer


class PDFPostProcessor:
    """
    Main engine for PDF post-processing.
    Filters document structure to identify body paragraphs, randomly selects words,
    renders selected words as transparent PNG images matching exact font metrics,
    redacts original text, and inserts images at exact locations.
    """

    def __init__(
        self,
        input_path: str,
        output_path: str,
        probability: float = 0.15,
        seed: Optional[int] = None,
        min_word_len: int = 3,
        dpi_scale: float = 3.0,
        font_path: Optional[str] = None,
        verbose: bool = False
    ):
        self.input_path = input_path
        self.output_path = output_path
        self.probability = max(0.0, min(1.0, probability))
        self.min_word_len = min_word_len
        self.verbose = verbose
        
        if seed is not None:
            random.seed(seed)

        self.renderer = WordImageRenderer(font_path=font_path, dpi_scale=dpi_scale)

    def _find_span_metrics(self, word_bbox: Tuple[float, float, float, float], page_dict: Dict[str, Any], default_size: float) -> Dict[str, Any]:
        """
        Extracts rich font properties (font name, size, color, bold, italic, baseline origin)
        for a target word bounding box by matching against PyMuPDF page spans.
        """
        wx0, wy0, wx1, wy1 = word_bbox
        for block in page_dict.get("blocks", []):
            if block.get("type", 0) != 0:
                continue
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    sx0, sy0, sx1, sy1 = span.get("bbox", (0, 0, 0, 0))
                    # Overlap check
                    if not (wx1 < sx0 or wx0 > sx1 or wy1 < sy0 or wy0 > sy1):
                        font_name = span.get("font", "")
                        font_size = span.get("size", default_size)
                        color = span.get("color", 0)
                        flags = span.get("flags", 0)

                        is_bold = bool(flags & 16) or ("bold" in font_name.lower())
                        is_italic = bool(flags & 2) or ("italic" in font_name.lower() or "oblique" in font_name.lower())

                        origin = span.get("origin")
                        if origin:
                            baseline_offset = max(1.0, origin[1] - wy0)
                        else:
                            baseline_offset = (wy1 - wy0) * 0.78

                        return {
                            "font_name": font_name,
                            "font_size": font_size,
                            "color": color,
                            "is_bold": is_bold,
                            "is_italic": is_italic,
                            "baseline_offset": baseline_offset
                        }

        return {
            "font_name": "",
            "font_size": default_size,
            "color": 0,
            "is_bold": False,
            "is_italic": False,
            "baseline_offset": (wy1 - wy0) * 0.78
        }

    def process(self) -> Dict[str, Any]:
        """
        Executes end-to-end processing on the PDF document.
        """
        doc = fitz.open(self.input_path)
        total_pages = len(doc)
        
        # Step 1: Pre-pass to extract document-wide font statistics
        pages_dict = [page.get_text("dict") for page in doc]
        font_stats = TextClassifier.calculate_document_font_stats(pages_dict)
        classifier = TextClassifier(doc_font_stats=font_stats)

        total_words_processed = 0
        total_words_replaced = 0

        # Step 2: Process page by page
        for page_idx in range(total_pages):
            page = doc[page_idx]
            page_dict = pages_dict[page_idx]
            page_height = page.rect.height

            # Extract words: (x0, y0, x1, y1, word_text, block_no, line_no, word_no)
            words = page.get_text("words")
            
            # Map block_no to block text for classification
            block_map = {}
            for block in page_dict.get("blocks", []):
                if block.get("type", 0) == 0:
                    block_no = block.get("number", -1)
                    full_block_text = ""
                    for line in block.get("lines", []):
                        for span in line.get("spans", []):
                            full_block_text += span.get("text", "") + " "
                    block_map[block_no] = (block.get("bbox", (0, 0, 0, 0)), full_block_text.strip())

            replacements_on_page: List[Tuple[fitz.Rect, str, Dict[str, Any]]] = []

            for w in words:
                x0, y0, x1, y1, word_text, block_no, line_no, word_no = w[:8]
                clean_word = word_text.strip()
                
                if len(clean_word) < self.min_word_len:
                    continue

                # Get font metrics
                metrics = self._find_span_metrics((x0, y0, x1, y1), page_dict, classifier.median_font_size)

                # Check if block is body paragraph
                if block_no in block_map:
                    bbox, block_text = block_map[block_no]
                    
                    # Classifier checks
                    if classifier.is_header_or_footer(bbox, page_height, block_text):
                        continue
                    if classifier.is_reference_heading_or_block(block_text):
                        continue
                    if classifier.is_toc_line(block_text):
                        continue
                    if classifier.is_caption(block_text):
                        continue
                    if classifier.is_title_or_heading(metrics["font_size"], 16 if metrics["is_bold"] else 0, block_text):
                        continue

                # Skip words inside inline citations e.g. [1] or (Smith, 2020)
                if clean_word.startswith("[") or clean_word.endswith("]") or clean_word.startswith("("):
                    continue

                total_words_processed += 1

                # Random probability selection
                if random.random() < self.probability:
                    word_rect = fitz.Rect(x0, y0, x1, y1)
                    replacements_on_page.append((word_rect, clean_word, metrics))
                    total_words_replaced += 1

            # Perform redaction and image overlay for selected words on this page
            if replacements_on_page:
                # Add redaction annotations to remove original vector text cleanly
                for rect, word_text, metrics in replacements_on_page:
                    page.add_redact_annot(rect, fill=(1, 1, 1))

                # Apply redactions to wipe text
                page.apply_redactions()

                # Insert transparent PNG rendered image for each word
                for rect, word_text, metrics in replacements_on_page:
                    image_bytes = self.renderer.render_word_image(
                        word_text=word_text,
                        bbox_width=rect.width,
                        bbox_height=rect.height,
                        font_name=metrics["font_name"],
                        font_size=metrics["font_size"],
                        is_bold=metrics["is_bold"],
                        is_italic=metrics["is_italic"],
                        baseline_offset=metrics["baseline_offset"],
                        text_color=metrics["color"]
                    )
                    page.insert_image(rect, stream=image_bytes)

                if self.verbose:
                    print(f"Page {page_idx + 1}/{total_pages}: Replaced {len(replacements_on_page)} words with images.")

        # Save processed PDF
        doc.save(self.output_path, garbage=4, deflate=True)
        doc.close()

        summary = {
            "total_pages": total_pages,
            "total_words_processed": total_words_processed,
            "total_words_replaced": total_words_replaced,
            "output_path": self.output_path
        }
        return summary
