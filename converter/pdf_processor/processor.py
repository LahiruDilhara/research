import pymupdf as fitz  # PyMuPDF
import random
from typing import Optional, List, Tuple, Dict, Any
from pdf_processor.classifier import TextClassifier
from pdf_processor.renderer import WordImageRenderer
from pdf_processor.zero_width import ZeroWidthInjector


class PDFPostProcessor:
    """
    Multi-stage pipeline engine for PDF post-processing.
    Stage 1: Replaces randomly selected body words with transparent rendered images.
    Stage 2: Injects invisible zero-width Unicode characters into selectable body words (using render_mode=3).
    Preserves document structure (titles, TOC, citations, references, headers, footers) and interactive links.
    """

    def __init__(
        self,
        input_path: str,
        output_path: str,
        probability: float = 0.15,
        zw_probability: float = 0.15,
        stage: str = "all",
        seed: Optional[int] = None,
        min_word_len: int = 3,
        dpi_scale: float = 3.0,
        font_path: Optional[str] = None,
        verbose: bool = False
    ):
        self.input_path = input_path
        self.output_path = output_path
        self.probability = max(0.0, min(1.0, probability))
        self.zw_probability = max(0.0, min(1.0, zw_probability))
        self.stage = (stage or "all").lower()
        self.min_word_len = min_word_len
        self.verbose = verbose
        
        if seed is not None:
            random.seed(seed)

        self.renderer = WordImageRenderer(font_path=font_path, dpi_scale=dpi_scale)
        self.zw_injector = ZeroWidthInjector(seed=seed)

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
                            baseline_y = origin[1]
                        else:
                            baseline_offset = (wy1 - wy0) * 0.78
                            baseline_y = wy0 + baseline_offset

                        return {
                            "font_name": font_name,
                            "font_size": font_size,
                            "color": color,
                            "is_bold": is_bold,
                            "is_italic": is_italic,
                            "baseline_offset": baseline_offset,
                            "baseline_y": baseline_y
                        }

        return {
            "font_name": "",
            "font_size": default_size,
            "color": 0,
            "is_bold": False,
            "is_italic": False,
            "baseline_offset": (wy1 - wy0) * 0.78,
            "baseline_y": wy0 + (wy1 - wy0) * 0.78
        }

    @staticmethod
    def _is_inside_link(word_rect: fitz.Rect, page_links: List[Dict[str, Any]]) -> bool:
        """
        Checks if a word bounding box intersects with any interactive link annotation
        (e.g., clickable Table of Contents items, clickable citations, or external URIs).
        """
        for link in page_links:
            link_rect = link.get("from")
            if link_rect and not (
                word_rect.x1 < link_rect.x0 or
                word_rect.x0 > link_rect.x1 or
                word_rect.y1 < link_rect.y0 or
                word_rect.y0 > link_rect.y1
            ):
                return True
        return False

    def process(self) -> Dict[str, Any]:
        """
        Executes multi-stage PDF post-processing on the document.
        """
        doc = fitz.open(self.input_path)
        total_pages = len(doc)
        
        # Pre-pass for document-wide font statistics
        pages_dict = [page.get_text("dict") for page in doc]
        font_stats = TextClassifier.calculate_document_font_stats(pages_dict)
        classifier = TextClassifier(doc_font_stats=font_stats)

        total_words_processed = 0
        total_image_replacements = 0
        total_zw_injections = 0

        run_stage1 = self.stage in ["all", "stage1"]
        run_stage2 = self.stage in ["all", "stage2"]

        # Process page by page
        for page_idx in range(total_pages):
            page = doc[page_idx]
            page_dict = pages_dict[page_idx]
            page_height = page.rect.height

            # Retrieve all existing interactive page links
            page_links = page.get_links()

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

            image_replacements: List[Tuple[fitz.Rect, str, Dict[str, Any]]] = []
            zw_injections: List[Tuple[fitz.Rect, str, str, Dict[str, Any]]] = []

            for w in words:
                x0, y0, x1, y1, word_text, block_no, line_no, word_no = w[:8]
                clean_word = word_text.strip()
                word_rect = fitz.Rect(x0, y0, x1, y1)

                if len(clean_word) < self.min_word_len:
                    continue

                # Protect words inside interactive link regions (TOC, clickable citations, URIs)
                if self._is_inside_link(word_rect, page_links):
                    continue

                # Get font metrics
                metrics = self._find_span_metrics((x0, y0, x1, y1), page_dict, classifier.median_font_size)

                # Check if block is body paragraph
                if block_no in block_map:
                    bbox, block_text = block_map[block_no]
                    
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

                # Stage 1 Sampling: Image Replacement
                is_stage1_selected = False
                if run_stage1 and random.random() < self.probability:
                    image_replacements.append((word_rect, clean_word, metrics))
                    total_image_replacements += 1
                    is_stage1_selected = True

                # Stage 2 Sampling: Zero-Width Character Injection (for remaining selectable body words)
                if run_stage2 and not is_stage1_selected and random.random() < self.zw_probability:
                    zw_word = self.zw_injector.inject_into_word(clean_word)
                    zw_injections.append((word_rect, clean_word, zw_word, metrics))
                    total_zw_injections += 1

            # Execute Stage 1 (Image Replacements)
            if image_replacements:
                for rect, word_text, metrics in image_replacements:
                    page.add_redact_annot(rect, fill=(1, 1, 1))

                page.apply_redactions()

                for rect, word_text, metrics in image_replacements:
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

            # Execute Stage 2 (Zero-Width Injections)
            if zw_injections:
                for rect, word_text, zw_word, metrics in zw_injections:
                    page.add_redact_annot(rect, fill=(1, 1, 1))

                page.apply_redactions()

                for rect, word_text, zw_word, metrics in zw_injections:
                    font_file = self.renderer._select_font_file(
                        metrics["font_name"], metrics["is_bold"], metrics["is_italic"]
                    )
                    rgb_color = self.renderer.int_color_to_rgb(metrics["color"])
                    color_tuple = (rgb_color[0] / 255.0, rgb_color[1] / 255.0, rgb_color[2] / 255.0)
                    
                    baseline_pt = fitz.Point(rect.x0, metrics["baseline_y"])
                    font_key = f"zw_font_{page_idx}_{metrics['is_bold']}_{metrics['is_italic']}"
                    
                    # 1. Insert clean visible word text
                    page.insert_text(
                        baseline_pt,
                        word_text,
                        fontsize=metrics["font_size"],
                        fontname=font_key,
                        fontfile=font_file,
                        color=color_tuple
                    )

                    # 2. Insert zero-width character text in INVISIBLE mode (render_mode=3)
                    page.insert_text(
                        baseline_pt,
                        zw_word,
                        fontsize=metrics["font_size"],
                        fontname=font_key,
                        fontfile=font_file,
                        render_mode=3
                    )

            # Re-bind interactive links if any were modified by redactions
            if image_replacements or zw_injections:
                existing_links = page.get_links()
                existing_rects = [l.get("from") for l in existing_links if l.get("from")]

                for link in page_links:
                    link_rect = link.get("from")
                    if link_rect and link_rect not in existing_rects:
                        try:
                            page.insert_link(link)
                        except Exception:
                            pass

                if self.verbose:
                    print(f"Page {page_idx + 1}/{total_pages}: {len(image_replacements)} image replacements, {len(zw_injections)} zero-width injections.")

        # Save processed PDF
        doc.save(self.output_path, garbage=4, deflate=True)
        doc.close()

        summary = {
            "total_pages": total_pages,
            "total_words_processed": total_words_processed,
            "total_image_replacements": total_image_replacements,
            "total_zw_injections": total_zw_injections,
            "output_path": self.output_path
        }
        return summary
