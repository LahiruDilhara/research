import pymupdf as fitz  # PyMuPDF
import random
import re
from typing import Optional, List, Tuple, Dict, Any
from pdf_processor.classifier import TextClassifier
from pdf_processor.renderer import WordImageRenderer
from pdf_processor.zero_width import ZeroWidthInjector
from pdf_processor.homoglyph import HomoglyphSubstitutor
from pdf_processor.disruption import LayoutDisruptor


class PDFPostProcessor:
    """
    Multi-stage pipeline engine for PDF post-processing.
    Stage 1: Replaces randomly selected body words with transparent rendered images.
    Stage 2: Substitutes Latin characters with visually identical Basic Cyrillic homoglyphs.
    Stage 3: Injects invisible zero-width Unicode characters into selectable body words (render_mode=3).
    Stage 4: Overlays invisible jumbled/scrambled text layers (render_mode=3) for stylized copy-paste disruption.
    Strictly isolates body paragraphs and preserves math equations, formulas, graphs, figures, tables, code blocks,
    titles, headings, TOC, references, headers, footers, and interactive links.
    """

    def __init__(
        self,
        input_path: str,
        output_path: str,
        probability: float = 0.15,
        homo_probability: float = 0.15,
        zw_probability: float = 0.15,
        disrupt_probability: float = 0.15,
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
        self.homo_probability = max(0.0, min(1.0, homo_probability))
        self.zw_probability = max(0.0, min(1.0, zw_probability))
        self.disrupt_probability = max(0.0, min(1.0, disrupt_probability))
        self.stage = (stage or "all").lower()
        self.min_word_len = min_word_len
        self.verbose = verbose
        
        if seed is not None:
            random.seed(seed)

        self.renderer = WordImageRenderer(font_path=font_path, dpi_scale=dpi_scale)
        self.homo_substitutor = HomoglyphSubstitutor(seed=seed)
        self.zw_injector = ZeroWidthInjector(seed=seed)
        self.disruptor = LayoutDisruptor(seed=seed)

    def _find_span_metrics(self, word_bbox: Tuple[float, float, float, float], page_dict: Dict[str, Any], default_size: float) -> Dict[str, Any]:
        """
        Extracts exact font properties for a target word bounding box using max-intersection area matching.
        """
        word_rect = fitz.Rect(word_bbox)
        best_span = None
        best_overlap = 0.0

        for block in page_dict.get("blocks", []):
            if block.get("type", 0) != 0:
                continue
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    span_rect = fitz.Rect(span.get("bbox", (0, 0, 0, 0)))
                    intersection = word_rect & span_rect
                    overlap_area = intersection.width * intersection.height

                    if overlap_area > best_overlap:
                        best_overlap = overlap_area
                        best_span = span

        if best_span:
            font_name = best_span.get("font", "")
            font_size = best_span.get("size", default_size)
            color = best_span.get("color", 0)
            flags = best_span.get("flags", 0)

            is_bold = TextClassifier.is_bold_font(font_name, flags)
            is_italic = TextClassifier.is_italic_font(font_name, flags)

            font_file = self.renderer._select_font_file(font_name, is_bold, is_italic)

            origin = best_span.get("origin")
            if origin:
                baseline_offset = max(1.0, origin[1] - word_bbox[1])
                baseline_y = origin[1]
            else:
                baseline_offset = (word_bbox[3] - word_bbox[1]) * 0.78
                baseline_y = word_bbox[1] + baseline_offset

            return {
                "font_name": font_name,
                "font_file": font_file,
                "font_size": font_size,
                "color": color,
                "is_bold": is_bold,
                "is_italic": is_italic,
                "baseline_offset": baseline_offset,
                "baseline_y": baseline_y
            }

        return {
            "font_name": "",
            "font_file": self.renderer.SERIF_REGULAR,
            "font_size": default_size,
            "color": 0,
            "is_bold": False,
            "is_italic": False,
            "baseline_offset": (word_bbox[3] - word_bbox[1]) * 0.78,
            "baseline_y": word_bbox[1] + (word_bbox[3] - word_bbox[1]) * 0.78
        }

    @staticmethod
    def _is_inside_link(word_rect: fitz.Rect, page_links: List[Dict[str, Any]]) -> bool:
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
        doc = fitz.open(self.input_path)
        total_pages = len(doc)

        pages_dict = [page.get_text("dict") for page in doc]
        font_stats = TextClassifier.calculate_document_font_stats(pages_dict)
        classifier = TextClassifier(doc_font_stats=font_stats)

        total_words_processed = 0
        total_image_replacements = 0
        total_homo_substitutions = 0
        total_zw_injections = 0
        total_layout_disruptions = 0

        run_stage1 = self.stage in ["all", "stage1"]
        run_stage2 = self.stage in ["all", "stage2"]
        run_stage3 = self.stage in ["all", "stage3"]
        run_stage4 = self.stage in ["all", "stage4"]

        for page_idx in range(total_pages):
            page = doc[page_idx]
            page_dict = pages_dict[page_idx]
            page_height = page.rect.height

            page_links = page.get_links()
            words = page.get_text("words")
            
            block_map = {}
            for block in page_dict.get("blocks", []):
                if block.get("type", 0) == 0:
                    block_no = block.get("number", -1)
                    full_block_text = ""
                    font_names = set()
                    span_sizes = []
                    for line in block.get("lines", []):
                        for span in line.get("spans", []):
                            full_block_text += span.get("text", "") + " "
                            font_names.add(span.get("font", ""))
                            span_sizes.append(span.get("size", font_stats.get("median_size", 10.0)))
                    
                    avg_size = (sum(span_sizes) / len(span_sizes)) if span_sizes else font_stats.get("median_size", 10.0)
                    block_map[block_no] = {
                        "bbox": block.get("bbox", (0, 0, 0, 0)),
                        "text": full_block_text.strip(),
                        "font_names": font_names,
                        "span_size": avg_size
                    }

            image_replacements: List[Tuple[fitz.Rect, str, Dict[str, Any]]] = []
            homo_substitutions: List[Tuple[fitz.Rect, str, str, Dict[str, Any]]] = []
            zw_injections: List[Tuple[fitz.Rect, str, str, Dict[str, Any]]] = []
            layout_disruptions: List[Tuple[fitz.Rect, str, str, Dict[str, Any]]] = []

            for w in words:
                x0, y0, x1, y1, word_text, block_no, line_no, word_no = w[:8]
                clean_word = word_text.strip()
                word_rect = fitz.Rect(x0, y0, x1, y1)

                stripped_word = clean_word.strip("(),.-[]\"':;!?")
                if len(stripped_word) < self.min_word_len or not re.search(r'[a-zA-Z]', stripped_word):
                    continue

                if self._is_inside_link(word_rect, page_links):
                    continue

                metrics = self._find_span_metrics((x0, y0, x1, y1), page_dict, classifier.median_font_size)

                if classifier.is_math_font(metrics["font_name"]):
                    continue

                if classifier.is_math_or_formula_text(clean_word):
                    continue

                if block_no in block_map:
                    b_info = block_map[block_no]
                    bbox = b_info["bbox"]
                    block_text = b_info["text"]

                    if classifier.is_header_or_footer(bbox, page_height, block_text):
                        continue
                    
                    if not classifier.is_body_paragraph(block_text, b_info["font_names"], b_info["span_size"]):
                        continue

                total_words_processed += 1

                # Stage 1 Sampling: Image Replacement
                is_selected = False
                if run_stage1 and random.random() < self.probability:
                    image_replacements.append((word_rect, clean_word, metrics))
                    total_image_replacements += 1
                    is_selected = True

                # Stage 2 Sampling: Homoglyph Substitution
                if run_stage2 and not is_selected and random.random() < self.homo_probability:
                    homo_word = self.homo_substitutor.substitute_word(clean_word)
                    homo_substitutions.append((word_rect, clean_word, homo_word, metrics))
                    total_homo_substitutions += 1
                    is_selected = True

                # Stage 3 Sampling: Zero-Width Character Injection
                if run_stage3 and not is_selected and random.random() < self.zw_probability:
                    zw_word = self.zw_injector.inject_into_word(clean_word)
                    zw_injections.append((word_rect, clean_word, zw_word, metrics))
                    total_zw_injections += 1
                    is_selected = True

                # Stage 4 Sampling: Stylized Layout Disruption
                if run_stage4 and not is_selected and random.random() < self.disrupt_probability:
                    disrupted_word = self.disruptor.disrupt_word(clean_word)
                    layout_disruptions.append((word_rect, clean_word, disrupted_word, metrics))
                    total_layout_disruptions += 1

            # Execute Stage 1
            if image_replacements:
                for rect, word_text, metrics in image_replacements:
                    page.add_redact_annot(rect, fill=(1, 1, 1))

                page.apply_redactions()

                for rect, word_text, metrics in image_replacements:
                    font_file = metrics["font_file"]
                    image_bytes = self.renderer.render_word_image(
                        word_text=word_text,
                        bbox_width=rect.width,
                        bbox_height=rect.height,
                        font_name=metrics["font_name"],
                        font_size=metrics["font_size"],
                        is_bold=metrics["is_bold"],
                        is_italic=metrics["is_italic"],
                        baseline_offset=metrics["baseline_offset"],
                        text_color=metrics["color"],
                        custom_font_file=font_file
                    )
                    page.insert_image(rect, stream=image_bytes)

            # Execute Stage 2
            if homo_substitutions:
                for rect, word_text, homo_word, metrics in homo_substitutions:
                    page.add_redact_annot(rect, fill=(1, 1, 1))

                page.apply_redactions()

                for rect, word_text, homo_word, metrics in homo_substitutions:
                    font_file = metrics["font_file"]
                    rgb_color = self.renderer.int_color_to_rgb(metrics["color"])
                    color_tuple = (rgb_color[0] / 255.0, rgb_color[1] / 255.0, rgb_color[2] / 255.0)
                    
                    baseline_pt = fitz.Point(rect.x0, metrics["baseline_y"])
                    font_key = f"homo_font_{page_idx}_{metrics['is_bold']}_{metrics['is_italic']}"
                    
                    page.insert_text(
                        baseline_pt,
                        homo_word,
                        fontsize=metrics["font_size"],
                        fontname=font_key,
                        fontfile=font_file,
                        color=color_tuple
                    )

            # Execute Stage 3
            if zw_injections:
                for rect, word_text, zw_word, metrics in zw_injections:
                    page.add_redact_annot(rect, fill=(1, 1, 1))

                page.apply_redactions()

                for rect, word_text, zw_word, metrics in zw_injections:
                    font_file = metrics["font_file"]
                    rgb_color = self.renderer.int_color_to_rgb(metrics["color"])
                    color_tuple = (rgb_color[0] / 255.0, rgb_color[1] / 255.0, rgb_color[2] / 255.0)
                    
                    baseline_pt = fitz.Point(rect.x0, metrics["baseline_y"])
                    font_key = f"zw_font_{page_idx}_{metrics['is_bold']}_{metrics['is_italic']}"
                    
                    page.insert_text(
                        baseline_pt,
                        word_text,
                        fontsize=metrics["font_size"],
                        fontname=font_key,
                        fontfile=font_file,
                        color=color_tuple
                    )

                    page.insert_text(
                        baseline_pt,
                        zw_word,
                        fontsize=metrics["font_size"],
                        fontname=font_key,
                        fontfile=font_file,
                        render_mode=3
                    )

            # Execute Stage 4
            if layout_disruptions:
                for rect, word_text, disrupted_word, metrics in layout_disruptions:
                    page.add_redact_annot(rect, fill=(1, 1, 1))

                page.apply_redactions()

                for rect, word_text, disrupted_word, metrics in layout_disruptions:
                    font_file = metrics["font_file"]
                    rgb_color = self.renderer.int_color_to_rgb(metrics["color"])
                    color_tuple = (rgb_color[0] / 255.0, rgb_color[1] / 255.0, rgb_color[2] / 255.0)
                    
                    baseline_pt = fitz.Point(rect.x0, metrics["baseline_y"])
                    font_key = f"disrupt_font_{page_idx}_{metrics['is_bold']}_{metrics['is_italic']}"
                    
                    page.insert_text(
                        baseline_pt,
                        word_text,
                        fontsize=metrics["font_size"],
                        fontname=font_key,
                        fontfile=font_file,
                        color=color_tuple
                    )

                    page.insert_text(
                        baseline_pt,
                        disrupted_word,
                        fontsize=metrics["font_size"],
                        fontname=font_key,
                        fontfile=font_file,
                        render_mode=3
                    )

            # Re-bind interactive links
            if image_replacements or homo_substitutions or zw_injections or layout_disruptions:
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
                    print(
                        f"Page {page_idx + 1}/{total_pages}: {len(image_replacements)} images, "
                        f"{len(homo_substitutions)} homoglyphs, {len(zw_injections)} zero-width, "
                        f"{len(layout_disruptions)} layout disruptions."
                    )

        doc.save(self.output_path, garbage=4, deflate=True)
        doc.close()

        summary = {
            "total_pages": total_pages,
            "total_words_processed": total_words_processed,
            "total_image_replacements": total_image_replacements,
            "total_homo_substitutions": total_homo_substitutions,
            "total_zw_injections": total_zw_injections,
            "total_layout_disruptions": total_layout_disruptions,
            "output_path": self.output_path
        }
        return summary
