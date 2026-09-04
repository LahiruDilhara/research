import pymupdf as fitz  # PyMuPDF
import random
import re
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Optional, List, Tuple, Dict, Any

from pdf_processor.classifier import TextClassifier
from pdf_processor.renderer import WordImageRenderer
from pdf_processor.zero_width import ZeroWidthInjector
from pdf_processor.homoglyph import HomoglyphSubstitutor
from pdf_processor.disruption import LayoutDisruptor


def _process_single_page_worker(task_args: Tuple[str, int, Dict[str, float], Dict[str, Any]]) -> Tuple[int, bytes, Dict[str, int]]:
    """
    Top-level worker function for parallel multi-process PDF page processing.
    Executes Stages 1-4 on a single PDF page in isolation.
    """
    input_path, page_idx, doc_font_stats, config = task_args

    seed = config.get("seed")
    page_seed = (seed + page_idx * 10007) if seed is not None else None
    if page_seed is not None:
        random.seed(page_seed)

    renderer = WordImageRenderer(font_path=config.get("font_path"), dpi_scale=config.get("dpi_scale", 2.0))
    homo_substitutor = HomoglyphSubstitutor(seed=page_seed)
    zw_injector = ZeroWidthInjector(seed=page_seed)
    disruptor = LayoutDisruptor(seed=page_seed)
    classifier = TextClassifier(doc_font_stats=doc_font_stats)

    doc = fitz.open(input_path)
    page = doc[page_idx]
    page_dict = page.get_text("dict")
    page_height = page.rect.height
    page_links = page.get_links()
    words = page.get_text("words")

    stage = config.get("stage", "all")
    run_stage1 = stage in ["all", "stage1"]
    run_stage2 = stage in ["all", "stage2"]
    run_stage3 = stage in ["all", "stage3"]
    run_stage4 = stage in ["all", "stage4"]

    prob1 = config.get("probability", 0.15)
    prob2 = config.get("homo_probability", 0.15)
    prob3 = config.get("zw_probability", 0.15)
    prob4 = config.get("disrupt_probability", 0.15)
    min_word_len = config.get("min_word_len", 3)
    max_images_per_page = config.get("max_images_per_page", 0)
    max_images_per_para = config.get("max_images_per_para", 0)
    zw_count = config.get("zw_count", 2)
    disrupt_multiplier = config.get("disrupt_multiplier", 1.5)

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
                    span_sizes.append(span.get("size", doc_font_stats.get("median_size", 10.0)))
            
            avg_size = (sum(span_sizes) / len(span_sizes)) if span_sizes else doc_font_stats.get("median_size", 10.0)
            block_map[block_no] = {
                "bbox": block.get("bbox", (0, 0, 0, 0)),
                "text": full_block_text.strip(),
                "font_names": font_names,
                "span_size": avg_size
            }

    image_replacements = []
    homo_substitutions = []
    zw_injections = []
    layout_disruptions = []

    words_scanned = 0
    page_images_count = 0
    para_images_count = {}

    def is_inside_link(word_rect):
        for link in page_links:
            lrect = link.get("from")
            if lrect and not (word_rect.x1 < lrect.x0 or word_rect.x0 > lrect.x1 or word_rect.y1 < lrect.y0 or word_rect.y0 > lrect.y1):
                return True
        return False

    def find_metrics(word_bbox):
        word_rect = fitz.Rect(word_bbox)
        best_span = None
        best_overlap = 0.0
        for block in page_dict.get("blocks", []):
            if block.get("type", 0) != 0:
                continue
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    srect = fitz.Rect(span.get("bbox", (0, 0, 0, 0)))
                    intersection = word_rect & srect
                    area = intersection.width * intersection.height
                    if area > best_overlap:
                        best_overlap = area
                        best_span = span

        if best_span:
            fname = best_span.get("font", "")
            fsize = best_span.get("size", classifier.median_font_size)
            color = best_span.get("color", 0)
            flags = best_span.get("flags", 0)
            is_bold = classifier.is_bold_font(fname, flags)
            is_italic = classifier.is_italic_font(fname, flags)
            ffile = renderer._select_font_file(fname, is_bold, is_italic)
            origin = best_span.get("origin")
            baseline_y = origin[1] if origin else (word_bbox[1] + (word_bbox[3] - word_bbox[1]) * 0.78)
            baseline_offset = max(1.0, baseline_y - word_bbox[1])
            return {
                "font_name": fname,
                "font_file": ffile,
                "font_size": fsize,
                "color": color,
                "is_bold": is_bold,
                "is_italic": is_italic,
                "baseline_offset": baseline_offset,
                "baseline_y": baseline_y
            }
        return {
            "font_name": "",
            "font_file": renderer.SERIF_REGULAR,
            "font_size": classifier.median_font_size,
            "color": 0,
            "is_bold": False,
            "is_italic": False,
            "baseline_offset": (word_bbox[3] - word_bbox[1]) * 0.78,
            "baseline_y": word_bbox[1] + (word_bbox[3] - word_bbox[1]) * 0.78
        }

    for w in words:
        x0, y0, x1, y1, word_text, block_no, line_no, word_no = w[:8]
        clean_word = word_text.strip()
        word_rect = fitz.Rect(x0, y0, x1, y1)

        stripped = clean_word.strip("(),.-[]\"':;!?")
        if len(stripped) < min_word_len or not re.search(r'[a-zA-Z]', stripped):
            continue

        if is_inside_link(word_rect):
            continue

        metrics = find_metrics((x0, y0, x1, y1))
        if classifier.is_math_font(metrics["font_name"]) or classifier.is_math_or_formula_text(clean_word):
            continue

        if block_no in block_map:
            binfo = block_map[block_no]
            if classifier.is_header_or_footer(binfo["bbox"], page_height, binfo["text"]):
                continue
            if not classifier.is_body_paragraph(binfo["text"], binfo["font_names"], binfo["span_size"]):
                continue

        words_scanned += 1

        can_add_image = run_stage1
        if can_add_image and max_images_per_page > 0 and page_images_count >= max_images_per_page:
            can_add_image = False
        para_count = para_images_count.get(block_no, 0)
        if can_add_image and max_images_per_para > 0 and para_count >= max_images_per_para:
            can_add_image = False

        is_selected = False
        if can_add_image and random.random() < prob1:
            image_replacements.append((word_rect, clean_word, metrics))
            page_images_count += 1
            para_images_count[block_no] = para_count + 1
            is_selected = True

        if run_stage2 and not is_selected and random.random() < prob2:
            homo_word = homo_substitutor.substitute_word(clean_word)
            homo_substitutions.append((word_rect, clean_word, homo_word, metrics))
            is_selected = True

        if run_stage3 and not is_selected and random.random() < prob3:
            zw_word = zw_injector.inject_into_word(clean_word, zw_count=zw_count)
            zw_injections.append((word_rect, clean_word, zw_word, metrics))
            is_selected = True

        if run_stage4 and not is_selected and random.random() < prob4:
            disrupted_word = disruptor.disrupt_word(clean_word, length_multiplier=disrupt_multiplier)
            layout_disruptions.append((word_rect, clean_word, disrupted_word, metrics))

    # Stage 1 Execution with 0-Distortion Bounding Box Placement
    if image_replacements:
        for rect, word_text, metrics in image_replacements:
            page.add_redact_annot(rect, fill=(1, 1, 1))
        page.apply_redactions()

        pad_pt = 2.0 / renderer.dpi_scale
        for rect, word_text, metrics in image_replacements:
            font_file = metrics["font_file"]
            img_bytes, glyph_w, glyph_h, top_offset = renderer.render_word_image(
                word_text=word_text,
                font_name=metrics["font_name"],
                font_size=metrics["font_size"],
                is_bold=metrics["is_bold"],
                is_italic=metrics["is_italic"],
                text_color=metrics["color"],
                custom_font_file=font_file
            )

            baseline_y = metrics["baseline_y"]
            insert_rect = fitz.Rect(
                rect.x0 - pad_pt,
                baseline_y + top_offset,
                rect.x0 - pad_pt + glyph_w,
                baseline_y + top_offset + glyph_h
            )
            page.insert_image(insert_rect, stream=img_bytes)

    # Stage 2 Execution
    if homo_substitutions:
        for rect, word_text, homo_word, metrics in homo_substitutions:
            page.add_redact_annot(rect, fill=(1, 1, 1))
        page.apply_redactions()

        for rect, word_text, homo_word, metrics in homo_substitutions:
            rgb = renderer.int_color_to_rgb(metrics["color"])
            color_tuple = (rgb[0] / 255.0, rgb[1] / 255.0, rgb[2] / 255.0)
            baseline_pt = fitz.Point(rect.x0, metrics["baseline_y"])
            font_key = f"homo_font_{page_idx}_{metrics['is_bold']}_{metrics['is_italic']}"
            page.insert_text(baseline_pt, homo_word, fontsize=metrics["font_size"], fontname=font_key, fontfile=metrics["font_file"], color=color_tuple)

    # Stage 3 Execution
    if zw_injections:
        for rect, word_text, zw_word, metrics in zw_injections:
            page.add_redact_annot(rect, fill=(1, 1, 1))
        page.apply_redactions()

        for rect, word_text, zw_word, metrics in zw_injections:
            rgb = renderer.int_color_to_rgb(metrics["color"])
            color_tuple = (rgb[0] / 255.0, rgb[1] / 255.0, rgb[2] / 255.0)
            baseline_pt = fitz.Point(rect.x0, metrics["baseline_y"])
            font_key = f"zw_font_{page_idx}_{metrics['is_bold']}_{metrics['is_italic']}"
            page.insert_text(baseline_pt, word_text, fontsize=metrics["font_size"], fontname=font_key, fontfile=metrics["font_file"], color=color_tuple)
            page.insert_text(baseline_pt, zw_word, fontsize=metrics["font_size"], fontname=font_key, fontfile=metrics["font_file"], render_mode=3)

    # Stage 4 Execution
    if layout_disruptions:
        for rect, word_text, disrupted_word, metrics in layout_disruptions:
            page.add_redact_annot(rect, fill=(1, 1, 1))
        page.apply_redactions()

        for rect, word_text, disrupted_word, metrics in layout_disruptions:
            rgb = renderer.int_color_to_rgb(metrics["color"])
            color_tuple = (rgb[0] / 255.0, rgb[1] / 255.0, rgb[2] / 255.0)
            baseline_pt = fitz.Point(rect.x0, metrics["baseline_y"])
            font_key = f"disrupt_font_{page_idx}_{metrics['is_bold']}_{metrics['is_italic']}"
            page.insert_text(baseline_pt, word_text, fontsize=metrics["font_size"], fontname=font_key, fontfile=metrics["font_file"], color=color_tuple)
            page.insert_text(baseline_pt, disrupted_word, fontsize=metrics["font_size"], fontname=font_key, fontfile=metrics["font_file"], render_mode=3)

    # Re-bind interactive links
    if image_replacements or homo_substitutions or zw_injections or layout_disruptions:
        existing_links = page.get_links()
        existing_rects = [l.get("from") for l in existing_links if l.get("from")]
        for link in page_links:
            lrect = link.get("from")
            if lrect and lrect not in existing_rects:
                try:
                    page.insert_link(link)
                except Exception:
                    pass

    single_doc = fitz.open()
    single_doc.insert_pdf(doc, from_page=page_idx, to_page=page_idx)
    page_bytes = single_doc.tobytes(garbage=0, deflate=True)
    single_doc.close()
    doc.close()

    stats = {
        "words_scanned": words_scanned,
        "images": len(image_replacements),
        "homo": len(homo_substitutions),
        "zw": len(zw_injections),
        "disrupt": len(layout_disruptions)
    }
    return (page_idx, page_bytes, stats)


class PDFPostProcessor:
    """
    Multi-stage pipeline engine for PDF post-processing.
    Supports sequential or multi-process parallel processing across PDF pages.
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
        dpi_scale: float = 2.0,
        max_images_per_page: int = 0,
        max_images_per_para: int = 0,
        zw_count: int = 2,
        disrupt_multiplier: float = 1.5,
        workers: int = 1,
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
        self.seed = seed
        self.min_word_len = min_word_len
        self.dpi_scale = max(1.0, dpi_scale)
        self.max_images_per_page = max(0, max_images_per_page)
        self.max_images_per_para = max(0, max_images_per_para)
        self.zw_count = max(1, zw_count)
        self.disrupt_multiplier = max(1.0, disrupt_multiplier)
        
        cpu_count = os.cpu_count() or 1
        if workers == 0:
            self.workers = cpu_count
        else:
            self.workers = max(1, workers)

        self.font_path = font_path
        self.verbose = verbose

    def process(self) -> Dict[str, Any]:
        doc = fitz.open(self.input_path)
        total_pages = len(doc)

        pages_dict = [page.get_text("dict") for page in doc]
        font_stats = TextClassifier.calculate_document_font_stats(pages_dict)
        doc.close()

        config = {
            "probability": self.probability,
            "homo_probability": self.homo_probability,
            "zw_probability": self.zw_probability,
            "disrupt_probability": self.disrupt_probability,
            "stage": self.stage,
            "seed": self.seed,
            "min_word_len": self.min_word_len,
            "dpi_scale": self.dpi_scale,
            "max_images_per_page": self.max_images_per_page,
            "max_images_per_para": self.max_images_per_para,
            "zw_count": self.zw_count,
            "disrupt_multiplier": self.disrupt_multiplier,
            "font_path": self.font_path
        }

        task_args = [
            (self.input_path, page_idx, font_stats, config)
            for page_idx in range(total_pages)
        ]

        total_words_processed = 0
        total_image_replacements = 0
        total_homo_substitutions = 0
        total_zw_injections = 0
        total_layout_disruptions = 0

        processed_pages: List[Tuple[int, bytes, Dict[str, int]]] = []

        if self.workers > 1 and total_pages > 1:
            if self.verbose:
                print(f"Processing {total_pages} pages in parallel using {self.workers} worker processes...")
            
            with ProcessPoolExecutor(max_workers=self.workers) as executor:
                futures = [executor.submit(_process_single_page_worker, t_arg) for t_arg in task_args]
                for future in as_completed(futures):
                    res = future.result()
                    processed_pages.append(res)
                    p_idx, _, stats = res
                    if self.verbose:
                        print(
                            f"Page {p_idx + 1}/{total_pages}: {stats['images']} images, "
                            f"{stats['homo']} homoglyphs, {stats['zw']} zero-width, "
                            f"{stats['disrupt']} layout disruptions."
                        )
        else:
            if self.verbose:
                print(f"Processing {total_pages} pages sequentially...")
            for t_arg in task_args:
                res = _process_single_page_worker(t_arg)
                processed_pages.append(res)
                p_idx, _, stats = res
                if self.verbose:
                    print(
                        f"Page {p_idx + 1}/{total_pages}: {stats['images']} images, "
                        f"{stats['homo']} homoglyphs, {stats['zw']} zero-width, "
                        f"{stats['disrupt']} layout disruptions."
                    )

        processed_pages.sort(key=lambda x: x[0])

        final_doc = fitz.open()
        for p_idx, page_bytes, stats in processed_pages:
            total_words_processed += stats["words_scanned"]
            total_image_replacements += stats["images"]
            total_homo_substitutions += stats["homo"]
            total_zw_injections += stats["zw"]
            total_layout_disruptions += stats["disrupt"]

            page_doc = fitz.open("pdf", page_bytes)
            final_doc.insert_pdf(page_doc)
            page_doc.close()

        final_doc.save(self.output_path, garbage=1, deflate=True)
        final_doc.close()

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
