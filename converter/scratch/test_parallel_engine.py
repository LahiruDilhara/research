import sys, os, time, random, re
sys.path.insert(0, os.path.abspath("."))
from typing import Dict, Any, Tuple, List, Optional
import pymupdf as fitz
from concurrent.futures import ProcessPoolExecutor

from pdf_processor.classifier import TextClassifier
from pdf_processor.renderer import WordImageRenderer
from pdf_processor.zero_width import ZeroWidthInjector
from pdf_processor.homoglyph import HomoglyphSubstitutor
from pdf_processor.disruption import LayoutDisruptor


def _process_page_worker(task_args: Tuple[str, int, Dict[str, float], Dict[str, Any]]) -> Tuple[int, bytes, Dict[str, int]]:
    input_path, page_idx, doc_font_stats, config = task_args

    seed = config.get("seed")
    page_seed = (seed + page_idx * 10007) if seed is not None else None
    if page_seed is not None:
        random.seed(page_seed)

    renderer = WordImageRenderer(font_path=config.get("font_path"), dpi_scale=config.get("dpi_scale", 3.0))
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
            if block.get("type", 0) != 0: continue
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
            baseline_y = origin[1] if origin else (word_bbox[1] + (word_bbox[3]-word_bbox[1])*0.78)
            baseline_offset = max(1.0, baseline_y - word_bbox[1])
            return {"font_name": fname, "font_file": ffile, "font_size": fsize, "color": color, "is_bold": is_bold, "is_italic": is_italic, "baseline_offset": baseline_offset, "baseline_y": baseline_y}
        return {"font_name": "", "font_file": renderer.SERIF_REGULAR, "font_size": classifier.median_font_size, "color": 0, "is_bold": False, "is_italic": False, "baseline_offset": (word_bbox[3]-word_bbox[1])*0.78, "baseline_y": word_bbox[1]+(word_bbox[3]-word_bbox[1])*0.78}

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

    # Stage 1
    if image_replacements:
        for rect, word_text, metrics in image_replacements:
            page.add_redact_annot(rect, fill=(1, 1, 1))
        page.apply_redactions()
        for rect, word_text, metrics in image_replacements:
            img_bytes = renderer.render_word_image(
                word_text=word_text, bbox_width=rect.width, bbox_height=rect.height,
                font_name=metrics["font_name"], font_size=metrics["font_size"],
                is_bold=metrics["is_bold"], is_italic=metrics["is_italic"],
                baseline_offset=metrics["baseline_offset"], text_color=metrics["color"],
                custom_font_file=metrics["font_file"]
            )
            page.insert_image(rect, stream=img_bytes)

    # Stage 2
    if homo_substitutions:
        for rect, word_text, homo_word, metrics in homo_substitutions:
            page.add_redact_annot(rect, fill=(1, 1, 1))
        page.apply_redactions()
        for rect, word_text, homo_word, metrics in homo_substitutions:
            rgb = renderer.int_color_to_rgb(metrics["color"])
            color_tuple = (rgb[0]/255.0, rgb[1]/255.0, rgb[2]/255.0)
            baseline_pt = fitz.Point(rect.x0, metrics["baseline_y"])
            font_key = f"homo_font_{page_idx}_{metrics['is_bold']}_{metrics['is_italic']}"
            page.insert_text(baseline_pt, homo_word, fontsize=metrics["font_size"], fontname=font_key, fontfile=metrics["font_file"], color=color_tuple)

    # Stage 3
    if zw_injections:
        for rect, word_text, zw_word, metrics in zw_injections:
            page.add_redact_annot(rect, fill=(1, 1, 1))
        page.apply_redactions()
        for rect, word_text, zw_word, metrics in zw_injections:
            rgb = renderer.int_color_to_rgb(metrics["color"])
            color_tuple = (rgb[0]/255.0, rgb[1]/255.0, rgb[2]/255.0)
            baseline_pt = fitz.Point(rect.x0, metrics["baseline_y"])
            font_key = f"zw_font_{page_idx}_{metrics['is_bold']}_{metrics['is_italic']}"
            page.insert_text(baseline_pt, word_text, fontsize=metrics["font_size"], fontname=font_key, fontfile=metrics["font_file"], color=color_tuple)
            page.insert_text(baseline_pt, zw_word, fontsize=metrics["font_size"], fontname=font_key, fontfile=metrics["font_file"], render_mode=3)

    # Stage 4
    if layout_disruptions:
        for rect, word_text, disrupted_word, metrics in layout_disruptions:
            page.add_redact_annot(rect, fill=(1, 1, 1))
        page.apply_redactions()
        for rect, word_text, disrupted_word, metrics in layout_disruptions:
            rgb = renderer.int_color_to_rgb(metrics["color"])
            color_tuple = (rgb[0]/255.0, rgb[1]/255.0, rgb[2]/255.0)
            baseline_pt = fitz.Point(rect.x0, metrics["baseline_y"])
            font_key = f"disrupt_font_{page_idx}_{metrics['is_bold']}_{metrics['is_italic']}"
            page.insert_text(baseline_pt, word_text, fontsize=metrics["font_size"], fontname=font_key, fontfile=metrics["font_file"], color=color_tuple)
            page.insert_text(baseline_pt, disrupted_word, fontsize=metrics["font_size"], fontname=font_key, fontfile=metrics["font_file"], render_mode=3)

    if image_replacements or homo_substitutions or zw_injections or layout_disruptions:
        existing_links = page.get_links()
        existing_rects = [l.get("from") for l in existing_links if l.get("from")]
        for link in page_links:
            lrect = link.get("from")
            if lrect and lrect not in existing_rects:
                try: page.insert_link(link)
                except Exception: pass

    single_doc = fitz.open()
    single_doc.insert_pdf(doc, from_page=page_idx, to_page=page_idx)
    page_bytes = single_doc.tobytes(garbage=4, deflate=True)
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


def test_parallel_execution():
    doc = fitz.open("main.pdf")
    total_pages = len(doc)
    pages_dict = [page.get_text("dict") for page in doc]
    font_stats = TextClassifier.calculate_document_font_stats(pages_dict)
    doc.close()

    config = {
        "probability": 0.25,
        "homo_probability": 0.15,
        "zw_probability": 0.15,
        "disrupt_probability": 0.15,
        "stage": "all",
        "seed": 42,
        "min_word_len": 3,
        "dpi_scale": 3.0,
        "max_images_per_page": 0,
        "max_images_per_para": 0,
        "zw_count": 2,
        "disrupt_multiplier": 1.5
    }

    workers = 4
    print(f"--- Running Parallel Benchmark on main.pdf (112 pages) with {workers} worker processes ---")
    t0 = time.time()

    task_args = [(os.path.abspath("main.pdf"), p_idx, font_stats, config) for p_idx in range(total_pages)]

    results = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        for res in executor.map(_process_page_worker, task_args):
            results.append(res)

    t_proc = time.time() - t0
    print(f"Parallel Processing completed in {t_proc:.2f} seconds!")

    results.sort(key=lambda x: x[0])
    final_doc = fitz.open()
    for p_idx, p_bytes, p_stats in results:
        p_doc = fitz.open("pdf", p_bytes)
        final_doc.insert_pdf(p_doc)
        p_doc.close()

    out_file = "tmp_parallel_main.pdf"
    final_doc.save(out_file, garbage=4, deflate=True)
    final_doc.close()

    t_total = time.time() - t0
    print(f"Merged output saved to '{out_file}' in {t_total:.2f} seconds total!")
    
    if os.path.exists(out_file):
        os.remove(out_file)

if __name__ == "__main__":
    test_parallel_execution()
