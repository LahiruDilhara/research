import os
import sys
import re
import docx
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import qn, nsdecls
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL

def calculate_proportional_column_widths(tbl, total_width_dxa):
    """Calculates proportional column widths based on content length to fit page width."""
    ncols = len(tbl.columns)
    if ncols == 0:
        return []
        
    lengths = []
    for col in tbl.columns:
        max_len = 0
        for cell in col.cells:
            lines = cell.text.strip().split('\n')
            cell_len = max([len(l) for l in lines] if lines else [0])
            if cell_len > max_len:
                max_len = cell_len
        lengths.append(max(max_len, 3))
        
    total_len = sum(lengths)
    if total_len == 0:
        base_w = total_width_dxa // ncols
        widths = [base_w] * ncols
        widths[-1] += total_width_dxa - sum(widths)
        return widths
        
    min_col_w = min(1000, total_width_dxa // (ncols * 2))
    raw_widths = []
    for l in lengths:
        w = int(total_width_dxa * (l / total_len))
        raw_widths.append(max(w, min_col_w))
        
    curr_sum = sum(raw_widths)
    scale = total_width_dxa / curr_sum
    widths = [int(w * scale) for w in raw_widths]
    diff = total_width_dxa - sum(widths)
    widths[-1] += diff
    return widths

def load_latex_toc_entries():
    """Loads and parses TOC entries from LaTeX out/main.toc if available."""
    candidates = ['out/main.toc', 'converter/build_docx/main.toc', 'main.toc']
    toc_entries = []
    for c in candidates:
        if os.path.exists(c):
            with open(c, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    line = line.strip()
                    m = re.match(r'\\contentsline\s*\{([^}]+)\}\{(?:\\numberline\s*\{([^}]+)\})?([^}]+)\}\{([^}]+)\}', line)
                    if m:
                        level_str, num, title, page = m.groups()
                        clean_title = title.strip()
                        clean_title = re.sub(r'\\(?:textbf|textit|emph|textsc|math\w+)\{([^}]+)\}', r'\1', clean_title)
                        clean_title = re.sub(r'\$[^$]*\$', '', clean_title)
                        clean_title = clean_title.replace('~', ' ').replace('\\&', '&')
                        clean_title = re.sub(r'\s+', ' ', clean_title).strip()
                        toc_entries.append((level_str, num, clean_title, page))
            if toc_entries:
                break
    return toc_entries

WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def load_latex_list_entries(filename):
    """Parses LaTeX .lof or .lot file into structured list of (num, title, page)."""
    candidates = [
        os.path.join(WORKSPACE_ROOT, 'out', filename),
        os.path.join(WORKSPACE_ROOT, 'output', filename),
        os.path.join(WORKSPACE_ROOT, filename),
        os.path.join('out', filename),
        os.path.join('output', filename),
        filename
    ]
    entries = []
    for c in candidates:
        if os.path.exists(c):
            with open(c, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    line = line.strip()
                    m = re.search(r"\\numberline\s*\{([^}]+)\}\s*\{\\ignorespaces\s*(.+?)\}\}\s*\{([^}]+)\}", line)
                    if m:
                        num, title, page = m.groups()
                        clean_title = title.strip()
                        clean_title = re.sub(r'\\(?:textbf|textit|emph|textsc|math\w+|mathbf|text)\s*\{([^}]*)\}', r'\1', clean_title)
                        clean_title = re.sub(r'\$([^$]*)\$', r'\1', clean_title)
                        clean_title = clean_title.replace('~', ' ').replace('\\&', '&').replace('\\', '')
                        clean_title = clean_title.replace('{', '').replace('}', '')
                        clean_title = re.sub(r'\s+', ' ', clean_title).strip()
                        entries.append((num.strip(), clean_title, page.strip()))
            if entries:
                break
    return entries

def find_toc_page_number(heading_text, toc_entries):
    """Finds exact page number for a given heading from LaTeX TOC entries."""
    if not toc_entries:
        return '1'
    t_clean = re.sub(r'[^\w\s]', '', heading_text.lower())
    t_clean = re.sub(r'\s+', ' ', t_clean).strip()
    
    for lvl, num, title, page in toc_entries:
        full = f"{num} {title}" if num else title
        f_clean = re.sub(r'[^\w\s]', '', full.lower())
        f_clean = re.sub(r'\s+', ' ', f_clean).strip()
        if t_clean == f_clean or (len(t_clean) > 8 and t_clean in f_clean) or (len(f_clean) > 8 and f_clean in t_clean):
            return page
            
    t_app = re.sub(r'^7\b', 'A', heading_text)
    t_app = re.sub(r'^8\b', 'B', t_app)
    t_app = re.sub(r'^9\b', 'C', t_app)
    t_app = re.sub(r'^10\b', 'D', t_app)
    t_app = re.sub(r'^11\b', 'E', t_app)
    t_clean_app = re.sub(r'[^\w\s]', '', t_app.lower())
    t_clean_app = re.sub(r'\s+', ' ', t_clean_app).strip()
    
    for lvl, num, title, page in toc_entries:
        full = f"{num} {title}" if num else title
        f_clean = re.sub(r'[^\w\s]', '', full.lower())
        f_clean = re.sub(r'\s+', ' ', f_clean).strip()
        if t_clean_app == f_clean or (len(t_clean_app) > 8 and t_clean_app in f_clean) or (len(f_clean) > 8 and f_clean in t_clean_app):
            return page

    t_words = set(t_clean.split())
    best_match = '1'
    best_score = 0
    for lvl, num, title, page in toc_entries:
        full = f"{num} {title}" if num else title
        f_clean = re.sub(r'[^\w\s]', '', full.lower())
        f_words = set(f_clean.split())
        overlap = len(t_words & f_words)
        if overlap > best_score and overlap >= 2:
            best_score = overlap
            best_match = page
    return best_match

def populate_exact_docx_page_numbers(docx_path):
    """
    Renders docx via LibreOffice to evaluate dynamic PAGEREF layout and 
    pre-populates all cached <w:t> elements inside PAGEREF fields with the exact 
    DOCX layout page numbers.
    """
    import subprocess, tempfile, shutil
    try:
        import pypdf
    except ImportError:
        return
        
    libreoffice_bin = shutil.which('libreoffice') or shutil.which('soffice')
    if not libreoffice_bin:
        return
        
    print('Calculating exact DOCX pagination via headless layout evaluation...')
    with tempfile.TemporaryDirectory() as tmpdir:
        pdf_name = os.path.splitext(os.path.basename(docx_path))[0] + '.pdf'
        res = subprocess.run([libreoffice_bin, '--headless', '--convert-to', 'pdf', docx_path, '--outdir', tmpdir],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if res.returncode != 0:
            return
            
        pdf_path = os.path.join(tmpdir, pdf_name)
        if not os.path.exists(pdf_path):
            return
            
        reader = pypdf.PdfReader(pdf_path)
        all_pages_text = [re.sub(r'\s+', ' ', p.extract_text() or '') for p in reader.pages]
        all_pages_clean = [re.sub(r'\s+', ' ', re.sub(r'[^\w\s]', ' ', p.extract_text().lower() if p.extract_text() else '')) for p in reader.pages]
        
        roman_map = {1: 'i', 2: 'ii', 3: 'iii', 4: 'iv', 5: 'v', 6: 'vi', 7: 'vii', 8: 'viii', 9: 'ix', 10: 'x', 11: 'xi', 12: 'xii', 13: 'xiii', 14: 'xiv', 15: 'xv', 16: 'xvi'}
        
        # Detect start of Arabic section (Chapter 1 Introduction)
        sec2_start_idx = None
        for idx, text in enumerate(all_pages_text):
            if re.search(r'1\s+Introduction', text, re.IGNORECASE) and 'Table of Contents' not in text and 'List of' not in text:
                sec2_start_idx = idx
                break
                    
        if sec2_start_idx is None:
            sec2_start_idx = 15
            
        bm_to_page = {
            '_Toc_Declaration': 'ii',
            '_Toc_Acknowledgement': 'iii',
            '_Toc_Abstract': 'iv',
            '_Toc_Table_Of_Contents': 'v',
        }
        
        for idx in range(sec2_start_idx):
            lines = [l.strip() for l in (reader.pages[idx].extract_text() or '').split('\n') if l.strip()]
            top_text = ' '.join(lines[:3]) if len(lines) >= 3 else ' '.join(lines)
            r_num = roman_map.get(idx + 1, str(idx + 1))
            if re.search(r'\bList of Figures\b', top_text, re.I) and '_Toc_List_Figures' not in bm_to_page:
                bm_to_page['_Toc_List_Figures'] = r_num
            if re.search(r'\bList of Tables\b', top_text, re.I) and '_Toc_List_Tables' not in bm_to_page:
                bm_to_page['_Toc_List_Tables'] = r_num
            if re.search(r'\bList of Abbreviations\b', top_text, re.I) and '_Toc_List_Abbreviations' not in bm_to_page:
                bm_to_page['_Toc_List_Abbreviations'] = r_num
                bm_to_page['_Toc_List_of_Abbreviations'] = r_num

        # Load document to match all Bookmarks (Headings, Figures, Tables)
        doc = docx.Document(docx_path)
        
        # Map bookmark name to its paragraph text
        bm_to_p_text = {}
        for p in doc.paragraphs:
            for bm in p._p.iter():
                if bm.tag.endswith('bookmarkStart'):
                    b_name = bm.get(qn('w:name'))
                    if b_name and (b_name.startswith('_Toc_') or b_name.startswith('_Fig_') or b_name.startswith('_Tbl_')):
                        bm_to_p_text[b_name] = p.text.strip()

        # Pre-extract lines per page
        all_pages_lines = [[re.sub(r'\s+', ' ', re.sub(r'[^\w\s]', ' ', l.lower())).strip() for l in (p.extract_text() or '').split('\n') if l.strip()] for p in reader.pages]

        # Match all bookmarks in body (sec2_start_idx to end of document)
        for bm_name, p_text in bm_to_p_text.items():
            if bm_name in bm_to_page:
                continue
                
            clean_p = re.sub(r'\s+', ' ', re.sub(r'[^\w\s]', ' ', p_text.lower())).strip()
            if not clean_p:
                continue
                
            found_arabic_page = None
            
            # Pass 1: Exact standalone line match (crucial for standalone headings like "References", "1 Introduction")
            for idx in range(sec2_start_idx, len(reader.pages)):
                if clean_p in all_pages_lines[idx]:
                    found_arabic_page = str(idx - sec2_start_idx + 1)
                    break
                    
            # Pass 2: Match line prefix (at least 15 characters)
            if found_arabic_page is None:
                snip15 = clean_p[:min(len(clean_p), 18)]
                for idx in range(sec2_start_idx, len(reader.pages)):
                    if any(l.startswith(snip15) or (len(l) >= 15 and snip15 in l) for l in all_pages_lines[idx]):
                        found_arabic_page = str(idx - sec2_start_idx + 1)
                        break
                        
            # Pass 3: Match 28-character prefix snippet across page text
            if found_arabic_page is None:
                snippet = clean_p[:min(len(clean_p), 28)].strip()
                for idx in range(sec2_start_idx, len(reader.pages)):
                    if snippet in all_pages_clean[idx]:
                        found_arabic_page = str(idx - sec2_start_idx + 1)
                        break
                        
            # Pass 4: Match first 4 words
            if found_arabic_page is None:
                words = snippet.split()[:4]
                short_snip = ' '.join(words)
                for idx in range(sec2_start_idx, len(reader.pages)):
                    if short_snip and short_snip in all_pages_clean[idx]:
                        found_arabic_page = str(idx - sec2_start_idx + 1)
                        break
                        
            # Pass 5: Match identifier prefix (e.g. "Figure 4.2", "Table 5.7", "4.7.1")
            if found_arabic_page is None:
                id_match = re.match(r'^(figure\s+\w+\s+\w+|table\s+\w+\s+\w+|\d+\s+\d+(?:\s+\d+)?)', clean_p)
                if id_match:
                    id_snip = id_match.group(1)
                    for idx in range(sec2_start_idx, len(reader.pages)):
                        if id_snip in all_pages_clean[idx]:
                            found_arabic_page = str(idx - sec2_start_idx + 1)
                            break
                            
            if found_arabic_page is not None:
                bm_to_page[bm_name] = found_arabic_page

        # Update all PAGEREF w:t cached values in the document
        updated_count = 0
        for p in doc.paragraphs:
            for fld in p._p.iter():
                if fld.tag.endswith('fldSimple'):
                    instr = fld.get(qn('w:instr')) or ''
                    if 'PAGEREF' in instr:
                        tokens = instr.split()
                        try:
                            p_idx = tokens.index('PAGEREF')
                            bm_target = tokens[p_idx + 1]
                            if bm_target in bm_to_page:
                                exact_pg = bm_to_page[bm_target]
                                for t in fld.iter():
                                    if t.tag.endswith('}t') or t.tag == 'w:t':
                                        t.text = str(exact_pg)
                                        updated_count += 1
                        except (ValueError, IndexError):
                            pass

        doc.save(docx_path)
        print(f'Successfully updated {updated_count} dynamic PAGEREF cached page numbers to exact DOCX layout values!')

def style_thesis_docx(docx_path):
    print(f'Loading {docx_path}...')
    doc = docx.Document(docx_path)
    
    # 1. Page Margins & A4 Dimensions (NSBM Guidelines: Left 1.25 in, Others 1.0 in)
    for s in doc.sections:
        s.top_margin = Inches(1.0)
        s.bottom_margin = Inches(1.0)
        s.left_margin = Inches(1.25)
        s.right_margin = Inches(1.0)
        s.page_width = Inches(8.27)
        s.page_height = Inches(11.69)
        
    TOTAL_WIDTH = 8666  # 6.02 in printable area in dxa (11906 - 1800 - 1440)
    
    # Ensure Bibliography paragraphs and their bookmarks are positioned immediately under the References Heading (before Appendices)
    ref_p = None
    for p in doc.paragraphs:
        st = p.style.name if p.style else ''
        if 'Heading 1' in st and p.text.strip() == 'References':
            ref_p = p
            break
            
    if ref_p is not None:
        body_elem = doc._body._element
        children = list(body_elem)
        ref_bm_map = {} # p_element -> list of (bm_id, bm_name)
        to_remove = []

        for i, child in enumerate(children):
            if child.tag.endswith('bookmarkStart'):
                name = child.get(qn('w:name'))
                if name and name.startswith('ref-'):
                    bm_id = child.get(qn('w:id'))
                    to_remove.append(child)
                    for j in range(i+1, min(len(children), i+4)):
                        if children[j].tag.endswith('p'):
                            ref_bm_map[children[j]] = (bm_id, name)
                            break
            elif child.tag.endswith('bookmarkEnd'):
                bm_id = child.get(qn('w:id'))
                if any(b_id == bm_id for b_id, _ in ref_bm_map.values()):
                    to_remove.append(child)

        # Remove old orphan bookmark elements from body
        for elem in to_remove:
            if elem.getparent() is not None:
                body_elem.remove(elem)

        # Attach bookmarks directly inside each mapped bibliography paragraph
        for p_elem, (bm_id, bm_name) in ref_bm_map.items():
            bm_start = OxmlElement('w:bookmarkStart')
            bm_start.set(qn('w:id'), bm_id)
            bm_start.set(qn('w:name'), bm_name)
            bm_end = OxmlElement('w:bookmarkEnd')
            bm_end.set(qn('w:id'), bm_id)
            p_elem.insert(0, bm_start)
            p_elem.append(bm_end)

        bib_paragraphs = []
        for p in doc.paragraphs:
            st = p.style.name if p.style else ''
            if st == 'Bibliography' or (p.text.strip().startswith('[') and ']' in p.text.strip()[:6]):
                bib_paragraphs.append(p)
                
        if bib_paragraphs:
            parent = ref_p._p.getparent()
            ref_idx = parent.index(ref_p._p)
            for i, bp in enumerate(bib_paragraphs):
                parent.remove(bp._p)
                parent.insert(ref_idx + 1 + i, bp._p)
            print(f'Repositioned {len(bib_paragraphs)} bibliography citations (with {len(ref_bm_map)} bookmarks) immediately under References heading (before Appendix A).')
    
    # 2. Add Bookmarks to all Headings for Hyperlinked Table of Contents
    toc_entries = load_latex_toc_entries()
    headings_data = []
    bookmark_id_counter = 500
    
    preliminary_toc = [
        (1, 'Declaration', '_Toc_Declaration', 'ii'),
        (1, 'Acknowledgement', '_Toc_Acknowledgement', 'iii'),
        (1, 'Abstract', '_Toc_Abstract', 'iv'),
        (1, 'Table of Contents', '_Toc_Table_Of_Contents', 'v'),
        (1, 'List of Figures', '_Toc_List_Figures', find_toc_page_number('List of Figures', toc_entries)),
        (1, 'List of Tables', '_Toc_List_Tables', find_toc_page_number('List of Tables', toc_entries)),
        (1, 'List of Abbreviations', '_Toc_List_of_Abbreviations', find_toc_page_number('List of Abbreviations', toc_entries)),
    ]
    
    for p in doc.paragraphs:
        st = p.style.name if p.style else ''
        t = p.text.strip()
        if not t:
            continue
            
        lvl = None
        if 'Heading 1' in st:
            lvl = 1
        elif 'Heading 2' in st:
            lvl = 2
        elif 'Heading 3' in st:
            lvl = 3
            
        if lvl is not None:
            if t in ['Declaration', 'Acknowledgement', 'Abstract', 'List of Abbreviations']:
                bm_name = f"_Toc_{re.sub(r'[^A-Za-z0-9]', '_', t)}"
                bm_start = OxmlElement('w:bookmarkStart')
                bm_start.set(qn('w:id'), str(bookmark_id_counter))
                bm_start.set(qn('w:name'), bm_name)
                bm_end = OxmlElement('w:bookmarkEnd')
                bm_end.set(qn('w:id'), str(bookmark_id_counter))
                bookmark_id_counter += 1
                p._p.insert(0, bm_start)
                p._p.append(bm_end)
                continue
            elif t.startswith('Table of Contents') or t.startswith('Contents') or t.startswith('List of Figures') or t.startswith('List of Tables'):
                continue
                
            bm_id = str(bookmark_id_counter)
            bm_name = f"_Toc_Heading_{bookmark_id_counter}"
            bookmark_id_counter += 1
            
            bm_start = OxmlElement('w:bookmarkStart')
            bm_start.set(qn('w:id'), bm_id)
            bm_start.set(qn('w:name'), bm_name)
            
            bm_end = OxmlElement('w:bookmarkEnd')
            bm_end.set(qn('w:id'), bm_id)
            
            p._p.insert(0, bm_start)
            p._p.append(bm_end)
            
            page_num = find_toc_page_number(t, toc_entries)
            headings_data.append((lvl, t, bm_name, page_num))
            
    headings_data = preliminary_toc + headings_data

    # Configure base Word Document Styles according to NSBM Guidelines
    styles_to_config = {
        'Normal': {'size': Pt(12), 'bold': False, 'font': 'Times New Roman'},
        'Body Text': {'size': Pt(12), 'bold': False, 'font': 'Times New Roman'},
        'First Paragraph': {'size': Pt(12), 'bold': False, 'font': 'Times New Roman'},
        'Heading 1': {'size': Pt(12), 'bold': True, 'font': 'Times New Roman'},
        'Heading 2': {'size': Pt(12), 'bold': True, 'font': 'Times New Roman'},
        'Heading 3': {'size': Pt(12), 'bold': False, 'font': 'Times New Roman'},
        'Heading 4': {'size': Pt(12), 'bold': False, 'font': 'Times New Roman'},
        'Caption': {'size': Pt(12), 'bold': False, 'font': 'Times New Roman'},
        'Table Caption': {'size': Pt(12), 'bold': False, 'font': 'Times New Roman'},
        'Image Caption': {'size': Pt(12), 'bold': False, 'font': 'Times New Roman'},
        'Captioned Figure': {'size': Pt(12), 'bold': False, 'font': 'Times New Roman'},
    }
    for s_name, cfg in styles_to_config.items():
        try:
            s = doc.styles[s_name]
            s.font.name = cfg['font']
            s.font.size = cfg['size']
            s.font.bold = cfg['bold']
            s.font.color.rgb = RGBColor(0x00, 0x00, 0x00)
            if s_name in ['Normal', 'Body Text', 'First Paragraph', 'Block Text', 'Bibliography']:
                s.paragraph_format.line_spacing = 1.5
            elif s_name in ['Caption', 'Table Caption', 'Image Caption', 'Captioned Figure']:
                s.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
        except Exception:
            pass

    # 3. Format Title Page & Insert Explicit Physical Page Breaks for Every Topic/Chapter
    cover_page_mode = True
    seen_first_h1 = False
    current_chapter = '0'
    fig_counter = {}
    tbl_counter = {}
    
    # We iterate over a snapshot of paragraphs
    for p in list(doc.paragraphs):
        st = p.style.name if p.style else ''
        t = p.text.strip()
        
        # Remove any stray duplicate TOC title
        if t == 'Table of Contents' or t == 'Contents':
            p._p.getparent().remove(p._p)
            continue
            
        # Prune empty artifact paragraphs from Pandoc in body matter
        if not cover_page_mode and not t:
            has_media = any(e.tag.endswith('drawing') or e.tag.endswith('shape') or e.tag.endswith('graphic') or e.tag.endswith('pict') for e in p._p.iter())
            has_br = any(e.tag.endswith('br') or e.tag.endswith('pageBreakBefore') for e in p._p.iter())
            if not has_media and not has_br:
                p._p.getparent().remove(p._p)
                continue
            
        if cover_page_mode:
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            
            if 'CUSTOMIZABLE PAPER-BASED VIRTUAL' in t.upper():
                p.paragraph_format.space_before = Pt(0)
                p.paragraph_format.space_after = Pt(50)
                p.paragraph_format.line_spacing = 1.3
                for r in p.runs:
                    r.font.name = 'Times New Roman'
                    r.font.size = Pt(16)
                    r.font.bold = True
                    r.font.color.rgb = RGBColor(0x00, 0x00, 0x00)
            elif 'A thesis submitted to NSBM' in t:
                p.paragraph_format.space_before = Pt(0)
                p.paragraph_format.space_after = Pt(2)
                p.paragraph_format.line_spacing = 1.15
                for r in p.runs:
                    r.font.name = 'Times New Roman'
                    r.font.size = Pt(14)
                    r.font.bold = False
                    r.font.color.rgb = RGBColor(0x00, 0x00, 0x00)
            elif 'Bachelor of Science' in t:
                p.paragraph_format.space_before = Pt(0)
                p.paragraph_format.space_after = Pt(50)
                p.paragraph_format.line_spacing = 1.15
                for r in p.runs:
                    r.font.name = 'Times New Roman'
                    r.font.size = Pt(14)
                    r.font.bold = False
                    r.font.color.rgb = RGBColor(0x00, 0x00, 0x00)
            elif 'By' == t:
                p.paragraph_format.space_before = Pt(0)
                p.paragraph_format.space_after = Pt(50)
                p.paragraph_format.line_spacing = 1.15
                for r in p.runs:
                    r.font.name = 'Times New Roman'
                    r.font.size = Pt(14)
                    r.font.bold = False
                    r.font.color.rgb = RGBColor(0x00, 0x00, 0x00)
            elif 'GANEPOLA ARACHCHIGE LAHIRU DILHARA' in t.upper() or 'LAHIRU DILHARA' in t.upper():
                p.paragraph_format.space_before = Pt(0)
                p.paragraph_format.space_after = Pt(55)
                p.paragraph_format.line_spacing = 1.15
                for r in p.runs:
                    r.font.name = 'Times New Roman'
                    r.font.size = Pt(14)
                    r.font.bold = True
                    r.font.color.rgb = RGBColor(0x00, 0x00, 0x00)
            elif any(inst in t for inst in ['Department of Computer Science', 'Faculty of Computing', 'NSBM Green University']):
                p.paragraph_format.space_before = Pt(0)
                p.paragraph_format.space_after = Pt(2)
                p.paragraph_format.line_spacing = 1.15
                for r in p.runs:
                    r.font.name = 'Times New Roman'
                    r.font.size = Pt(14)
                    r.font.bold = False
                    r.font.color.rgb = RGBColor(0x00, 0x00, 0x00)
            elif 'Sri Lanka' in t:
                p.paragraph_format.space_before = Pt(0)
                p.paragraph_format.space_after = Pt(50)
                p.paragraph_format.line_spacing = 1.15
                for r in p.runs:
                    r.font.name = 'Times New Roman'
                    r.font.size = Pt(14)
                    r.font.bold = False
                    r.font.color.rgb = RGBColor(0x00, 0x00, 0x00)
            elif 'September 2026' in t:
                p.paragraph_format.space_before = Pt(0)
                p.paragraph_format.space_after = Pt(0)
                p.paragraph_format.line_spacing = 1.15
                for r in p.runs:
                    r.font.name = 'Times New Roman'
                    r.font.size = Pt(14)
                    r.font.bold = False
                    r.font.color.rgb = RGBColor(0x00, 0x00, 0x00)
                # Explicit Pagebreak after cover page
                r_pb = p.add_run()
                r_pb.add_break(WD_BREAK.PAGE)
                cover_page_mode = False
            else:
                p.paragraph_format.space_after = Pt(6)
                for r in p.runs:
                    r.font.name = 'Times New Roman'
                    r.font.size = Pt(14)
                    r.font.bold = False
                    r.font.color.rgb = RGBColor(0x00, 0x00, 0x00)
                    
        else:
            # Academic Headings & Body Styling (Times New Roman, Pure Black)
            if 'Heading 1' in st:
                # Track chapter
                if t.startswith('1\t') or t.startswith('1 '): current_chapter = '1'
                elif t.startswith('2\t') or t.startswith('2 '): current_chapter = '2'
                elif t.startswith('3\t') or t.startswith('3 '): current_chapter = '3'
                elif t.startswith('4\t') or t.startswith('4 '): current_chapter = '4'
                elif t.startswith('5\t') or t.startswith('5 '): current_chapter = '5'
                elif t.startswith('6\t') or t.startswith('6 '): current_chapter = '6'
                elif 'Appendix A' in t or t.startswith('7\t'): current_chapter = 'A'
                elif 'Appendix B' in t or t.startswith('8\t'): current_chapter = 'B'
                elif 'Appendix C' in t or t.startswith('9\t'): current_chapter = 'C'
                elif 'Appendix D' in t or t.startswith('10\t'): current_chapter = 'D'
                elif 'Appendix E' in t or t.startswith('11\t'): current_chapter = 'E'
                elif 'References' in t: current_chapter = 'Ref'

                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                p.paragraph_format.space_before = Pt(18)
                p.paragraph_format.space_after = Pt(10)
                p.paragraph_format.keep_with_next = True
                
                # Insert physical explicit page break before each Chapter and Preliminary Section (except the very first one right after title page)
                if seen_first_h1:
                    p_break = p.insert_paragraph_before()
                    p_break.paragraph_format.space_before = Pt(0)
                    p_break.paragraph_format.space_after = Pt(0)
                    r_br = p_break.add_run()
                    r_br.add_break(WD_BREAK.PAGE)
                else:
                    seen_first_h1 = True
                    
                for r in p.runs:
                    r.font.name = 'Times New Roman'
                    r.font.size = Pt(12)
                    r.font.bold = True
                    r.font.color.rgb = RGBColor(0x00, 0x00, 0x00)
                    
            elif 'Heading 2' in st:
                p.paragraph_format.space_before = Pt(14)
                p.paragraph_format.space_after = Pt(6)
                p.paragraph_format.keep_with_next = True
                for r in p.runs:
                    r.font.name = 'Times New Roman'
                    r.font.size = Pt(12)
                    r.font.bold = True
                    r.font.color.rgb = RGBColor(0x00, 0x00, 0x00)
                    
            elif 'Heading 3' in st:
                p.paragraph_format.space_before = Pt(12)
                p.paragraph_format.space_after = Pt(4)
                p.paragraph_format.keep_with_next = True
                for r in p.runs:
                    r.font.name = 'Times New Roman'
                    r.font.size = Pt(12)
                    r.font.bold = False  # NSBM Guideline: 1st Numeral with 2 decimals is Simple (Not Bold)
                    r.font.color.rgb = RGBColor(0x00, 0x00, 0x00)
                    
            elif 'Heading 4' in st:
                p.paragraph_format.space_before = Pt(10)
                p.paragraph_format.space_after = Pt(4)
                p.paragraph_format.keep_with_next = True
                for r in p.runs:
                    r.font.name = 'Times New Roman'
                    r.font.size = Pt(12)
                    r.font.bold = False  # NSBM Guideline: 1st Numeral with 3 decimals is Simple (Not Bold)
                    r.font.color.rgb = RGBColor(0x00, 0x00, 0x00)
                    
            elif 'Table Caption' in st or (t.startswith('Table ') and len(t) < 200 and not t.startswith('Table of Contents')) or ('Table' in st and 'Caption' in st):
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                p.paragraph_format.space_before = Pt(12)
                p.paragraph_format.space_after = Pt(4)
                p.paragraph_format.keep_with_next = True
                
                c_num = tbl_counter.get(current_chapter, 0) + 1
                tbl_counter[current_chapter] = c_num
                label_prefix = f"Table {current_chapter}.{c_num}: "
                
                clean_title = re.sub(r"^Table\s+\w+\.\w+:\s*", "", t)
                p.text = ""
                
                # Add bookmark to table caption
                tbl_key = f"{current_chapter}_{c_num}"
                bm_tbl_name = f"_Tbl_{tbl_key}"
                bm_id = str(bookmark_id_counter)
                bookmark_id_counter += 1
                bm_start = OxmlElement('w:bookmarkStart')
                bm_start.set(qn('w:id'), bm_id)
                bm_start.set(qn('w:name'), bm_tbl_name)
                bm_end = OxmlElement('w:bookmarkEnd')
                bm_end.set(qn('w:id'), bm_id)
                p._p.append(bm_start)
                
                r_lbl = p.add_run(label_prefix)
                r_lbl.font.name = 'Times New Roman'
                r_lbl.font.size = Pt(12)
                r_lbl.font.bold = True
                r_lbl.font.color.rgb = RGBColor(0x00, 0x00, 0x00)
                
                r_txt = p.add_run(clean_title)
                r_txt.font.name = 'Times New Roman'
                r_txt.font.size = Pt(12)
                r_txt.font.bold = False
                r_txt.font.color.rgb = RGBColor(0x00, 0x00, 0x00)
                
                p._p.append(bm_end)
                
            elif 'Image Caption' in st or (st == 'Caption' and not t.startswith('Table ')) or (t.startswith('Figure ') and len(t) < 200):
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                p.paragraph_format.space_before = Pt(4)
                p.paragraph_format.space_after = Pt(12)
                
                c_num = fig_counter.get(current_chapter, 0) + 1
                fig_counter[current_chapter] = c_num
                label_prefix = f"Figure {current_chapter}.{c_num}: "
                
                clean_title = re.sub(r"^Figure\s+\w+\.\w+:\s*", "", t)
                p.text = ""
                
                # Add bookmark to figure caption
                fig_key = f"{current_chapter}_{c_num}"
                bm_fig_name = f"_Fig_{fig_key}"
                bm_id = str(bookmark_id_counter)
                bookmark_id_counter += 1
                bm_start = OxmlElement('w:bookmarkStart')
                bm_start.set(qn('w:id'), bm_id)
                bm_start.set(qn('w:name'), bm_fig_name)
                bm_end = OxmlElement('w:bookmarkEnd')
                bm_end.set(qn('w:id'), bm_id)
                p._p.append(bm_start)
                
                r_lbl = p.add_run(label_prefix)
                r_lbl.font.name = 'Times New Roman'
                r_lbl.font.size = Pt(12)
                r_lbl.font.bold = True
                r_lbl.font.color.rgb = RGBColor(0x00, 0x00, 0x00)
                
                r_txt = p.add_run(clean_title)
                r_txt.font.name = 'Times New Roman'
                r_txt.font.size = Pt(12)
                r_txt.font.bold = False
                r_txt.font.color.rgb = RGBColor(0x00, 0x00, 0x00)
                
                p._p.append(bm_end)
                    
            elif 'Captioned Figure' in st or any(e.tag.endswith('drawing') or e.tag.endswith('shape') or e.tag.endswith('graphic') or e.tag.endswith('pict') for e in p._p.iter()):
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                p.paragraph_format.space_before = Pt(6)
                p.paragraph_format.space_after = Pt(4)
                p.paragraph_format.keep_with_next = True
            else:
                if 'I declare that the content' in t:
                    p.paragraph_format.space_before = Pt(0)
                    p.paragraph_format.space_after = Pt(80)
                    p.paragraph_format.line_spacing = 1.5
                elif t.startswith('Signature of the Supervisors'):
                    p.paragraph_format.space_before = Pt(110)
                    p.paragraph_format.space_after = Pt(14)
                    p.paragraph_format.line_spacing = 1.5
                elif t.startswith('...') or t.startswith('…') or t.startswith('···') or t.startswith('....'):
                    p.text = '........................................................'
                    p.paragraph_format.space_before = Pt(12)
                    p.paragraph_format.space_after = Pt(10)
                    p.paragraph_format.line_spacing = 1.0
                elif t == 'Prof. Chaminda Wijesinghe':
                    p.paragraph_format.space_before = Pt(6)
                    p.paragraph_format.space_after = Pt(2)
                    p.paragraph_format.line_spacing = 1.15
                elif t in ['Principal Supervisor', 'Department of Computer Science,', 'NSBM Green University'] and current_chapter == '0':
                    p.paragraph_format.space_before = Pt(0)
                    p.paragraph_format.space_after = Pt(2)
                    p.paragraph_format.line_spacing = 1.15
                else:
                    p.paragraph_format.space_before = Pt(0)
                    p.paragraph_format.space_after = Pt(2)
                    p.paragraph_format.line_spacing = 1.5  # NSBM Guideline: 1.5-line spacing throughout
                    
                for r in p.runs:
                    if not r.font.name:
                        r.font.name = 'Times New Roman'
                    if not r.font.size:
                        r.font.size = Pt(12)  # NSBM Guideline: Body font 12 pt
                    r.font.color.rgb = RGBColor(0x00, 0x00, 0x00)

    # 4. Build Hyperlinked Table of Contents
    target_p = None
    for p in doc.paragraphs:
        if 'List of Abbreviations' in p.text:
            target_p = p
            break
            
    if target_p and headings_data:
        print(f'Building Hyperlinked Table of Contents ({len(headings_data)} entries)...')
        
        toc_title = target_p.insert_paragraph_before('Table of Contents')
        toc_title.alignment = WD_ALIGN_PARAGRAPH.CENTER
        toc_title.paragraph_format.space_before = Pt(18)
        toc_title.paragraph_format.space_after = Pt(12)
        for r in toc_title.runs:
            r.font.name = 'Times New Roman'
            r.font.size = Pt(12)
            r.font.bold = True
            r.font.color.rgb = RGBColor(0x00, 0x00, 0x00)
            
        bm_toc_start = OxmlElement('w:bookmarkStart')
        bm_toc_start.set(qn('w:id'), '97')
        bm_toc_start.set(qn('w:name'), '_Toc_Table_Of_Contents')
        bm_toc_end = OxmlElement('w:bookmarkEnd')
        bm_toc_end.set(qn('w:id'), '97')
        toc_title._p.insert(0, bm_toc_start)
        toc_title._p.append(bm_toc_end)
        
        for lvl, title_text, bm_name, page_num in headings_data:
            p_toc = target_p.insert_paragraph_before()
            
            old_pPr = p_toc._p.find(qn('w:pPr'))
            if old_pPr is not None:
                p_toc._p.remove(old_pPr)
                
            left_indent_dxa = 0 if lvl == 1 else (288 if lvl == 2 else 576)
            space_before_dpt = 80 if lvl == 1 else 20
            
            pPr_xml = f'''<w:pPr {nsdecls("w")}>
  <w:tabs>
    <w:tab w:val="right" w:leader="dot" w:pos="8666"/>
  </w:tabs>
  <w:spacing w:before="{space_before_dpt}" w:after="40" w:line="276" w:lineRule="auto"/>
  <w:ind w:left="{left_indent_dxa}"/>
</w:pPr>'''
            p_toc._p.insert(0, parse_xml(pPr_xml))
            
            sz_val = '24' if lvl == 1 else ('22' if lvl == 2 else '20')
            
            # 1. Heading title hyperlink
            hyperlink_title = OxmlElement('w:hyperlink')
            hyperlink_title.set(qn('w:anchor'), bm_name)
            hyperlink_title.set(qn('w:history'), '1')
            
            run_title = OxmlElement('w:r')
            rPr_title = OxmlElement('w:rPr')
            rFont_title = OxmlElement('w:rFonts')
            rFont_title.set(qn('w:ascii'), 'Times New Roman')
            rFont_title.set(qn('w:hAnsi'), 'Times New Roman')
            rPr_title.append(rFont_title)
            rPr_title.append(parse_xml(f'<w:color {nsdecls("w")} w:val="000000"/>'))
            if lvl == 1:
                rPr_title.append(OxmlElement('w:b'))
            rPr_title.append(parse_xml(f'<w:sz {nsdecls("w")} w:val="{sz_val}"/>'))
            run_title.append(rPr_title)
            
            text_elem = OxmlElement('w:t')
            text_elem.text = title_text
            run_title.append(text_elem)
            hyperlink_title.append(run_title)
            p_toc._p.append(hyperlink_title)
            
            # 2. Tab character run with dot leader (direct child of p)
            run_tab = OxmlElement('w:r')
            rPr_tab = OxmlElement('w:rPr')
            rFont_tab = OxmlElement('w:rFonts')
            rFont_tab.set(qn('w:ascii'), 'Times New Roman')
            rFont_tab.set(qn('w:hAnsi'), 'Times New Roman')
            rPr_tab.append(rFont_tab)
            rPr_tab.append(parse_xml(f'<w:color {nsdecls("w")} w:val="000000"/>'))
            if lvl == 1:
                rPr_tab.append(OxmlElement('w:b'))
            rPr_tab.append(parse_xml(f'<w:sz {nsdecls("w")} w:val="{sz_val}"/>'))
            run_tab.append(rPr_tab)
            run_tab.append(OxmlElement('w:tab'))
            p_toc._p.append(run_tab)
            
            # 3. Dynamic Page Number Hyperlink via PAGEREF field (direct child of p)
            is_prelim = bm_name in ['_Toc_Declaration', '_Toc_Acknowledgement', '_Toc_Abstract', '_Toc_Table_Of_Contents', '_Toc_List_Figures', '_Toc_List_Tables', '_Toc_List_Abbreviations']
            num_format = r'\* roman' if is_prelim else r'\* Arabic'
            b_tag = '<w:b/>' if lvl == 1 else ''
            fld_xml = f'''<w:fldSimple {nsdecls("w")} w:instr="PAGEREF {bm_name} \\h {num_format}">
  <w:r>
    <w:rPr>
      <w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman"/>
      <w:color w:val="000000"/>
      {b_tag}
      <w:sz w:val="{sz_val}"/>
    </w:rPr>
    <w:t>{page_num}</w:t>
  </w:r>
</w:fldSimple>'''
            p_toc._p.append(parse_xml(fld_xml))

        # Page break after TOC before List of Figures
        p_lof_break = target_p.insert_paragraph_before()
        p_lof_break.paragraph_format.space_before = Pt(0)
        p_lof_break.paragraph_format.space_after = Pt(0)
        p_lof_break.add_run().add_break(WD_BREAK.PAGE)

        # 4B. Build List of Figures (Appendix V: No dot leaders, right-aligned Page header)
        lof_entries = load_latex_list_entries('main.lof')
        if lof_entries:
            print(f'Building List of Figures ({len(lof_entries)} entries)...')
            lof_title = target_p.insert_paragraph_before('List of Figures')
            lof_title.alignment = WD_ALIGN_PARAGRAPH.CENTER
            lof_title.paragraph_format.space_before = Pt(18)
            lof_title.paragraph_format.space_after = Pt(12)
            for r in lof_title.runs:
                r.font.name = 'Times New Roman'
                r.font.size = Pt(12)
                r.font.bold = True
                r.font.color.rgb = RGBColor(0x00, 0x00, 0x00)

            bm_lof_start = OxmlElement('w:bookmarkStart')
            bm_lof_start.set(qn('w:id'), '98')
            bm_lof_start.set(qn('w:name'), '_Toc_List_Figures')
            bm_lof_end = OxmlElement('w:bookmarkEnd')
            bm_lof_end.set(qn('w:id'), '98')
            lof_title._p.insert(0, bm_lof_start)
            lof_title._p.append(bm_lof_end)

            # Right-aligned Page column header
            p_lof_header = target_p.insert_paragraph_before()
            old_pPr = p_lof_header._p.find(qn('w:pPr'))
            if old_pPr is not None:
                p_lof_header._p.remove(old_pPr)
            pPr_hdr_xml = f'''<w:pPr {nsdecls("w")}>
  <w:tabs>
    <w:tab w:val="right" w:leader="none" w:pos="8666"/>
  </w:tabs>
  <w:spacing w:before="60" w:after="80" w:line="276" w:lineRule="auto"/>
  <w:ind w:left="0"/>
</w:pPr>'''
            p_lof_header._p.insert(0, parse_xml(pPr_hdr_xml))
            r_tab_hdr = p_lof_header.add_run()
            r_tab_hdr.add_tab()
            r_txt_hdr = p_lof_header.add_run('Page')
            r_txt_hdr.font.name = 'Times New Roman'
            r_txt_hdr.font.size = Pt(12)
            r_txt_hdr.font.bold = False
            r_txt_hdr.font.color.rgb = RGBColor(0x00, 0x00, 0x00)

            for num, fig_title, page_num in lof_entries:
                p_lof = target_p.insert_paragraph_before()
                old_pPr = p_lof._p.find(qn('w:pPr'))
                if old_pPr is not None:
                    p_lof._p.remove(old_pPr)
                
                pPr_xml = f'''<w:pPr {nsdecls("w")}>
  <w:tabs>
    <w:tab w:val="right" w:leader="none" w:pos="8666"/>
  </w:tabs>
  <w:spacing w:before="40" w:after="40" w:line="276" w:lineRule="auto"/>
  <w:ind w:left="0"/>
</w:pPr>'''
                p_lof._p.insert(0, parse_xml(pPr_xml))

                bm_fig_target = f"_Fig_{num.replace('.', '_')}"

                # 1. Figure title hyperlink
                hyperlink_title = OxmlElement('w:hyperlink')
                hyperlink_title.set(qn('w:anchor'), bm_fig_target)
                hyperlink_title.set(qn('w:history'), '1')

                run_text = OxmlElement('w:r')
                rPr_text = OxmlElement('w:rPr')
                rFont_text = OxmlElement('w:rFonts')
                rFont_text.set(qn('w:ascii'), 'Times New Roman')
                rFont_text.set(qn('w:hAnsi'), 'Times New Roman')
                rPr_text.append(rFont_text)
                rPr_text.append(parse_xml(f'<w:color {nsdecls("w")} w:val="000000"/>'))
                rPr_text.append(parse_xml(f'<w:sz {nsdecls("w")} w:val="24"/>'))
                run_text.append(rPr_text)
                t_elem = OxmlElement('w:t')
                t_elem.set(qn('xml:space'), 'preserve')
                t_elem.text = f"Figure {num}    {fig_title}"
                run_text.append(t_elem)
                hyperlink_title.append(run_text)
                p_lof._p.append(hyperlink_title)

                # 2. Tab
                run_tab = OxmlElement('w:r')
                rPr_tab = OxmlElement('w:rPr')
                rFont_tab = OxmlElement('w:rFonts')
                rFont_tab.set(qn('w:ascii'), 'Times New Roman')
                rFont_tab.set(qn('w:hAnsi'), 'Times New Roman')
                rPr_tab.append(rFont_tab)
                rPr_tab.append(parse_xml(f'<w:color {nsdecls("w")} w:val="000000"/>'))
                rPr_tab.append(parse_xml(f'<w:sz {nsdecls("w")} w:val="24"/>'))
                run_tab.append(rPr_tab)
                run_tab.append(OxmlElement('w:tab'))
                p_lof._p.append(run_tab)

                # 3. Dynamic Page Number Hyperlink via PAGEREF field
                fld_fig_xml = f'''<w:fldSimple {nsdecls("w")} w:instr="PAGEREF {bm_fig_target} \\h">
  <w:r>
    <w:rPr>
      <w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman"/>
      <w:color w:val="000000"/>
      <w:sz w:val="24"/>
    </w:rPr>
    <w:t>{page_num}</w:t>
  </w:r>
</w:fldSimple>'''
                p_lof._p.append(parse_xml(fld_fig_xml))

            # Page break after List of Figures before List of Tables
            p_lot_break = target_p.insert_paragraph_before()
            p_lot_break.paragraph_format.space_before = Pt(0)
            p_lot_break.paragraph_format.space_after = Pt(0)
            p_lot_break.add_run().add_break(WD_BREAK.PAGE)

        # 4C. Build List of Tables (Appendix VI: No dot leaders, right-aligned Page header)
        lot_entries = load_latex_list_entries('main.lot')
        if lot_entries:
            print(f'Building List of Tables ({len(lot_entries)} entries)...')
            lot_title = target_p.insert_paragraph_before('List of Tables')
            lot_title.alignment = WD_ALIGN_PARAGRAPH.CENTER
            lot_title.paragraph_format.space_before = Pt(18)
            lot_title.paragraph_format.space_after = Pt(12)
            for r in lot_title.runs:
                r.font.name = 'Times New Roman'
                r.font.size = Pt(12)
                r.font.bold = True
                r.font.color.rgb = RGBColor(0x00, 0x00, 0x00)

            bm_lot_start = OxmlElement('w:bookmarkStart')
            bm_lot_start.set(qn('w:id'), '99')
            bm_lot_start.set(qn('w:name'), '_Toc_List_Tables')
            bm_lot_end = OxmlElement('w:bookmarkEnd')
            bm_lot_end.set(qn('w:id'), '99')
            lot_title._p.insert(0, bm_lot_start)
            lot_title._p.append(bm_lot_end)

            # Right-aligned Page column header
            p_lot_header = target_p.insert_paragraph_before()
            old_pPr = p_lot_header._p.find(qn('w:pPr'))
            if old_pPr is not None:
                p_lot_header._p.remove(old_pPr)
            pPr_hdr_xml = f'''<w:pPr {nsdecls("w")}>
  <w:tabs>
    <w:tab w:val="right" w:leader="none" w:pos="8666"/>
  </w:tabs>
  <w:spacing w:before="60" w:after="80" w:line="276" w:lineRule="auto"/>
  <w:ind w:left="0"/>
</w:pPr>'''
            p_lot_header._p.insert(0, parse_xml(pPr_hdr_xml))
            r_tab_hdr = p_lot_header.add_run()
            r_tab_hdr.add_tab()
            r_txt_hdr = p_lot_header.add_run('Page')
            r_txt_hdr.font.name = 'Times New Roman'
            r_txt_hdr.font.size = Pt(12)
            r_txt_hdr.font.bold = False
            r_txt_hdr.font.color.rgb = RGBColor(0x00, 0x00, 0x00)

            for num, tbl_title, page_num in lot_entries:
                p_lot = target_p.insert_paragraph_before()
                old_pPr = p_lot._p.find(qn('w:pPr'))
                if old_pPr is not None:
                    p_lot._p.remove(old_pPr)
                
                pPr_xml = f'''<w:pPr {nsdecls("w")}>
  <w:tabs>
    <w:tab w:val="right" w:leader="none" w:pos="8666"/>
  </w:tabs>
  <w:spacing w:before="40" w:after="40" w:line="276" w:lineRule="auto"/>
  <w:ind w:left="0"/>
</w:pPr>'''
                p_lot._p.insert(0, parse_xml(pPr_xml))

                bm_tbl_target = f"_Tbl_{num.replace('.', '_')}"

                # 1. Table title hyperlink
                hyperlink_title = OxmlElement('w:hyperlink')
                hyperlink_title.set(qn('w:anchor'), bm_tbl_target)
                hyperlink_title.set(qn('w:history'), '1')

                run_text = OxmlElement('w:r')
                rPr_text = OxmlElement('w:rPr')
                rFont_text = OxmlElement('w:rFonts')
                rFont_text.set(qn('w:ascii'), 'Times New Roman')
                rFont_text.set(qn('w:hAnsi'), 'Times New Roman')
                rPr_text.append(rFont_text)
                rPr_text.append(parse_xml(f'<w:color {nsdecls("w")} w:val="000000"/>'))
                rPr_text.append(parse_xml(f'<w:sz {nsdecls("w")} w:val="24"/>'))
                run_text.append(rPr_text)
                t_elem = OxmlElement('w:t')
                t_elem.set(qn('xml:space'), 'preserve')
                t_elem.text = f"Table {num}    {tbl_title}"
                run_text.append(t_elem)
                hyperlink_title.append(run_text)
                p_lot._p.append(hyperlink_title)

                # 2. Tab
                run_tab = OxmlElement('w:r')
                rPr_tab = OxmlElement('w:rPr')
                rFont_tab = OxmlElement('w:rFonts')
                rFont_tab.set(qn('w:ascii'), 'Times New Roman')
                rFont_tab.set(qn('w:hAnsi'), 'Times New Roman')
                rPr_tab.append(rFont_tab)
                rPr_tab.append(parse_xml(f'<w:color {nsdecls("w")} w:val="000000"/>'))
                rPr_tab.append(parse_xml(f'<w:sz {nsdecls("w")} w:val="24"/>'))
                run_tab.append(rPr_tab)
                run_tab.append(OxmlElement('w:tab'))
                p_lot._p.append(run_tab)

                # 3. Dynamic Page Number Hyperlink via PAGEREF field
                fld_tbl_xml = f'''<w:fldSimple {nsdecls("w")} w:instr="PAGEREF {bm_tbl_target} \\h">
  <w:r>
    <w:rPr>
      <w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman"/>
      <w:color w:val="000000"/>
      <w:sz w:val="24"/>
    </w:rPr>
    <w:t>{page_num}</w:t>
  </w:r>
</w:fldSimple>'''
                p_lot._p.append(parse_xml(fld_tbl_xml))

            # Page break after List of Tables before List of Abbreviations
            p_abbr_break = target_p.insert_paragraph_before()
            p_abbr_break.paragraph_format.space_before = Pt(0)
            p_abbr_break.paragraph_format.space_after = Pt(0)
            r_abr = p_abbr_break.add_run()
            r_abr.add_break(WD_BREAK.PAGE)

    # 5. Format Tables: 100% Crisp, Black, Visible Academic Booktabs Borders
    print(f'Formatting {len(doc.tables)} tables in academic Booktabs style...')
    for tbl in doc.tables:
        tbl_text = " ".join([c.text.strip() for r in tbl.rows for c in r.cells])
        is_declaration_tbl = ('Signature' in tbl_text and 'Date' in tbl_text)
        
        tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
        tblPr = tbl._tbl.tblPr
        
        # Set Table Width to exact printable width
        tblW = tblPr.find(qn('w:tblW'))
        if tblW is None:
            tblW = OxmlElement('w:tblW')
            tblPr.append(tblW)
        tblW.set(qn('w:w'), str(TOTAL_WIDTH))
        tblW.set(qn('w:type'), 'dxa')
        
        if is_declaration_tbl:
            p_parent = tbl._tbl.getparent()
            tbl_idx = p_parent.index(tbl._tbl)
            
            # Paragraph 1: Signature & Date Dotted Line
            p1 = parse_xml(f'''
<w:p {nsdecls("w")}>
  <w:pPr>
    <w:tabs>
      <w:tab w:val="right" w:leader="dot" w:pos="4200"/>
      <w:tab w:val="left" w:leader="none" w:pos="5800"/>
      <w:tab w:val="right" w:leader="dot" w:pos="8666"/>
    </w:tabs>
    <w:spacing w:before="0" w:after="160" w:line="360" w:lineRule="auto"/>
    <w:ind w:left="0"/>
  </w:pPr>
  <w:r>
    <w:rPr>
      <w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman"/>
      <w:sz w:val="24"/>
      <w:color w:val="000000"/>
    </w:rPr>
    <w:t>Signature</w:t>
  </w:r>
  <w:r>
    <w:rPr>
      <w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman"/>
      <w:sz w:val="24"/>
      <w:color w:val="000000"/>
    </w:rPr>
    <w:tab/>
  </w:r>
  <w:r>
    <w:rPr>
      <w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman"/>
      <w:sz w:val="24"/>
      <w:color w:val="000000"/>
    </w:rPr>
    <w:tab/>
  </w:r>
  <w:r>
    <w:rPr>
      <w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman"/>
      <w:sz w:val="24"/>
      <w:color w:val="000000"/>
    </w:rPr>
    <w:tab/>
  </w:r>
</w:p>''')

            # Paragraph 2: Name & Date Label
            p2 = parse_xml(f'''
<w:p {nsdecls("w")}>
  <w:pPr>
    <w:tabs>
      <w:tab w:val="right" w:leader="dot" w:pos="4200"/>
      <w:tab w:val="center" w:leader="none" w:pos="7233"/>
    </w:tabs>
    <w:spacing w:before="0" w:after="0" w:line="360" w:lineRule="auto"/>
    <w:ind w:left="0"/>
  </w:pPr>
  <w:r>
    <w:rPr>
      <w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman"/>
      <w:sz w:val="24"/>
      <w:color w:val="000000"/>
    </w:rPr>
    <w:t xml:space="preserve">    Name</w:t>
  </w:r>
  <w:r>
    <w:rPr>
      <w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman"/>
      <w:sz w:val="24"/>
      <w:color w:val="000000"/>
    </w:rPr>
    <w:tab/>
  </w:r>
  <w:r>
    <w:rPr>
      <w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman"/>
      <w:sz w:val="24"/>
      <w:color w:val="000000"/>
    </w:rPr>
    <w:tab/>
  </w:r>
  <w:r>
    <w:rPr>
      <w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman"/>
      <w:sz w:val="24"/>
      <w:color w:val="000000"/>
    </w:rPr>
    <w:t>Date</w:t>
  </w:r>
</w:p>''')
            p_parent.insert(tbl_idx, p1)
            p_parent.insert(tbl_idx + 1, p2)
            p_parent.remove(tbl._tbl)
            continue
            
        # Explicit solid black table-level borders for academic data tables
        borders_xml = (
            f'<w:tblBorders {nsdecls("w")}>'
            '<w:top w:val="single" w:sz="12" w:space="0" w:color="000000"/>'
            '<w:bottom w:val="single" w:sz="12" w:space="0" w:color="000000"/>'
            '<w:left w:val="none"/>'
            '<w:right w:val="none"/>'
            '<w:insideH w:val="single" w:sz="4" w:space="0" w:color="CCCCCC"/>'
            '<w:insideV w:val="none"/>'
            '</w:tblBorders>'
        )
        old_b = tblPr.find(qn('w:tblBorders'))
        if old_b is not None:
            tblPr.remove(old_b)
        tblPr.append(parse_xml(borders_xml))
        
        ncols = len(tbl.columns)
        nrows = len(tbl.rows)
        if ncols == 0 or nrows == 0:
            continue
            
        col_widths = calculate_proportional_column_widths(tbl, TOTAL_WIDTH)
        is_dense = (ncols >= 6)
        
        for r_idx, row in enumerate(tbl.rows):
            trPr = row._tr.get_or_add_trPr()
            cantSplit = OxmlElement('w:cantSplit')
            trPr.append(cantSplit)
            
            is_header = (r_idx == 0)
            is_last_row = (r_idx == nrows - 1)
            
            if is_header:
                tblHeader = OxmlElement('w:tblHeader')
                trPr.append(tblHeader)
                
            row_text = " ".join([cell.text.strip() for cell in row.cells])
            is_summary_row = any(term in row_text for term in ['Overall', 'Average', 'Total', 'Baseline', 'Selected']) and not is_header
            
            for c_idx, cell in enumerate(row.cells):
                tcPr = cell._tc.get_or_add_tcPr()
                
                # Column width
                if c_idx < len(col_widths):
                    tcW = tcPr.find(qn('w:tcW'))
                    if tcW is None:
                        tcW = OxmlElement('w:tcW')
                        tcPr.append(tcW)
                    tcW.set(qn('w:w'), str(col_widths[c_idx]))
                    tcW.set(qn('w:type'), 'dxa')
                    
                # Clean padding
                tcMar = OxmlElement('w:tcMar')
                for m_name, val in [('top', 100), ('bottom', 100), ('left', 140), ('right', 140)]:
                    node = OxmlElement(f'w:{m_name}')
                    node.set(qn('w:w'), str(val))
                    node.set(qn('w:type'), 'dxa')
                    tcMar.append(node)
                tcPr.append(tcMar)
                
                cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
                
                # REMOVE ANY SHADING / BACKGROUND COLOR
                old_shd = tcPr.find(qn('w:shd'))
                if old_shd is not None:
                    tcPr.remove(old_shd)
                    
                # Set explicit visible black borders on EVERY cell so Google Docs & Word render them with 100% fidelity:
                tcBorders = OxmlElement('w:tcBorders')
                
                if is_header:
                    # Top: solid black 1.5 pt (\toprule)
                    b_top = OxmlElement('w:top')
                    b_top.set(qn('w:val'), 'single')
                    b_top.set(qn('w:sz'), '12')
                    b_top.set(qn('w:space'), '0')
                    b_top.set(qn('w:color'), '000000')
                    tcBorders.append(b_top)
                    
                    # Bottom: solid black 1.0 pt (\midrule)
                    b_bot = OxmlElement('w:bottom')
                    b_bot.set(qn('w:val'), 'single')
                    b_bot.set(qn('w:sz'), '8')
                    b_bot.set(qn('w:space'), '0')
                    b_bot.set(qn('w:color'), '000000')
                    tcBorders.append(b_bot)
                    
                elif is_last_row:
                    # Bottom: solid black 1.5 pt (\bottomrule)
                    b_bot = OxmlElement('w:bottom')
                    b_bot.set(qn('w:val'), 'single')
                    b_bot.set(qn('w:sz'), '12')
                    b_bot.set(qn('w:space'), '0')
                    b_bot.set(qn('w:color'), '000000')
                    tcBorders.append(b_bot)
                    
                    if is_summary_row:
                        b_top = OxmlElement('w:top')
                        b_top.set(qn('w:val'), 'single')
                        b_top.set(qn('w:sz'), '8')
                        b_top.set(qn('w:space'), '0')
                        b_top.set(qn('w:color'), '000000')
                        tcBorders.append(b_top)
                        
                elif is_summary_row:
                    b_top = OxmlElement('w:top')
                    b_top.set(qn('w:val'), 'single')
                    b_top.set(qn('w:sz'), '8')
                    b_top.set(qn('w:space'), '0')
                    b_top.set(qn('w:color'), '000000')
                    tcBorders.append(b_top)
                    
                    b_bot = OxmlElement('w:bottom')
                    b_bot.set(qn('w:val'), 'single')
                    b_bot.set(qn('w:sz'), '4')
                    b_bot.set(qn('w:space'), '0')
                    b_bot.set(qn('w:color'), 'CCCCCC')
                    tcBorders.append(b_bot)
                    
                else:
                    # Clean horizontal gridline for normal rows
                    b_bot = OxmlElement('w:bottom')
                    b_bot.set(qn('w:val'), 'single')
                    b_bot.set(qn('w:sz'), '4')
                    b_bot.set(qn('w:space'), '0')
                    b_bot.set(qn('w:color'), 'CCCCCC')
                    tcBorders.append(b_bot)
                    
                # No vertical lines
                b_left = OxmlElement('w:left')
                b_left.set(qn('w:val'), 'none')
                tcBorders.append(b_left)
                
                b_right = OxmlElement('w:right')
                b_right.set(qn('w:val'), 'none')
                tcBorders.append(b_right)
                
                old_tc_b = tcPr.find(qn('w:tcBorders'))
                if old_tc_b is not None:
                    tcPr.remove(old_tc_b)
                tcPr.append(tcBorders)
                
                # Text formatting inside table cell
                for p in cell.paragraphs:
                    p.paragraph_format.space_before = Pt(2)
                    p.paragraph_format.space_after = Pt(2)
                    p.paragraph_format.line_spacing = 1.05
                    for r in p.runs:
                        r.font.name = 'Times New Roman'
                        r.font.size = Pt(8.5 if is_dense else 10.0)
                        r.font.color.rgb = RGBColor(0x00, 0x00, 0x00)
                        if is_header:
                            r.font.bold = True

    # 6. Format and Scale Math Equations (OMML) to Match NSBM 12 pt Guidelines
    MATH_SZ = '22'  # 11 pt (22 half-points), optical match for 12pt Times New Roman
    print(f'Formatting and scaling mathematical equations to guideline proportions ({int(MATH_SZ)/2} pt)...')
    
    # Scale all math runs
    for m_r in doc._body._element.xpath('.//m:r'):
        m_rPr = m_r.find(qn('m:rPr'))
        if m_rPr is None:
            m_rPr = OxmlElement('m:rPr')
            m_r.insert(0, m_rPr)
        
        w_rPr = m_rPr.find(qn('w:rPr'))
        if w_rPr is None:
            w_rPr = OxmlElement('w:rPr')
            m_rPr.append(w_rPr)
            
        sz = w_rPr.find(qn('w:sz'))
        if sz is None:
            sz = OxmlElement('w:sz')
            w_rPr.append(sz)
        sz.set(qn('w:val'), MATH_SZ)
        
        szCs = w_rPr.find(qn('w:szCs'))
        if szCs is None:
            szCs = OxmlElement('w:szCs')
            w_rPr.append(szCs)
        szCs.set(qn('w:val'), MATH_SZ)

        color = w_rPr.find(qn('w:color'))
        if color is None:
            color = OxmlElement('w:color')
            w_rPr.append(color)
        color.set(qn('w:val'), '000000')

    # Scale all math control properties (delimiters, fractions, radicals, n-ary, matrices)
    for parent in doc._body._element.xpath('.//m:dPr | .//m:fPr | .//m:radPr | .//m:naryPr | .//m:accPr | .//m:mPr | .//m:sSubPr | .//m:sSupPr'):
        ctrlPr = parent.find(qn('m:ctrlPr'))
        if ctrlPr is None:
            ctrlPr = OxmlElement('m:ctrlPr')
            parent.append(ctrlPr)
            
        w_rPr = ctrlPr.find(qn('w:rPr'))
        if w_rPr is None:
            w_rPr = OxmlElement('w:rPr')
            ctrlPr.append(w_rPr)
            
        sz = w_rPr.find(qn('w:sz'))
        if sz is None:
            sz = OxmlElement('w:sz')
            w_rPr.append(sz)
        sz.set(qn('w:val'), MATH_SZ)
        
        szCs = w_rPr.find(qn('w:szCs'))
        if szCs is None:
            szCs = OxmlElement('w:szCs')
            w_rPr.append(szCs)
        szCs.set(qn('w:val'), MATH_SZ)

    # 7. Configure NSBM Guideline Two-Section Pagination
    print('Configuring NSBM Guideline Two-Section Pagination (Roman Front Matter, Arabic Main Matter)...')
    intro_p = None
    intro_idx = None
    for i, p in enumerate(doc.paragraphs):
        st = p.style.name if p.style else ''
        if 'Heading 1' in st and (p.text.startswith('1\t') or p.text.startswith('1 ')):
            intro_p = p
            intro_idx = i
            break
            
    if intro_p and intro_idx > 0:
        p_before = doc.paragraphs[intro_idx - 1]
        pPr = p_before._p.get_or_add_pPr()
        sectPr_sec1 = pPr.find(qn('w:sectPr'))
        if sectPr_sec1 is None:
            sectPr_sec1 = OxmlElement('w:sectPr')
            pPr.append(sectPr_sec1)
            
        # Section 1 page dimensions & margins
        sectPr_sec1.append(parse_xml(f'<w:pgSz {nsdecls("w")} w:w="11906" w:h="16838"/>'))
        sectPr_sec1.append(parse_xml(f'<w:pgMar {nsdecls("w")} w:top="1440" w:bottom="1440" w:left="1800" w:right="1440" w:footer="720"/>'))
        
        # Section 1: Lower-case Roman numerals starting at 1 (i)
        sectPr_sec1.append(parse_xml(f'<w:pgNumType {nsdecls("w")} w:fmt="lowerRoman" w:start="1"/>'))
        
        # Section 1: Title page counts as page i, but number does not appear
        sectPr_sec1.append(parse_xml(f'<w:titlePg {nsdecls("w")}/>'))
        
        if len(doc.sections) >= 2:
            sec1 = doc.sections[0]
            sec2 = doc.sections[1]
            
            # Configure Section 1 footers
            sec1.different_first_page_header_footer = True
            
            # Normal footer for Section 1 (Declaration ii, Acknowledgement iii, Abstract iv, TOC v, Abbreviations vi)
            f_sec1 = sec1.footer
            f_sec1.is_linked_to_previous = False
            p_f1 = f_sec1.paragraphs[0]
            p_f1.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p_f1.text = ''
            p_f1._p.append(parse_xml(f'<w:fldSimple {nsdecls("w")} w:instr="PAGE"><w:r><w:rPr><w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman"/><w:sz w:val="24"/><w:color w:val="000000"/></w:rPr><w:t>ii</w:t></w:r></w:fldSimple>'))
            
            # First Page footer for Section 1 is empty (Title Page has NO page number)
            f_sec1_first = sec1.first_page_footer
            f_sec1_first.is_linked_to_previous = False
            f_sec1_first.paragraphs[0].text = ''
            
            # Section 2: Main Matter (Arabic numerals starting at 1 at Chapter 1 Introduction)
            sec2.different_first_page_header_footer = False
            sec2.header.is_linked_to_previous = False
            f_sec2 = sec2.footer
            f_sec2.is_linked_to_previous = False
            
            # Set Section 2 page numbering: decimal, start at 1
            sectPr_sec2 = sec2._sectPr
            pgNum_sec2 = sectPr_sec2.find(qn('w:pgNumType'))
            if pgNum_sec2 is not None:
                sectPr_sec2.remove(pgNum_sec2)
            sectPr_sec2.append(parse_xml(f'<w:pgNumType {nsdecls("w")} w:fmt="decimal" w:start="1"/>'))
            
            p_f2 = f_sec2.paragraphs[0]
            p_f2.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p_f2.text = ''
            p_f2._p.append(parse_xml(f'<w:fldSimple {nsdecls("w")} w:instr="PAGE"><w:r><w:rPr><w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman"/><w:sz w:val="24"/><w:color w:val="000000"/></w:rPr><w:t>1</w:t></w:r></w:fldSimple>'))

    # Ensure Word automatically refreshes fields on document open
    settings_elem = doc.settings.element
    update_fields = settings_elem.find(qn('w:updateFields'))
    if update_fields is None:
        update_fields = OxmlElement('w:updateFields')
        update_fields.set(qn('w:val'), 'true')
        settings_elem.append(update_fields)
    else:
        update_fields.set(qn('w:val'), 'true')

    doc.save(docx_path)
    print(f'Successfully styled {docx_path} in academic Booktabs format with full pagination and explicit page breaks!')
    
    # Pre-populate exact DOCX page numbers using headless LibreOffice evaluation
    populate_exact_docx_page_numbers(docx_path)

if __name__ == '__main__':
    target = sys.argv[1] if len(sys.argv) > 1 else 'thesis.docx'
    style_thesis_docx(target)

