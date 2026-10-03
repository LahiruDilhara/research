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
    
    # 2. Add Bookmarks to all Headings for Hyperlinked Table of Contents
    headings_data = []
    bookmark_id_counter = 100
    
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
            
        if lvl is not None and not t.startswith('Table of Contents') and not t.startswith('Contents'):
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
            
            headings_data.append((lvl, t, bm_name))

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
            
        if cover_page_mode:
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            
            if 'CUSTOMIZABLE PAPER-BASED VIRTUAL' in t.upper():
                p.paragraph_format.space_before = Pt(36)
                p.paragraph_format.space_after = Pt(28)
                p.paragraph_format.line_spacing = 1.5
                for r in p.runs:
                    r.font.name = 'Times New Roman'
                    r.font.size = Pt(16)
                    r.font.bold = True
                    r.font.color.rgb = RGBColor(0x00, 0x00, 0x00)
            elif 'G A LAHIRU DILHARA' in t.upper():
                p.paragraph_format.space_before = Pt(24)
                p.paragraph_format.space_after = Pt(24)
                for r in p.runs:
                    r.font.name = 'Times New Roman'
                    r.font.size = Pt(14)
                    r.font.bold = True
                    r.font.color.rgb = RGBColor(0x00, 0x00, 0x00)
            elif 'A thesis submitted to NSBM' in t or 'Bachelor of Science' in t:
                p.paragraph_format.space_after = Pt(6)
                for r in p.runs:
                    r.font.name = 'Times New Roman'
                    r.font.size = Pt(14)
                    r.font.bold = False
                    r.font.color.rgb = RGBColor(0x00, 0x00, 0x00)
            elif 'By' == t:
                p.paragraph_format.space_before = Pt(18)
                p.paragraph_format.space_after = Pt(12)
                for r in p.runs:
                    r.font.name = 'Times New Roman'
                    r.font.size = Pt(14)
                    r.font.bold = False
                    r.font.color.rgb = RGBColor(0x00, 0x00, 0x00)
            elif any(inst in t for inst in ['Department of Computer Science', 'Faculty of Computing', 'NSBM Green University', 'Sri Lanka']):
                p.paragraph_format.space_after = Pt(4)
                for r in p.runs:
                    r.font.name = 'Times New Roman'
                    r.font.size = Pt(14)
                    r.font.bold = True
                    r.font.color.rgb = RGBColor(0x00, 0x00, 0x00)
            elif 'September 2026' in t:
                p.paragraph_format.space_before = Pt(18)
                p.paragraph_format.space_after = Pt(24)
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
                
                c_num = tbl_counter.get(current_chapter, 0) + 1
                tbl_counter[current_chapter] = c_num
                label_prefix = f"Table {current_chapter}.{c_num}: "
                
                clean_title = re.sub(r"^Table\s+\w+\.\w+:\s*", "", t)
                p.text = ""
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
                
            elif 'Image Caption' in st or (st == 'Caption' and not t.startswith('Table ')) or (t.startswith('Figure ') and len(t) < 200):
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                p.paragraph_format.space_before = Pt(4)
                p.paragraph_format.space_after = Pt(12)
                
                c_num = fig_counter.get(current_chapter, 0) + 1
                fig_counter[current_chapter] = c_num
                label_prefix = f"Figure {current_chapter}.{c_num}: "
                
                clean_title = re.sub(r"^Figure\s+\w+\.\w+:\s*", "", t)
                p.text = ""
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
                    
            else:
                p.paragraph_format.space_before = Pt(0)
                p.paragraph_format.space_after = Pt(6)
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
            
        for lvl, title_text, bm_name in headings_data:
            p_toc = target_p.insert_paragraph_before()
            p_toc.paragraph_format.space_after = Pt(3)
            p_toc.paragraph_format.line_spacing = 1.15
            
            if lvl == 1:
                p_toc.paragraph_format.left_indent = Inches(0.0)
                p_toc.paragraph_format.space_before = Pt(6)
            elif lvl == 2:
                p_toc.paragraph_format.left_indent = Inches(0.25)
            elif lvl == 3:
                p_toc.paragraph_format.left_indent = Inches(0.50)
                
            hyperlink = OxmlElement('w:hyperlink')
            hyperlink.set(qn('w:anchor'), bm_name)
            hyperlink.set(qn('w:history'), '1')
            
            run = OxmlElement('w:r')
            rPr = OxmlElement('w:rPr')
            
            rFont = OxmlElement('w:rFonts')
            rFont.set(qn('w:ascii'), 'Times New Roman')
            rFont.set(qn('w:hAnsi'), 'Times New Roman')
            rPr.append(rFont)
            
            color = OxmlElement('w:color')
            color.set(qn('w:val'), '000000')
            rPr.append(color)
            
            if lvl == 1:
                b = OxmlElement('w:b')
                rPr.append(b)
                
            sz = OxmlElement('w:sz')
            sz.set(qn('w:val'), '24' if lvl == 1 else ('22' if lvl == 2 else '20'))
            rPr.append(sz)
            
            run.append(rPr)
            
            text_elem = OxmlElement('w:t')
            text_elem.text = title_text
            run.append(text_elem)
            
            hyperlink.append(run)
            p_toc._p.append(hyperlink)

        # Page break after TOC before List of Abbreviations
        p_abbr_break = target_p.insert_paragraph_before()
        p_abbr_break.paragraph_format.space_before = Pt(0)
        p_abbr_break.paragraph_format.space_after = Pt(0)
        r_abr = p_abbr_break.add_run()
        r_abr.add_break(WD_BREAK.PAGE)

    # 5. Format Tables: 100% Crisp, Black, Visible Academic Booktabs Borders
    print(f'Formatting {len(doc.tables)} tables in academic Booktabs style...')
    for tbl in doc.tables:
        tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
        tblPr = tbl._tbl.tblPr
        
        # Set Table Width to exact printable width
        tblW = tblPr.find(qn('w:tblW'))
        if tblW is None:
            tblW = OxmlElement('w:tblW')
            tblPr.append(tblW)
        tblW.set(qn('w:w'), str(TOTAL_WIDTH))
        tblW.set(qn('w:type'), 'dxa')
        
        # Explicit solid black table-level borders
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

    doc.save(docx_path)
    print(f'Successfully styled {docx_path} in academic Booktabs format with full pagination and explicit page breaks!')

if __name__ == '__main__':
    target = sys.argv[1] if len(sys.argv) > 1 else 'thesis.docx'
    style_thesis_docx(target)
