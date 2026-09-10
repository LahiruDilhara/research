import os
import sys
import docx
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import qn, nsdecls
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL

def style_thesis_docx(docx_path):
    print(f'Loading {docx_path}...')
    doc = docx.Document(docx_path)
    
    # 1. Page Margins (1 inch = 1440 dxa)
    for s in doc.sections:
        s.top_margin = Inches(1.0)
        s.bottom_margin = Inches(1.0)
        s.left_margin = Inches(1.0)
        s.right_margin = Inches(1.0)
        s.page_width = Inches(8.5)
        s.page_height = Inches(11.0)
        
    TOTAL_WIDTH = 9360  # 6.5 in printable area in dxa
    
    # 2. Add Bookmarks to all Headings for Hyperlinking
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

    # 3. Format Cover Page & Cleanup Duplicate TOC Headings
    cover_page_mode = True
    cover_page_elements = [
        'Customizable Paper-Based Virtual Keyboard System',
        'G A Lahiru Dilhara',
        'Student No: 29096',
        'Supervised by:',
        'Prof. Chaminda Wijesinghe',
        'Degree of Bachelor of Science',
        'Department of Computer Science',
        'Faculty of Computing',
        'National School of Business Management'
    ]
    
    for p in list(doc.paragraphs):
        st = p.style.name if p.style else ''
        t = p.text.strip()
        
        # Check if cover page ends
        if any(run.contains_page_break for run in p.runs) or 'List of Abbreviations' in t or 'Chapter 1' in t:
            cover_page_mode = False
            
        if cover_page_mode:
            # Remove any unwanted TOC title duplicated on cover page
            if (t == 'Table of Contents' or t == 'Contents') and not any(elem in t for elem in cover_page_elements):
                p._p.getparent().remove(p._p)
                continue
                
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            
            if 'Customizable Paper-Based Virtual Keyboard System' in t:
                p.paragraph_format.space_before = Pt(36)
                p.paragraph_format.space_after = Pt(28)
                for r in p.runs:
                    r.font.name = 'Calibri'
                    r.font.size = Pt(22)
                    r.font.bold = True
                    r.font.color.rgb = RGBColor(0x0F, 0x29, 0x42)
            elif 'G A Lahiru Dilhara' in t:
                p.paragraph_format.space_before = Pt(24)
                p.paragraph_format.space_after = Pt(4)
                for r in p.runs:
                    r.font.name = 'Calibri'
                    r.font.size = Pt(14)
                    r.font.bold = True
                    r.font.color.rgb = RGBColor(0x1E, 0x29, 0x3B)
            elif 'Student No' in t:
                p.paragraph_format.space_after = Pt(24)
                for r in p.runs:
                    r.font.name = 'Calibri'
                    r.font.size = Pt(11)
                    r.font.color.rgb = RGBColor(0x47, 0x55, 0x69)
            elif 'Supervised by:' in t:
                p.paragraph_format.space_after = Pt(2)
                for r in p.runs:
                    r.font.name = 'Calibri'
                    r.font.size = Pt(11)
                    r.font.bold = True
                    r.font.color.rgb = RGBColor(0x1E, 0x29, 0x3B)
            elif 'Prof. Chaminda Wijesinghe' in t:
                p.paragraph_format.space_after = Pt(36)
                for r in p.runs:
                    r.font.name = 'Calibri'
                    r.font.size = Pt(11.5)
                    r.font.bold = True
                    r.font.color.rgb = RGBColor(0x1E, 0x29, 0x3B)
            elif 'Degree of Bachelor of Science' in t:
                p.paragraph_format.space_after = Pt(36)
                for r in p.runs:
                    r.font.name = 'Calibri'
                    r.font.size = Pt(11)
                    r.font.italic = True
                    r.font.color.rgb = RGBColor(0x1E, 0x29, 0x3B)
            elif any(inst in t for inst in ['Department of Computer Science', 'Faculty of Computing', 'National School of Business Management']):
                p.paragraph_format.space_after = Pt(4)
                for r in p.runs:
                    r.font.name = 'Calibri'
                    r.font.size = Pt(11)
                    r.font.bold = True
                    r.font.color.rgb = RGBColor(0x0F, 0x29, 0x42)
            else:
                p.paragraph_format.space_after = Pt(6)
                for r in p.runs:
                    r.font.name = 'Calibri'
                    r.font.size = Pt(11)
                    r.font.color.rgb = RGBColor(0x1E, 0x29, 0x3B)
                    
        else:
            if 'Heading 1' in st:
                p.paragraph_format.space_before = Pt(24)
                p.paragraph_format.space_after = Pt(8)
                p.paragraph_format.keep_with_next = True
                if t and not t.startswith('List of Abbreviations') and any(str(i) in t for i in range(1, 8)):
                    p.paragraph_format.page_break_before = True
                elif 'References' in t:
                    p.paragraph_format.page_break_before = True
                elif 'List of Abbreviations' in t:
                    p.paragraph_format.page_break_before = True
                    
                for r in p.runs:
                    r.font.name = 'Calibri'
                    r.font.size = Pt(17)
                    r.font.bold = True
                    r.font.color.rgb = RGBColor(0x0F, 0x29, 0x42)
                    
            elif 'Heading 2' in st:
                p.paragraph_format.space_before = Pt(14)
                p.paragraph_format.space_after = Pt(4)
                p.paragraph_format.keep_with_next = True
                for r in p.runs:
                    r.font.name = 'Calibri'
                    r.font.size = Pt(13.5)
                    r.font.bold = True
                    r.font.color.rgb = RGBColor(0x1E, 0x40, 0xAF)
                    
            elif 'Heading 3' in st:
                p.paragraph_format.space_before = Pt(10)
                p.paragraph_format.space_after = Pt(2)
                p.paragraph_format.keep_with_next = True
                for r in p.runs:
                    r.font.name = 'Calibri'
                    r.font.size = Pt(11.5)
                    r.font.bold = True
                    r.font.color.rgb = RGBColor(0x33, 0x41, 0x55)
                    
            elif 'Caption' in st or t.startswith('Figure ') or t.startswith('Table '):
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                p.paragraph_format.space_before = Pt(4)
                p.paragraph_format.space_after = Pt(10)
                for r in p.runs:
                    r.font.name = 'Calibri'
                    r.font.size = Pt(10)
                    r.font.italic = True
                    r.font.color.rgb = RGBColor(0x47, 0x55, 0x69)
                    
            else:
                p.paragraph_format.space_after = Pt(5)
                p.paragraph_format.line_spacing = 1.15
                for r in p.runs:
                    if not r.font.name:
                        r.font.name = 'Calibri'
                    if not r.font.size:
                        r.font.size = Pt(11)
                    if not r.font.color.rgb:
                        r.font.color.rgb = RGBColor(0x1E, 0x29, 0x3B)

    # 4. Build 100% Clickable Hyperlinked Table of Contents for Google Docs & Word
    target_p = None
    for p in doc.paragraphs:
        if 'List of Abbreviations' in p.text:
            target_p = p
            break
            
    if target_p and headings_data:
        print(f'Building 100% Hyperlinked Table of Contents ({len(headings_data)} entries)...')
        
        toc_title = target_p.insert_paragraph_before('Table of Contents')
        toc_title.paragraph_format.space_before = Pt(24)
        toc_title.paragraph_format.space_after = Pt(12)
        toc_title.paragraph_format.page_break_before = True
        for r in toc_title.runs:
            r.font.name = 'Calibri'
            r.font.size = Pt(18)
            r.font.bold = True
            r.font.color.rgb = RGBColor(0x0F, 0x29, 0x42)
            
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
                
            # Create OpenXML Hyperlink element pointing to heading bookmark
            hyperlink = OxmlElement('w:hyperlink')
            hyperlink.set(qn('w:anchor'), bm_name)
            hyperlink.set(qn('w:history'), '1')
            
            run = OxmlElement('w:r')
            rPr = OxmlElement('w:rPr')
            
            rFont = OxmlElement('w:rFonts')
            rFont.set(qn('w:ascii'), 'Calibri')
            rFont.set(qn('w:hAnsi'), 'Calibri')
            rPr.append(rFont)
            
            color = OxmlElement('w:color')
            if lvl == 1:
                color.set(qn('w:val'), '0F2942')
                b = OxmlElement('w:b')
                rPr.append(b)
            elif lvl == 2:
                color.set(qn('w:val'), '1E40AF')
            else:
                color.set(qn('w:val'), '334155')
            rPr.append(color)
            
            sz = OxmlElement('w:sz')
            sz.set(qn('w:val'), '22' if lvl == 1 else ('21' if lvl == 2 else '20'))
            rPr.append(sz)
            
            run.append(rPr)
            
            text_elem = OxmlElement('w:t')
            text_elem.text = title_text
            run.append(text_elem)
            
            hyperlink.append(run)
            p_toc._p.append(hyperlink)

    # 5. Format Tables (including List of Abbreviations)
    print(f'Formatting {len(doc.tables)} tables...')
    for tbl in doc.tables:
        tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
        tblPr = tbl._tbl.tblPr
        
        # Set Table Width
        tblW = tblPr.find(qn('w:tblW'))
        if tblW is None:
            tblW = OxmlElement('w:tblW')
            tblPr.append(tblW)
        tblW.set(qn('w:w'), str(TOTAL_WIDTH))
        tblW.set(qn('w:type'), 'dxa')
        
        # Borders
        borders_xml = (
            f'<w:tblBorders {nsdecls("w")}>'
            '<w:top w:val="single" w:sz="12" w:space="0" w:color="1B365D"/>'
            '<w:left w:val="none"/>'
            '<w:bottom w:val="single" w:sz="12" w:space="0" w:color="1B365D"/>'
            '<w:right w:val="none"/>'
            '<w:insideH w:val="single" w:sz="4" w:space="0" w:color="E2E8F0"/>'
            '<w:insideV w:val="none"/>'
            '</w:tblBorders>'
        )
        old_b = tblPr.find(qn('w:tblBorders'))
        if old_b is not None:
            tblPr.remove(old_b)
        tblPr.append(parse_xml(borders_xml))
        
        ncols = len(tbl.columns)
        if ncols == 0:
            continue
            
        if ncols == 2:
            col_widths = [2340, 7020]
        elif ncols == 3:
            col_widths = [1680, 2520, 5160]
        elif ncols == 4:
            col_widths = [1872, 2340, 2808, 2340]
        elif ncols == 5:
            col_widths = [1872] * 5
        elif ncols >= 7:
            base_w = TOTAL_WIDTH // ncols
            col_widths = [base_w] * ncols
            col_widths[-1] += TOTAL_WIDTH - sum(col_widths)
        else:
            base_w = TOTAL_WIDTH // ncols
            col_widths = [base_w] * ncols
            col_widths[-1] += TOTAL_WIDTH - sum(col_widths)
            
        is_dense = (ncols >= 7)
        
        for r_idx, row in enumerate(tbl.rows):
            trPr = row._tr.get_or_add_trPr()
            cantSplit = OxmlElement('w:cantSplit')
            trPr.append(cantSplit)
            
            is_header = (r_idx == 0)
            if is_header:
                tblHeader = OxmlElement('w:tblHeader')
                trPr.append(tblHeader)
                
            for c_idx, cell in enumerate(row.cells):
                tcPr = cell._tc.get_or_add_tcPr()
                
                if c_idx < len(col_widths):
                    tcW = tcPr.find(qn('w:tcW'))
                    if tcW is None:
                        tcW = OxmlElement('w:tcW')
                        tcPr.append(tcW)
                    tcW.set(qn('w:w'), str(col_widths[c_idx]))
                    tcW.set(qn('w:type'), 'dxa')
                    
                tcMar = OxmlElement('w:tcMar')
                for m_name, val in [('top', 120), ('bottom', 120), ('left', 160), ('right', 160)]:
                    node = OxmlElement(f'w:{m_name}')
                    node.set(qn('w:w'), str(val))
                    node.set(qn('w:type'), 'dxa')
                    tcMar.append(node)
                tcPr.append(tcMar)
                
                cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
                
                if is_header:
                    shd_xml = f'<w:shd {nsdecls("w")} w:fill="1B365D"/>'
                    tcPr.append(parse_xml(shd_xml))
                    for p in cell.paragraphs:
                        p.paragraph_format.space_before = Pt(2)
                        p.paragraph_format.space_after = Pt(2)
                        p.paragraph_format.line_spacing = 1.05
                        for r in p.runs:
                            r.font.name = 'Calibri'
                            r.font.size = Pt(8.5 if is_dense else 9.5)
                            r.font.bold = True
                            r.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
                else:
                    bg_col = 'F8FAFC' if (r_idx % 2 == 0) else 'FFFFFF'
                    if bg_col != 'FFFFFF':
                        shd_xml = f'<w:shd {nsdecls("w")} w:fill="{bg_col}"/>'
                        tcPr.append(parse_xml(shd_xml))
                        
                    for p in cell.paragraphs:
                        p.paragraph_format.space_before = Pt(1.5)
                        p.paragraph_format.space_after = Pt(1.5)
                        p.paragraph_format.line_spacing = 1.08
                        for r in p.runs:
                            r.font.name = 'Calibri'
                            r.font.size = Pt(8.0 if is_dense else 9.5)
                            r.font.color.rgb = RGBColor(0x1E, 0x29, 0x3B)
                            
    doc.save(docx_path)
    print(f'Successfully styled {docx_path}!')

if __name__ == '__main__':
    target = sys.argv[1] if len(sys.argv) > 1 else 'thesis.docx'
    style_thesis_docx(target)
