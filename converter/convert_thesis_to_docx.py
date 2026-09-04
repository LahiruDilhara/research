#!/usr/bin/env python3
"""
LaTeX to DOCX Conversion Suite for Thesis
1. Strips \resizebox wrappers from figures and tables.
2. Renders all TikZ diagrams into 300 DPI high-resolution PNG images.
3. Inserts \includegraphics directives pointing to generated images.
4. Cleans longtable header/footer artifacts for Pandoc.
5. Adds clear unnumbered Chapter headings for preliminary sections and References.
6. Invokes Pandoc with IEEE citation processing (--citeproc, --csl, --bibliography),
   native OMML mathematics, real Table of Contents, and section numbering.
"""

import os
import re
import sys
import shutil
import subprocess
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
BUILD_DIR = WORKSPACE_ROOT / "converter" / "build_docx"
FIGURES_DIR = WORKSPACE_ROOT / "figures"
BUILD_FIGURES_DIR = BUILD_DIR / "figures"
PANDOC_BIN = Path.home() / ".local" / "bin" / "pandoc"
if not PANDOC_BIN.exists():
    PANDOC_BIN = Path(shutil.which("pandoc") or "pandoc")

TIKZ_HEADER = r"""\documentclass[border=3mm,tikz,preview]{standalone}
\usepackage[utf8]{inputenc}
\usepackage{amsmath,amssymb,amsfonts}
\usepackage{graphicx}
\usepackage{tikz}
\usetikzlibrary{shapes.geometric, arrows.meta, positioning, shadows, calc, fit, backgrounds}
\usepackage{xcolor}
\begin{document}
"""

TIKZ_FOOTER = r"""
\end{document}
"""

def strip_resizebox(text: str) -> str:
    """Strips all \\resizebox{...}{...}{...} wrappers, preserving inner content."""
    out = []
    i = 0
    n = len(text)
    while i < n:
        idx = text.find(r'\resizebox', i)
        if idx == -1:
            out.append(text[i:])
            break
        out.append(text[i:idx])
        
        pos = idx + len(r'\resizebox')
        
        def get_brace_content(p):
            while p < n and text[p] in ' \t\r\n%':
                if text[p] == '%':
                    while p < n and text[p] != '\n':
                        p += 1
                p += 1
            if p >= n or text[p] != '{':
                return None, p
            start = p + 1
            depth = 1
            p += 1
            while p < n and depth > 0:
                if text[p] == '{' and (p == 0 or text[p-1] != '\\'):
                    depth += 1
                elif text[p] == '}' and (p == 0 or text[p-1] != '\\'):
                    depth -= 1
                p += 1
            return text[start:p-1], p

        arg1, pos1 = get_brace_content(pos)
        arg2, pos2 = get_brace_content(pos1)
        arg3, pos3 = get_brace_content(pos2)
        
        if arg3 is not None:
            content = arg3.strip()
            if content.startswith('%'):
                content = content[1:].lstrip()
            out.append(content)
            i = pos3
        else:
            out.append(text[idx:idx+10])
            i = idx + 10
            
    return ''.join(out)

def extract_and_render_tikz(tex_content: str, chapter_name: str, fig_tracker: dict) -> str:
    """Finds TikZ pictures, compiles them into 300 DPI PNGs, and replaces with \\includegraphics."""
    pattern = re.compile(r"\\begin\{tikzpicture\}.*?\\end\{tikzpicture\}%?", re.DOTALL)
    
    def replacer(match):
        tikz_code = match.group(0)
        if tikz_code.endswith("%"):
            tikz_code = tikz_code[:-1]
            
        fig_idx = fig_tracker.get(chapter_name, 0) + 1
        fig_tracker[chapter_name] = fig_idx
        
        fig_stem = f"{chapter_name}_fig{fig_idx:02d}"
        tex_file = BUILD_DIR / f"{fig_stem}.tex"
        pdf_file = BUILD_DIR / f"{fig_stem}.pdf"
        png_prefix = BUILD_DIR / fig_stem
        final_png = FIGURES_DIR / f"{fig_stem}.png"
        build_png = BUILD_FIGURES_DIR / f"{fig_stem}.png"
        
        if not final_png.exists():
            print(f"[*] Compiling TikZ diagram: {fig_stem}...")
            with open(tex_file, "w", encoding="utf-8") as f:
                f.write(TIKZ_HEADER + "\n" + tikz_code + "\n" + TIKZ_FOOTER)
                
            cmd_latex = [
                "pdflatex",
                "-interaction=nonstopmode",
                f"-output-directory={BUILD_DIR}",
                str(tex_file)
            ]
            res = subprocess.run(cmd_latex, capture_output=True, text=True)
            if res.returncode != 0:
                print(f"[!] Warning: pdflatex failed for {fig_stem}")
                return tikz_code
                
            cmd_ppm = [
                "pdftoppm",
                "-png",
                "-r", "300",
                "-singlefile",
                str(pdf_file),
                str(png_prefix)
            ]
            res_ppm = subprocess.run(cmd_ppm, capture_output=True, text=True)
            generated_png = BUILD_DIR / f"{fig_stem}.png"
            if generated_png.exists():
                shutil.copy(generated_png, final_png)
                shutil.copy(generated_png, build_png)
                print(f"[+] Successfully generated: {final_png.name}")
            else:
                print(f"[!] Warning: PNG generation failed for {fig_stem}")
                return tikz_code
        else:
            shutil.copy(final_png, build_png)
            print(f"[+] Reusing existing diagram: {final_png.name}")
            
        return f"\n\\includegraphics[width=0.85\\textwidth]{{figures/{fig_stem}.png}}\n"

    return pattern.sub(replacer, tex_content)

def convert_algorithms_to_tables(text: str) -> str:
    """Converts figure-minipage single-cell algorithm boxes into structured 2-column LaTeX tables."""
    pattern = re.compile(
        r"\\begin\{figure\}\[htbp\]\s*\\centering\s*\\begin\{minipage\}\{0\.92\\textwidth\}.*?"
        r"\\begin\{tabular\}\{\|p\{0\.96\\linewidth\}\|\}.*?\\hline\s*\\textbf\{Algorithm\s*(\d+):\s*(.*?)\}\s*\\\\\s*\\hline\s*(.*?)\\hline\s*"
        r"\\end\{tabular\}\s*\\end\{minipage\}\s*\\caption\{(.*?)\}\s*\\label\{(.*?)\}\s*\\end\{figure\}",
        re.DOTALL
    )
    
    def replacer(m):
        alg_num, title, body, caption, label = m.groups()
        lines = [l.strip() for l in body.split(r"\\") if l.strip()]
        
        out = []
        out.append(r"\begin{table}[htbp]")
        out.append(r"\centering")
        out.append(f"\\caption{{{caption.strip()}}}")
        out.append(f"\\label{{{label.strip()}}}")
        out.append(r"\begin{tabular}{l p{13.5cm}}")
        out.append(r"\toprule")
        out.append(f"\\textbf{{Step / Field}} & \\textbf{{Algorithm {alg_num}: {title.strip()}}} \\\\")
        out.append(r"\midrule")
        
        for line in lines:
            line_clean = re.sub(r"^\{?\\small(?:\\sffamily)?\s*", "", line)
            if line_clean.endswith("}"):
                line_clean = line_clean[:-1].strip()
            
            if line_clean.startswith(r"\textbf{Input:}"):
                content = line_clean[len(r"\textbf{Input:}"):].strip()
                out.append(f"\\textbf{{Input:}} & {content} \\\\")
            elif line_clean.startswith(r"\textbf{Output:}"):
                content = line_clean[len(r"\textbf{Output:}"):].strip()
                out.append(f"\\textbf{{Output:}} & {content} \\\\")
                out.append(r"\midrule")
            else:
                match_step = re.match(r"^(\d+:)\s*(.*)$", line_clean)
                if match_step:
                    s_num, s_desc = match_step.groups()
                    out.append(f"\\textbf{{{s_num}}} & {s_desc} \\\\")
                elif line_clean:
                    out.append(f" & {line_clean} \\\\")
                    
        out.append(r"\bottomrule")
        out.append(r"\end{tabular}")
        out.append(r"\end{table}")
        return "\n".join(out)
        
    return pattern.sub(replacer, text)

def clean_for_pandoc(content: str) -> str:
    """Preprocess LaTeX quirks for clean Pandoc conversion."""
    content = re.sub(r"\\endfirsthead.*?\\endhead", "", content, flags=re.DOTALL)
    content = re.sub(r"\\endhead", "", content)
    content = re.sub(r"\\endfoot.*?\\endlastfoot", "", content, flags=re.DOTALL)
    content = re.sub(r"\\endfoot", "", content)
    content = re.sub(r"\\endlastfoot", "", content)
    return content

def main():
    print("=== Starting LaTeX to DOCX Conversion Suite ===")
    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    BUILD_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    
    # Process Chapters
    chapters_dir = WORKSPACE_ROOT / "chapters"
    build_chapters_dir = BUILD_DIR / "chapters"
    build_chapters_dir.mkdir(parents=True, exist_ok=True)
    
    fig_tracker = {}
    
    for tex_file in sorted(chapters_dir.glob("*.tex")):
        chapter_name = tex_file.stem
        print(f"\nProcessing chapter: {tex_file.name}")
        with open(tex_file, "r", encoding="utf-8") as f:
            content = f.read()
            
        # Step 1: Strip \resizebox
        content_no_resize = strip_resizebox(content)
        # Step 2: Convert single-cell Algorithm minipages to clean 2-column tables
        content_with_algs = convert_algorithms_to_tables(content_no_resize)
        # Step 3: Render TikZ diagrams to PNG
        content_with_figs = extract_and_render_tikz(content_with_algs, chapter_name, fig_tracker)
        # Step 4: Clean LaTeX artifacts
        cleaned_content = clean_for_pandoc(content_with_figs)
        
        target_file = build_chapters_dir / tex_file.name
        with open(target_file, "w", encoding="utf-8") as f:
            f.write(cleaned_content)
            
    # Process main.tex
    main_tex = WORKSPACE_ROOT / "main.tex"
    with open(main_tex, "r", encoding="utf-8") as f:
        main_content = f.read()
        
    main_content_no_resize = strip_resizebox(main_content)
    main_content_with_algs = convert_algorithms_to_tables(main_content_no_resize)
    main_content_with_figs = extract_and_render_tikz(main_content_with_algs, "main", fig_tracker)
    cleaned_main_content = clean_for_pandoc(main_content_with_figs)
    
    # Ensure explicit References heading before bibliography
    cleaned_main_content = cleaned_main_content.replace(
        r"\printbibliography[title={References}]",
        "\\chapter*{References}\n\\printbibliography"
    )
    
    build_main_tex = BUILD_DIR / "main.tex"
    with open(build_main_tex, "w", encoding="utf-8") as f:
        f.write(cleaned_main_content)
        
    # Copy research-db and ieee.csl
    build_research_db = BUILD_DIR / "research-db"
    build_research_db.mkdir(parents=True, exist_ok=True)
    shutil.copy(WORKSPACE_ROOT / "research-db" / "references.bib", build_research_db / "references.bib")
    
    csl_file = WORKSPACE_ROOT / "converter" / "ieee.csl"
    output_docx = WORKSPACE_ROOT / "thesis.docx"
    
    print("\n=== Running Pandoc ===")
    cmd_pandoc = [
        str(PANDOC_BIN),
        str(build_main_tex),
        "--from=latex",
        "--to=docx",
        "--citeproc",
        f"--bibliography={build_research_db / 'references.bib'}",
        f"--csl={csl_file}",
        "--toc",
        "--toc-depth=3",
        "--number-sections",
        f"--resource-path={BUILD_DIR}:{BUILD_FIGURES_DIR}:{WORKSPACE_ROOT}:{FIGURES_DIR}",
        "-o", str(output_docx)
    ]
    
    print(f"Command: {' '.join(cmd_pandoc)}")
    res = subprocess.run(cmd_pandoc, capture_output=True, text=True, cwd=str(BUILD_DIR))
    
    if res.returncode == 0:
        print(f"\n[SUCCESS] Pandoc exported initial DOCX.")
    else:
        print(f"\n[ERROR] Pandoc failed with exit code {res.returncode}:\n{res.stderr}")
        sys.exit(1)

    # Step 5: Run OpenXML Styler
    styler_script = WORKSPACE_ROOT / "converter" / "docx_styler.py"
    print(f"\n=== Applying OpenXML Professional Academic Styling ({styler_script.name}) ===")
    res_style = subprocess.run(
        ["uv", "run", "--with", "python-docx", "python3", str(styler_script), str(output_docx)],
        capture_output=True, text=True
    )
    if res_style.returncode != 0:
        print(f"[ERROR] Styler failed:\n{res_style.stderr}")
        sys.exit(1)
        
    print(res_style.stdout)
    size_mb = output_docx.stat().st_size / (1024*1024)
    print(f"\n[COMPLETED] Polished Thesis DOCX ready at:\n{output_docx} ({size_mb:.2f} MB)")

if __name__ == "__main__":
    main()
