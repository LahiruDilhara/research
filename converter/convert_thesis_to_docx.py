#!/usr/bin/env python3
"""
LaTeX to DOCX Conversion Suite for Thesis
1. Strips \resizebox wrappers from figures and tables.
2. Renders all TikZ diagrams into 300 DPI high-resolution PNG images.
3. Converts single-cell algorithm minipages into structured 2-column LaTeX tables.
4. Cleans longtable/tabularx artifacts for clean Pandoc conversion.
5. Preserves all native LaTeX equations (display and inline) for Pandoc native OMML output.
6. Generates a clean, Pandoc-compatible main.tex with minimal preamble.
7. Invokes Pandoc with IEEE citation processing (--citeproc, --csl, --bibliography),
   real Table of Contents, native equations, and section numbering.
8. Applies docx_styler.py for explicit page breaks on all chapters/preliminary sections,
   crisp visible Booktabs table borders, and Times New Roman academic typography.
"""

import os
import re
import sys
import shutil
import subprocess
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
            subprocess.run(cmd_ppm, capture_output=True, text=True)
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

def clean_for_pandoc(content: str, filename: str = "") -> str:
    """Preprocess LaTeX quirks for clean Pandoc conversion."""
    # Convert longtable specs to simple columns for Pandoc table parsing
    content = re.sub(r"\\endfirsthead.*?\\endlastfoot\s*", "", content, flags=re.DOTALL)
    content = re.sub(r"\\begin\{longtable\}[^\n]+", r"\\begin{longtable}{l p{13.5cm}}", content)
    content = re.sub(r"\\multicolumn\{2\}\{r@?\{\}\}\{\\scriptsize Continued on next page\\dots\}", "", content)
    
    # Convert \tabularx specs to regular tabular
    content = re.sub(r"\\begin\{tabularx\}\{.*?\}(\{.*?\})", r"\\begin{tabular}\1", content)
    content = re.sub(r"\\end\{tabularx\}", r"\\end{tabular}", content)
    
    # Convert \tabular* specs to regular tabular
    content = re.sub(r"\\begin\{tabular\*\}\{.*?\}(\{.*?\})", r"\\begin{tabular}\1", content)
    content = re.sub(r"\\end\{tabular\*\}", r"\\end{tabular}", content)

    # Convert \paragraph{...} to inline bold lead-in \textbf{...} to prevent weird heading numbers
    content = re.sub(r"\\paragraph\{(.*?)\}", r"\n\n\\textbf{\1} ", content)
    
    # Clean LaTeX math in captions to readable Unicode text so Pandoc doesn't produce [Equation] or empty text
    def clean_caption_math(m):
        cap = m.group(1)
        cap = re.sub(r"\$L_{\\text\{hand\}}\$", r"L_hand", cap)
        cap = re.sub(r"\$\\mathbf\{P\}_(\d+)\$", r"P_\1", cap)
        cap = re.sub(r"\$P_(\d+)\$", r"P_\1", cap)
        cap = re.sub(r"\$W\$", r"W", cap)
        cap = re.sub(r"\$S\$", r"S", cap)
        cap = re.sub(r"\$0\^\\circ\$", r"0°", cap)
        cap = re.sub(r"\$75\^\\circ\$", r"75°", cap)
        cap = re.sub(r"\$29\.09\\text\{\s*ms\}\$", r"29.09 ms", cap)
        cap = re.sub(r"\$(\d+(?:\.\d+)?)\s*\\text\{\s*ms\}\$", r"\1 ms", cap)
        cap = re.sub(r"\$([^\$]+)\$", r"\1", cap)
        return f"\\caption{{{cap}}}"
        
    content = re.sub(r"\\caption\{(.*?)\}", clean_caption_math, content)

    # Clean specific preliminary files so Pandoc gets clear unnumbered chapters
    if filename == "declaration.tex":
        content = re.sub(r"\\begin\{center\}.*?\\textbf\{DECLARATION\}.*?\\end\{center\}", r"\\chapter*{Declaration}", content, flags=re.DOTALL)
    elif filename == "acknowledgement.tex":
        content = re.sub(r"\\begin\{center\}.*?\\textbf\{ACKNOWLEDGEMENT\}.*?\\end\{center\}", r"\\chapter*{Acknowledgement}", content, flags=re.DOTALL)
    elif filename == "abstract.tex":
        content = re.sub(r"\\begin\{center\}.*?\\textbf\{ABSTRACT\}.*?\\end\{center\}", r"\\chapter*{Abstract}", content, flags=re.DOTALL)
        
    return content

def generate_pandoc_main_tex() -> str:
    """Generates a clean, minimal main.tex for Pandoc containing the single official Title Page and inputs."""
    tex = [
        r"\documentclass[12pt,a4paper]{report}",
        r"\usepackage[utf8]{inputenc}",
        r"\usepackage{amsmath,amsfonts,amssymb}",
        r"\usepackage{graphicx}",
        r"\usepackage{booktabs}",
        r"\usepackage{longtable}",
        r"\usepackage{hyperref}",
        r"\begin{document}",
        r"",
        r"% Title Page",
        r"\begin{center}",
        r"{\fontsize{16pt}{24pt}\selectfont \textbf{CUSTOMIZABLE PAPER-BASED VIRTUAL MACRO KEYBOARD SYSTEM USING COMPUTER VISION AND DEEP LEARNING} \par}",
        r"\vspace*{1.5cm}",
        r"{\fontsize{14pt}{20pt}\selectfont A thesis submitted to NSBM Green University for the degree of \par}",
        r"{\fontsize{14pt}{20pt}\selectfont Bachelor of Science (Honours) in Computer Science \par}",
        r"\vspace*{1.2cm}",
        r"{\fontsize{14pt}{20pt}\selectfont By \par}",
        r"\vspace*{1.0cm}",
        r"{\fontsize{14pt}{20pt}\selectfont \textbf{G A LAHIRU DILHARA} \par}",
        r"\vspace*{1.5cm}",
        r"{\fontsize{14pt}{20pt}\selectfont Department of Computer Science \par}",
        r"{\fontsize{14pt}{20pt}\selectfont Faculty of Computing \par}",
        r"{\fontsize{14pt}{20pt}\selectfont NSBM Green University \par}",
        r"{\fontsize{14pt}{20pt}\selectfont Sri Lanka \par}",
        r"\vspace*{1.0cm}",
        r"{\fontsize{14pt}{20pt}\selectfont September 2026 \par}",
        r"\end{center}",
        r"",
        r"\pagebreak",
        r"",
        r"% Preliminary Matter",
        r"\input{chapters/declaration.tex}",
        r"\pagebreak",
        r"\input{chapters/acknowledgement.tex}",
        r"\pagebreak",
        r"\input{chapters/abstract.tex}",
        r"\pagebreak",
        r"\input{chapters/abbreviations.tex}",
        r"\pagebreak",
        r"",
        r"% Main Chapters",
        r"\input{chapters/chapter01.tex}",
        r"\pagebreak",
        r"\input{chapters/chapter02.tex}",
        r"\pagebreak",
        r"\input{chapters/chapter03.tex}",
        r"\pagebreak",
        r"\input{chapters/chapter04.tex}",
        r"\pagebreak",
        r"\input{chapters/chapter05.tex}",
        r"\pagebreak",
        r"\input{chapters/chapter06.tex}",
        r"\pagebreak",
        r"",
        r"% Appendices",
        r"\appendix",
        r"\input{chapters/appendixA.tex}",
        r"\pagebreak",
        r"\input{chapters/appendixB.tex}",
        r"\pagebreak",
        r"\input{chapters/appendixC.tex}",
        r"\pagebreak",
        r"\input{chapters/appendixD.tex}",
        r"\pagebreak",
        r"\input{chapters/appendixE.tex}",
        r"\pagebreak",
        r"",
        r"% References (Placed at end of document)",
        r"\chapter*{References}",
        r"\printbibliography",
        r"",
        r"\end{document}"
    ]
    return "\n".join(tex)

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
        print(f"Processing chapter: {tex_file.name}")
        with open(tex_file, "r", encoding="utf-8") as f:
            content = f.read()
            
        # Step 1: Strip \resizebox
        content_no_resize = strip_resizebox(content)
        # Step 2: Convert single-cell Algorithm minipages to clean 2-column tables
        content_with_algs = convert_algorithms_to_tables(content_no_resize)
        # Step 3: Render TikZ diagrams to 300 DPI PNG
        content_with_figs = extract_and_render_tikz(content_with_algs, chapter_name, fig_tracker)
        # Step 4: Clean LaTeX artifacts for Pandoc (preserving native LaTeX equations)
        cleaned_content = clean_for_pandoc(content_with_figs, tex_file.name)
        
        target_file = build_chapters_dir / tex_file.name
        with open(target_file, "w", encoding="utf-8") as f:
            f.write(cleaned_content)
            
    # Generate clean main.tex for Pandoc
    build_main_tex = BUILD_DIR / "main.tex"
    with open(build_main_tex, "w", encoding="utf-8") as f:
        f.write(generate_pandoc_main_tex())
        
    # Copy research-db and ieee.csl
    build_research_db = BUILD_DIR / "research-db"
    build_research_db.mkdir(parents=True, exist_ok=True)
    shutil.copy(WORKSPACE_ROOT / "research-db" / "references.bib", build_research_db / "references.bib")
    
    csl_file = WORKSPACE_ROOT / "converter" / "ieee.csl"
    out_dir = WORKSPACE_ROOT / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    output_docx = out_dir / "thesis.docx"
    
    print("\n=== Running Pandoc ===")
    cmd_pandoc = [
        str(PANDOC_BIN),
        str(build_main_tex),
        "--from=latex",
        "--to=docx",
        "--citeproc",
        "--metadata=link-citations:true",
        f"--bibliography={build_research_db / 'references.bib'}",
        f"--csl={csl_file}",
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
    
    # Sync with out/ directory for backward compatibility
    out_compat = WORKSPACE_ROOT / "out"
    if out_compat.exists():
        shutil.copy2(output_docx, out_compat / "thesis.docx")

    size_mb = output_docx.stat().st_size / (1024*1024)
    print(f"\n[COMPLETED] Polished Thesis DOCX ready at:\n{output_docx} ({size_mb:.2f} MB)")

if __name__ == "__main__":
    main()
