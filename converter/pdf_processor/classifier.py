import re
from statistics import median
from typing import Dict, List, Any, Tuple, Set


class TextClassifier:
    """
    Classifies text blocks and spans in PDF documents to strictly isolate body paragraphs
    and exclude math formulas, equations, graphs, figures, tables, code blocks, titles,
    headings, TOC, references, headers, footers, and citations.
    """

    MATH_FONT_KEYWORDS = [
        "cmmi", "cmsy", "cmex", "lmmi", "lmsy", "math", "stix",
        "msam", "msbm", "eufm", "eurb", "txsy", "pxsy"
    ]

    SERIF_FONT_KEYWORDS = [
        "cmr", "cmbx", "cmti", "lmr", "lmroman", "roman", "times",
        "serif", "georgia", "garamond", "palatino", "baskerville"
    ]

    MONO_FONT_KEYWORDS = [
        "cmtt", "lmtt", "mono", "courier", "console", "code", "typewriter"
    ]

    def __init__(self, doc_font_stats: Dict[str, float] = None):
        self.median_font_size = doc_font_stats.get("median_size", 10.0) if doc_font_stats else 10.0
        self.in_references_section = False

    @staticmethod
    def calculate_document_font_stats(pages_dict: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Calculates median font size across the document to set baseline body text size.
        """
        font_sizes = []
        for page in pages_dict:
            for block in page.get("blocks", []):
                if block.get("type", 0) != 0:
                    continue
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        text = span.get("text", "").strip()
                        if text:
                            font_sizes.append(span.get("size", 10.0))
        
        if not font_sizes:
            return {"median_size": 10.0}
        
        return {"median_size": float(median(font_sizes))}

    @classmethod
    def is_math_font(cls, font_name: str) -> bool:
        """
        Detects if font is a LaTeX math or symbol font.
        """
        fn = font_name.lower()
        return any(k in fn for k in cls.MATH_FONT_KEYWORDS)

    @classmethod
    def is_serif_font(cls, font_name: str) -> bool:
        """
        Detects if font is a Serif / Roman font (like Computer Modern Roman CMR).
        """
        fn = font_name.lower()
        return any(k in fn for k in cls.SERIF_FONT_KEYWORDS)

    @classmethod
    def is_mono_font(cls, font_name: str) -> bool:
        """
        Detects if font is a Monospace font.
        """
        fn = font_name.lower()
        return any(k in fn for k in cls.MONO_FONT_KEYWORDS)

    @classmethod
    def is_bold_font(cls, font_name: str, flags: int) -> bool:
        """
        Strictly detects bold weight using explicit font family keywords.
        Prevents flags & 16 false positives on regular TeX Roman fonts.
        """
        fn = font_name.lower()
        if any(b in fn for b in ["bold", "cmb", "bx", "bld"]):
            return True
        return False

    @classmethod
    def is_italic_font(cls, font_name: str, flags: int) -> bool:
        """
        Detects italic style using font family keywords or PyMuPDF italic bit.
        """
        fn = font_name.lower()
        if any(i in fn for i in ["italic", "oblique", "cmti", "lmti", "mi", "it"]):
            return True
        return bool(flags & 2)

    @staticmethod
    def is_math_or_formula_text(text: str) -> bool:
        """
        Detects mathematical equations, formulas, math variables, and equation numbering.
        Does NOT flag normal English hyphenated words ('paper-based', 'real-time') or slashes ('and/or').
        """
        clean_text = text.strip()
        
        # Standalone equation number format e.g. (1.1), (3.4)
        if re.match(r'^\s*\(\s*\d+(?:\.\d+)*\s*\)\s*$', clean_text):
            return True

        # Math equation/formula indicators (=, \sum, \int, \alpha, \beta, etc.)
        if re.search(r'[=×÷≠≤≥\^_\{\}\\|α-ωΑ-Ω∫∑√∂∇∝∞≈±]', clean_text):
            if re.search(r'[=≠≤≥∫∑√∂∇∝≈]', clean_text) or re.search(r'\\[a-zA-Z]+', clean_text):
                return True

        return False

    def is_header_or_footer(self, bbox: Tuple[float, float, float, float], page_height: float, text: str) -> bool:
        """
        Detects top running headers, bottom running footers, and page numbers.
        """
        y0, y1 = bbox[1], bbox[3]
        
        if y0 < (page_height * 0.08):
            return True
            
        if y1 > (page_height * 0.92):
            return True
            
        clean_text = text.strip()
        if re.match(r'^(?:page\s+)?\d+(?:\s+of\s+\d+)?$', clean_text, re.IGNORECASE):
            return True
            
        return False

    def is_title_or_heading(self, span_size: float, span_flags: int, font_names: Any, text: str) -> bool:
        """
        Detects document titles, chapter titles, section headings, and subsections.
        """
        clean_text = text.strip()
        if not clean_text:
            return False

        if span_size >= (self.median_font_size * 1.12):
            return True

        heading_patterns = [
            r'^(?:chapter|section|part)\s+\d+',
            r'^\d+(?:\.\d+)*\s+[A-Z]',
            r'^(?:abstract|introduction|background|methods|results|discussion|conclusion|acknowledgments)$'
        ]
        for pattern in heading_patterns:
            if re.match(pattern, clean_text, re.IGNORECASE):
                return True

        return False

    def is_toc_line(self, text: str) -> bool:
        """
        Detects Table of Contents lines.
        """
        if re.search(r'\.{3,}\s*\d+$', text) or (re.search(r'\s+\d+$', text) and ("contents" in text.lower() or "table" in text.lower())):
            return True
        return False

    def is_reference_heading_or_block(self, text: str) -> bool:
        """
        Detects if current block starts the references section or is a bibliographic entry.
        """
        clean_text = text.strip().lower()
        if clean_text in ["references", "bibliography", "works cited", "literature cited"]:
            self.in_references_section = True
            return True

        if self.in_references_section:
            return True

        if re.match(r'^\[\d+\]', text.strip()):
            return True

        return False

    def is_caption(self, text: str) -> bool:
        """
        Detects figure and table captions.
        """
        clean_text = text.strip()
        if re.match(r'^(?:figure|fig\.|table|chart|graph)\s+\d+', clean_text, re.IGNORECASE):
            return True
        return False

    def is_body_paragraph(self, block_text: str, font_names: Set[str], span_size: float) -> bool:
        """
        Strictly verifies that a block is a narrative body paragraph.
        Excludes document titles, chapter titles, headings, math formulas, tables, figures, code blocks, lists, and metadata.
        """
        clean_text = block_text.strip()
        words = clean_text.split()
        word_count = len(words)

        if word_count < 8:
            return False

        if self.is_title_or_heading(span_size, 0, font_names, clean_text):
            return False

        if any(self.is_math_font(fn) for fn in font_names):
            return False

        if self.is_math_or_formula_text(clean_text):
            return False

        if any(self.is_mono_font(fn) for fn in font_names):
            return False

        if self.is_reference_heading_or_block(clean_text):
            return False

        if self.is_toc_line(clean_text):
            return False

        if self.is_caption(clean_text):
            return False

        metadata_patterns = [
            r'submitted\s+in\s+partial\s+fulfillment',
            r'degree\s+of',
            r'department\s+of',
            r'faculty\s+of',
            r'university\s+of',
            r'bachelor\s+of\s+science',
            r'master\s+of\s+science'
        ]
        for pattern in metadata_patterns:
            if re.search(pattern, clean_text, re.IGNORECASE):
                return False

        return True
