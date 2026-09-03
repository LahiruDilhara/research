import re
from statistics import median
from typing import Dict, List, Any, Tuple


class TextClassifier:
    """
    Classifies text blocks and spans in PDF documents to isolate body paragraphs
    and exclude titles, headings, TOC, references, headers, footers, and citations.
    """

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

    def is_header_or_footer(self, bbox: Tuple[float, float, float, float], page_height: float, text: str) -> bool:
        """
        Detects top running headers, bottom running footers, and page numbers.
        """
        y0, y1 = bbox[1], bbox[3]
        
        # Check top 8% of page for running headers
        if y0 < (page_height * 0.08):
            return True
            
        # Check bottom 8% of page for running footers
        if y1 > (page_height * 0.92):
            return True
            
        # Check for standalone page number patterns e.g. "Page 5", "- 5 -", "5"
        clean_text = text.strip()
        if re.match(r'^(?:page\s+)?\d+(?:\s+of\s+\d+)?$', clean_text, re.IGNORECASE):
            return True
            
        return False

    def is_title_or_heading(self, span_size: float, span_flags: int, text: str) -> bool:
        """
        Detects document titles, chapter titles, and section headings.
        """
        clean_text = text.strip()
        if not clean_text:
            return False

        # Font size significantly larger than body font (20% larger)
        if span_size >= (self.median_font_size * 1.20):
            return True

        # Heading section pattern matching
        heading_patterns = [
            r'^(?:chapter|section|part)\s+\d+',
            r'^\d+(?:\.\d+)*\s+[A-Z]',
            r'^(?:abstract|introduction|background|methods|results|discussion|conclusion|acknowledgments)$'
        ]
        for pattern in heading_patterns:
            if re.match(pattern, clean_text, re.IGNORECASE):
                return True

        # Short bold line without ending punctuation
        is_bold = bool(span_flags & 2) or ("bold" in clean_text.lower())
        if is_bold and len(clean_text) < 60 and not clean_text.endswith(('.', '?', '!')):
            return True

        return False

    def is_toc_line(self, text: str) -> bool:
        """
        Detects Table of Contents lines (dot leaders, page numbers).
        """
        # Checks for dot leaders e.g. "Chapter 1 ..... 12"
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

        # Checks for standard citation bracket at the start of a block e.g. "[1]", "[23]"
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

    def filter_inline_citations(self, text: str) -> List[Tuple[int, int]]:
        """
        Returns list of (start_idx, end_idx) character spans that represent inline citations
        such as [1], [2-5], or (Smith et al., 2020) so they can be skipped.
        """
        spans = []
        # Bracket citations e.g. [1], [12, 14], [3-7]
        for m in re.finditer(r'\[\d+(?:\s*,\s*\d+|-|\d+)*\]', text):
            spans.append(m.span())

        # Author-year citations e.g. (Smith, 2020) or (Jones et al., 2019)
        for m in re.finditer(r'\([A-Z][a-zA-Z\s,]+(?:et al\.?)?,?\s*\d{4}\)', text):
            spans.append(m.span())

        return spans
