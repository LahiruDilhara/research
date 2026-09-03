# PDF Post-Processor CLI Application

A Python CLI application designed to post-process PDF documents through a 3-stage protective pipeline. The tool filters document text elements to identify body paragraphs and sentences, excluding document titles, headings, running headers, footers, table of contents, and references.

- **Stage 1 (Image Replacement)**: Randomly converts selected body words into transparent PNG image snippets at exact bounding box locations.
- **Stage 2 (Homoglyph Substitution)**: Replaces selective Latin characters (`a-z` and `A-Z`) with visually identical Unicode homoglyphs from Cyrillic, Greek, Armenian, and Roman Numeral blocks. Visual text (e.g. "Lahiru") looks 100% identical to a human reader, while character codes are completely replaced.
- **Stage 3 (Zero-Width Character Injection)**: Scans remaining selectable body words and injects invisible zero-width Unicode characters (e.g. `\u200B`, `\u200C`, `\u200D`, `\u2060`, `\uFEFF`) using PDF text rendering mode 3 (`render_mode=3`).

## Features

- **3-Stage Sequential Execution**: Run Stage 1, Stage 2, Stage 3, or all stages sequentially (`--stage all`).
- **Full Word Visual Match**: Every homoglyph substitution retains 100% visual fidelity for human readers while tricking string matchers and tokenizers.
- **Document Structure Protection**: Preserves document titles, chapter headings, running headers/footers, table of contents, captions, and reference lists.
- **Interactive Link Preservation**: Keeps all interactive PDF links (clickable TOC items, clickable citations, external URIs) 100% active and connected.

---

## Installation & Setup

1. Make sure Python 3.12+ and `uv` are installed.
2. Sync dependencies:

```bash
uv sync
```

---

## Usage

Run the CLI tool using `main.py`:

```bash
./.venv/bin/python main.py -i input_sample.pdf -o output_sample.pdf -p 0.15 --homo-prob 0.15 --zw-prob 0.15 -v
```

### CLI Options

| Flag | Full Option | Description | Default |
| :--- | :--- | :--- | :--- |
| `-i` | `--input` | Path to the input PDF file (required) | N/A |
| `-o` | `--output` | Path to save the processed PDF file | `<input_stem>_processed.pdf` |
| `-p` | `--probability` | Stage 1 ratio (0.0 to 1.0) for image replacement | `0.15` |
| | `--homo-prob` | Stage 2 ratio (0.0 to 1.0) for homoglyph substitution | `0.15` |
| | `--zw-prob` | Stage 3 ratio (0.0 to 1.0) for zero-width injection | `0.15` |
| | `--stage` | Execution mode: `all`, `stage1`, `stage2`, or `stage3` | `all` |
| | `--seed` | Random seed integer for reproducible runs | `None` |
| | `--min-word-len` | Minimum character length of words to consider | `3` |
| | `--dpi-scale` | Resolution multiplier for image rendering | `3.0` |
| | `--font-path` | Path to custom TrueType (`.ttf`/`.otf`) font file | `None` |
| `-v` | `--verbose` | Print detailed processing information | `False` |
