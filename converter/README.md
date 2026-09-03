# PDF Post-Processor CLI Application

A Python CLI application designed to post-process PDF documents through a multi-stage pipeline. The tool filters document text elements to identify body paragraphs and sentences, excluding document titles, headings, running headers, footers, table of contents, and references.

- **Stage 1 (Image Replacement)**: Randomly converts selected body words into transparent PNG image snippets at exact bounding box locations.
- **Stage 2 (Zero-Width Character Injection)**: Scans remaining selectable body words and injects invisible zero-width Unicode characters (e.g. `\u200B`, `\u200C`, `\u200D`, `\u2060`, `\uFEFF`) between letters to break standard tokenizers, NLP extractors, and string matching without changing the visual appearance of the text.

## Features

- **Multi-Stage Execution**: Run Stage 1 only, Stage 2 only, or both stages sequentially (`--stage all`).
- **Document Structure Protection**: Preserves document titles, chapter headings, running headers/footers, table of contents, captions, and reference lists.
- **Interactive Link Preservation**: Keeps all interactive PDF links (clickable TOC items, clickable citations, external URIs) 100% active and connected.
- **Precision Text & Image Alignment**: Uses transparent `RGBA` images and matching TrueType font baselines (`anchor="ls"`).

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
./.venv/bin/python main.py -i input.pdf -o output.pdf -p 0.15 --zw-prob 0.15 -v
```

### CLI Options

| Flag | Full Option | Description | Default |
| :--- | :--- | :--- | :--- |
| `-i` | `--input` | Path to the input PDF file (required) | N/A |
| `-o` | `--output` | Path to save the processed PDF file | `<input_stem>_processed.pdf` |
| `-p` | `--probability` | Stage 1 ratio (0.0 to 1.0) for image replacement | `0.15` |
| | `--zw-prob` | Stage 2 ratio (0.0 to 1.0) for zero-width injection | `0.15` |
| | `--stage` | Execution mode: `all`, `stage1`, or `stage2` | `all` |
| | `--seed` | Random seed integer for reproducible runs | `None` |
| | `--min-word-len` | Minimum character length of words to consider | `3` |
| | `--dpi-scale` | Resolution multiplier for image rendering | `3.0` |
| | `--font-path` | Path to custom TrueType (`.ttf`/`.otf`) font file | `None` |
| `-v` | `--verbose` | Print detailed processing information | `False` |
