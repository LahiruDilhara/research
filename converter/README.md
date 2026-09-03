# PDF Post-Processor CLI Application

A Python CLI application designed to post-process PDF documents. It filters document text elements to identify body paragraphs and sentences, excluding document titles, headings, running headers, footers, table of contents, and references. It then randomly selects words in body sentences based on a configurable probability ratio, renders those words into image snippets, redacts the original text, and inserts the rendered word images at exact bounding box locations.

## Features

- **Document Structure Filtering**: Automatically identifies and preserves document titles, chapter headings, running headers/footers, table of contents, captions, and reference lists.
- **In-Place Image Replacement**: Renders target text words into PNG images matching the original text font size, text color, and background fill.
- **Redaction & Precision Alignment**: Removes vector text from the PDF text layer and overlays images onto exact bounding box coordinates `(x0, y0, x1, y1)`.
- **Configurable CLI Flags**: Custom replacement probability, random seed for reproducibility, DPI resolution scaling, custom TrueType font loading, and minimum word length filtering.

---

## Installation & Setup

1. Make sure Python 3.12+ and `uv` are installed.
2. Install dependencies using `uv`:

```bash
uv sync
```

---

## Usage

Run the CLI tool using Python:

```bash
./.venv/bin/python main.py -i input.pdf -o output.pdf -p 0.15 -v
```

### CLI Options

| Flag | Full Option | Description | Default |
| :--- | :--- | :--- | :--- |
| `-i` | `--input` | Path to the input PDF file (required) | N/A |
| `-o` | `--output` | Path to save the processed PDF file | `<input_stem>_processed.pdf` |
| `-p` | `--probability` | Probability ratio (0.0 to 1.0) of replacing body words | `0.15` |
| | `--seed` | Random seed integer for reproducible runs | `None` |
| | `--min-word-len` | Minimum character length of words to consider | `3` |
| | `--dpi-scale` | Resolution multiplier for image rendering | `3.0` |
| | `--font-path` | Path to custom TrueType (`.ttf`/`.otf`) font file | `None` |
| `-v` | `--verbose` | Print detailed processing information | `False` |

---

## Example Execution

Generate sample test PDF:

```bash
./.venv/bin/python create_sample_pdf.py
```

Process sample PDF with 20% word replacement probability:

```bash
./.venv/bin/python main.py -i sample_test.pdf -o output_sample.pdf -p 0.2 --seed 42 -v
```
