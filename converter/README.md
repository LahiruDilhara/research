# PDF Post-Processor CLI Application

A Python CLI application designed to post-process PDF documents through a 4-stage protective pipeline. The tool filters document text elements to identify body paragraphs and sentences, excluding document titles, headings, running headers, footers, table of contents, math equations, and references.

- **Stage 1 (Image Replacement)**: Randomly converts selected body words into transparent PNG image snippets at exact bounding box locations.
- **Stage 2 (Homoglyph Substitution)**: Replaces selective Latin characters (`a-z` and `A-Z`) with visually identical Basic Cyrillic Unicode homoglyphs.
- **Stage 3 (Zero-Width Character Injection)**: Injects invisible zero-width Unicode characters (e.g. `\u200B`, `\u200C`, `\u200D`, `\u2060`, `\uFEFF`) using PDF text rendering mode 3 (`render_mode=3`).
- **Stage 4 (Stylized Layout Disruption)**: Overlays jumbled/scrambled text layer streams invisibly (`render_mode=3`) over the visible text baseline. Text renders 100% clean and readable on screen, but copying and pasting text from the PDF yields garbled, jumbled character sequences.

---

## Features

- **4-Stage Sequential Execution**: Run Stage 1, Stage 2, Stage 3, Stage 4, or all stages sequentially (`--stage all`).
- **Multi-Core Parallel Page Processing**: Process multiple pages simultaneously in parallel across CPU cores (`-w` / `--workers`).
- **Fine-Grained Factor Controls**: Budget maximum image replacements per page or paragraph, customize zero-width character density per word, and adjust layout disruption string length multipliers.
- **Full Word Visual Match**: Every homoglyph substitution and layout disruption retains 100% visual fidelity for human readers while tricking string matchers, tokenizers, and copy-paste text extractors.
- **Document Structure Protection**: Preserves document titles, chapter headings, running headers/footers, table of contents, math equations, captions, and reference lists.
- **Interactive Link Preservation**: Keeps all interactive PDF links (clickable TOC items, clickable citations, external URIs) 100% active and connected.

---

## Installation & Setup

1. Make sure Python 3.12+ and `uv` are installed.
2. Sync dependencies:

```bash
uv sync
```

---

## Quick Start Command

```bash
./.venv/bin/python main.py -i input_sample.pdf -o output_sample.pdf -p 0.15 --homo-prob 0.15 --zw-prob 0.15 --disrupt-prob 0.15 -w 4 -v
```

---

## Detailed CLI Options Reference

| Flag | Full Option | Description | Default |
| :--- | :--- | :--- | :--- |
| `-i` | `--input` | Path to the input PDF file (required) | N/A |
| `-o` | `--output` | Path to save the processed PDF file | `<input_stem>_processed.pdf` |
| `-w` | `--workers` | Number of parallel worker processes for page execution (`0` = auto-detect CPU cores, `1` = sequential) | `1` |
| `-p` | `--probability` | Stage 1 probability (0.0 to 1.0) of converting body words to PNG images | `0.15` |
| | `--max-images-per-page` | Stage 1 cap: Maximum word images allowed on any single page (`0` = unlimited) | `0` |
| | `--max-images-per-para` | Stage 1 cap: Maximum word images allowed in any single body paragraph (`0` = unlimited) | `0` |
| | `--homo-prob` | Stage 2 probability (0.0 to 1.0) of substituting letters with Cyrillic homoglyphs | `0.15` |
| | `--zw-prob` | Stage 3 probability (0.0 to 1.0) of injecting zero-width characters into words | `0.15` |
| | `--zw-count` | Stage 3 density: Exact number of invisible zero-width characters injected per selected word | `2` |
| | `--disrupt-prob` | Stage 4 probability (0.0 to 1.0) of overlaying scrambled text disruption streams | `0.15` |
| | `--disrupt-multiplier` | Stage 4 density: Length multiplier for scrambled invisible disruption overlay text | `1.5` |
| | `--stage` | Execution mode: `all`, `stage1`, `stage2`, `stage3`, `stage4` | `all` |
| | `--seed` | Integer random seed for reproducible output runs | `None` |
| | `--min-word-len` | Minimum character length of words to consider for processing | `3` |
| | `--dpi-scale` | Resolution DPI multiplier for text-to-image rendering quality (`2.0` = optimal 144 DPI for lightweight file size) | `2.0` |
| | `--font-path` | Optional path to custom TrueType (`.ttf`/`.otf`) font file | `None` |
| `-v` | `--verbose` | Print detailed execution log output per page | `False` |

---

## Detailed Explanation of Performance & Factor Controls

### 1. Multi-Core Parallel Page Processing (`-w` / `--workers`)
Speeds up processing of large multi-page PDF documents by distributing page processing across multiple parallel CPU worker processes.

- `-w 4` or `--workers 4`: Spawns 4 parallel worker processes to process 4 pages simultaneously.
- `-w 0` or `--workers 0`: Automatically detects the system's available CPU core count (e.g. 8 cores) and utilizes all cores in parallel.
- `-w 1`: Standard single-process sequential page processing.

### 2. Stage 1 Image Budget Controls (`--max-images-per-page` & `--max-images-per-para`)
Controls the maximum number of word image replacements allowed per page or per body paragraph. When the cap is reached, further image replacements are skipped for that section while allowing Stages 2, 3, and 4 to process normally.

- `--max-images-per-page 20`: Ensures no single page gets more than 20 image replacements.
- `--max-images-per-para 5`: Ensures no single body paragraph gets more than 5 image replacements.

### 3. Stage 3 Zero-Width Character Injection Controls (`--zw-prob` & `--zw-count`)
Controls how many invisible zero-width Unicode characters (`U+200B`, `U+200C`, `U+200D`, `U+FEFF`) are injected into each selected word.

- `--zw-prob 0.20`: Selects 20% of eligible body words for zero-width injection.
- `--zw-count 4`: Injects exactly 4 invisible zero-width characters inside each selected word.

### 4. Stage 4 Layout Disruption Controls (`--disrupt-prob` & `--disrupt-multiplier`)
Controls the length and density of the scrambled invisible overlay text layer (`render_mode=3`).

- `--disrupt-prob 0.20`: Selects 20% of eligible body words for layout disruption.
- `--disrupt-multiplier 2.0`: Generates a scrambled invisible overlay text string that is 2.0x the character length of the original word.

---

## Advanced Usage Examples

### Fast Parallel 4-Stage Protective Processing on 8 CPU Cores
```bash
./.venv/bin/python main.py \
  -i thesis.pdf \
  -o thesis_protected.pdf \
  -w 0 \
  -p 0.25 \
  --max-images-per-page 25 \
  --max-images-per-para 5 \
  --homo-prob 0.20 \
  --zw-prob 0.20 \
  --zw-count 3 \
  --disrupt-prob 0.20 \
  --disrupt-multiplier 2.0 \
  --seed 42 \
  -v
```

### Stage 1 Image Replacement Only (4 Parallel Workers, Capped at 10 Images / Page)
```bash
./.venv/bin/python main.py -i main.pdf -o output_images_only.pdf -w 4 --stage stage1 -p 0.30 --max-images-per-page 10 -v
```

### Invisible Protection Only (Zero-Width + Layout Disruption)
```bash
./.venv/bin/python main.py -i main.pdf -o output_invisible.pdf -w 4 --stage all -p 0.0 --homo-prob 0.0 --zw-prob 0.30 --zw-count 4 --disrupt-prob 0.30 --disrupt-multiplier 2.0 -v
```
