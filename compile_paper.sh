#!/usr/bin/env bash
# ==============================================================================
# compile_paper.sh
# Compiles the IEEE conference research paper to PDF via latexmk.
# Places final document directly into ./output/research_paper.pdf
# ==============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PAPER_DIR="$SCRIPT_DIR/research_paper"
OUTPUT_DIR="$SCRIPT_DIR/output"

mkdir -p "$OUTPUT_DIR"
mkdir -p "$PAPER_DIR/out"

CLEAN=false

for arg in "$@"; do
    case "$arg" in
        --clean)
            CLEAN=true
            ;;
        *)
            ;;
    esac
done

cd "$PAPER_DIR"

if [ "$CLEAN" = true ]; then
    echo "Cleaning intermediate research paper build files..."
    latexmk -c -outdir=out main.tex || true
fi

echo "========================================================================"
echo "  COMPILING IEEE RESEARCH PAPER (latexmk)"
echo "  Source Dir : $PAPER_DIR"
echo "  Target PDF : $OUTPUT_DIR/research_paper.pdf"
echo "========================================================================"

latexmk -pdf -outdir=out -interaction=nonstopmode main.tex

if [ -f "out/main.pdf" ]; then
    cp -f "out/main.pdf" "$OUTPUT_DIR/research_paper.pdf"
    mkdir -p "$PAPER_DIR/output"
    cp -f "out/main.pdf" "$PAPER_DIR/output/research_paper.pdf"
    echo ""
    echo "✓ Final Research Paper PDF -> $OUTPUT_DIR/research_paper.pdf"
    echo "========================================================================"
    ls -lh "$OUTPUT_DIR/research_paper.pdf"
    echo "========================================================================"
else
    echo "==> Error: out/main.pdf not found!"
    exit 1
fi
