#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

mkdir -p out
mkdir -p output

echo "==> Compiling IEEE Research Paper with latexmk..."
latexmk -pdf -outdir=out -interaction=nonstopmode main.tex

if [ -f "out/main.pdf" ]; then
    cp -f out/main.pdf output/research_paper.pdf
    cp -f out/main.pdf "$SCRIPT_DIR/../output/research_paper.pdf"
    echo "==> Compilation successful: output/research_paper.pdf and ../output/research_paper.pdf"
    ls -lh "$SCRIPT_DIR/../output/research_paper.pdf"
else
    echo "==> Error: out/main.pdf not found!"
    exit 1
fi
