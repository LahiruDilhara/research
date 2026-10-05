#!/bin/bash
set -e

# Change directory to script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== Compiling LaTeX Beamer Presentation ==="
mkdir -p out

# Compile with pdflatex
pdflatex -interaction=nonstopmode -output-directory=out presentation.tex
pdflatex -interaction=nonstopmode -output-directory=out presentation.tex

# Copy to local presentation.pdf
cp out/presentation.pdf ./presentation.pdf

# Copy to root output directory
mkdir -p "$SCRIPT_DIR/../output"
cp out/presentation.pdf "$SCRIPT_DIR/../output/presentation.pdf"

# Copy full bundle to output/presentation/ if compiling from project root presentation/
if [ "$SCRIPT_DIR" != "$(cd "$SCRIPT_DIR/../output/presentation" 2>/dev/null && pwd)" ]; then
  mkdir -p "$SCRIPT_DIR/../output/presentation"
  cp -rf "$SCRIPT_DIR"/* "$SCRIPT_DIR/../output/presentation/" 2>/dev/null || true
fi

echo "=== Presentation Compiled Successfully ==="
echo "Artifacts placed at:"
echo "  1) $SCRIPT_DIR/presentation.pdf"
echo "  2) $SCRIPT_DIR/../output/presentation.pdf"
echo "  3) $SCRIPT_DIR/../output/presentation/presentation.pdf"
