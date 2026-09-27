#!/usr/bin/env bash
# ==============================================================================
# compile_thesis.sh
# Compiles LaTeX thesis to PDF via latexmk and optionally builds DOCX.
# Places all final documents (main.pdf, thesis.docx) into ./output/
# ==============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

OUTPUT_DIR="$SCRIPT_DIR/output"
OUT_BUILD_DIR="$SCRIPT_DIR/out"

mkdir -p "$OUTPUT_DIR"
mkdir -p "$OUT_BUILD_DIR"

BUILD_DOCX=false
CLEAN=false

for arg in "$@"; do
    case "$arg" in
        --docx)
            BUILD_DOCX=true
            ;;
        --clean)
            CLEAN=true
            ;;
        *)
            ;;
    esac
done

if [ "$CLEAN" = true ]; then
    echo "Cleaning LaTeX intermediate build files..."
    latexmk -c -outdir=out main.tex || true
fi

echo "========================================================================"
echo "  [1/2] COMPILING THESIS PDF (latexmk)"
echo "========================================================================"
latexmk -pdf -outdir=out main.tex

# Copy final PDF to ./output/
if [ -f "out/main.pdf" ]; then
    cp -f "out/main.pdf" "$OUTPUT_DIR/main.pdf"
    echo "✓ Final Thesis PDF -> ./output/main.pdf"
fi

if [ "$BUILD_DOCX" = true ]; then
    echo ""
    echo "========================================================================"
    echo "  [2/2] COMPILING THESIS DOCX (convert_thesis_to_docx.py)"
    echo "========================================================================"
    python3 "$SCRIPT_DIR/converter/convert_thesis_to_docx.py"
    echo "✓ Final Thesis DOCX -> ./output/thesis.docx"
fi

echo ""
echo "========================================================================"
echo "  [BUILD COMPLETED]"
echo "  Generated files in ./output/:"
ls -lh "$OUTPUT_DIR"
echo "========================================================================"
