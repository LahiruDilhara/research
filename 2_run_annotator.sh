#!/usr/bin/env bash
# ==============================================================================
# 2_run_annotator.sh
# Launches the CustomTkinter 12 FPS Hand Landmark & Touch Annotator GUI
# ==============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="python3"

if [ -f "$SCRIPT_DIR/virtualKeyboardSetup/detector/.venv/bin/python3" ]; then
    PYTHON_BIN="$SCRIPT_DIR/virtualKeyboardSetup/detector/.venv/bin/python3"
elif [ -f "$SCRIPT_DIR/mediapipeDetector/.venv/bin/python3" ]; then
    PYTHON_BIN="$SCRIPT_DIR/mediapipeDetector/.venv/bin/python3"
fi

echo "========================================================================"
echo "  [STAGE 2] LAUNCHING TOUCH DATASET ANNOTATOR GUI"
echo "  Python Binary : $PYTHON_BIN"
echo "  Annotator Dir : ./annotator/"
echo "========================================================================"

cd "$SCRIPT_DIR/annotator"
"$PYTHON_BIN" app.py "$@"
