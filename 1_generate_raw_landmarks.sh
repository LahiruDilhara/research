#!/usr/bin/env bash
# ==============================================================================
# 1_generate_raw_landmarks.sh
# Runs MediaPipe HandLandmarker frame analysis on video files in ./videos
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
echo "  [STAGE 1] GENERATING MEDIAPIPE RAW LANDMARKS"
echo "  Python Binary : $PYTHON_BIN"
echo "  Target Videos : ./videos/"
echo "========================================================================"

TARGET="${1:-$SCRIPT_DIR/videos/*.mp4}"

"$PYTHON_BIN" "$SCRIPT_DIR/annotator/mediapipe_extractor.py" "$TARGET"
echo "[STAGE 1 COMPLETED] Raw landmark CSV files generated in ./videos/"
