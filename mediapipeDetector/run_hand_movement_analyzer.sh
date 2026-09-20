#!/usr/bin/env bash
# ==============================================================================
# run_hand_movement_analyzer.sh — Launcher for Hand Movement & Velocity Analyzer
# ==============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PYTHON="$SCRIPT_DIR/.venv/bin/python3"

if [ -f "$VENV_PYTHON" ]; then
    PYTHON_BIN="$VENV_PYTHON"
else
    PYTHON_BIN="python3"
fi

echo "========================================================================"
echo "  STARTING HAND MOVEMENT & VELOCITY THRESHOLD ANALYZER (12 FPS)"
echo "  Python Binary : $PYTHON_BIN"
echo "========================================================================"

exec "$PYTHON_BIN" "$SCRIPT_DIR/datacreator/hand_movement_analyzer_ui.py" "$@"
