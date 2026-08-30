#!/usr/bin/env bash
# ==============================================================================
# run_realtime.sh — Launcher script for Real-Time MediaPipe Touch Detector
# ==============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PYTHON="$SCRIPT_DIR/.venv/bin/python3"

if [ -f "$VENV_PYTHON" ]; then
    PYTHON_BIN="$VENV_PYTHON"
else
    PYTHON_BIN="python3"
fi

echo "========================================================================"
echo "  STARTING REAL-TIME MEDIAPIPE TOUCH DETECTOR HUD"
echo "  Python Binary : $PYTHON_BIN"
echo "========================================================================"

exec "$PYTHON_BIN" "$SCRIPT_DIR/realtimeprocess/main_realtime_ui.py" "$@"
