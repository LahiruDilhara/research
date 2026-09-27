#!/usr/bin/env bash
# ==============================================================================
# run_detector.sh
# Launches the Virtual Keyboard Detector & Runtime Engine (PySide6 Fluent Window)
# ==============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DETECTOR_DIR="$SCRIPT_DIR/detector"

if [ ! -d "$DETECTOR_DIR" ]; then
    echo "[ERROR] Detector directory not found at: $DETECTOR_DIR"
    exit 1
fi

cd "$DETECTOR_DIR"

# Ensure .env exists from .env.example if missing
if [ ! -f ".env" ] && [ -f ".env.example" ]; then
    echo "[INFO] Initializing .env from .env.example..."
    cp ".env.example" ".env"
fi

# Synchronize latest trained best model weight if present in modelBenchmark/best_models
BEST_WEIGHT="$SCRIPT_DIR/../modelBenchmark/best_models/best_finger_touch_lstm.pth"
if [ -f "$BEST_WEIGHT" ]; then
    mkdir -p "$DETECTOR_DIR/ai_model_plugins/lstm_all_combined"
    cp -f "$BEST_WEIGHT" "$DETECTOR_DIR/ai_model_plugins/lstm_all_combined/weights.pth" 2>/dev/null || true
fi

# Detect python runtime (prefer existing virtual environment, fallback to uv / system python)
if [ -f "$DETECTOR_DIR/.venv/bin/python3" ]; then
    PYTHON_EXEC="$DETECTOR_DIR/.venv/bin/python3"
elif command -v uv >/dev/null 2>&1; then
    PYTHON_EXEC="uv run python3"
else
    PYTHON_EXEC="python3"
fi

echo "========================================================================"
echo "  Starting Virtual Keyboard Detector & Runtime Engine..."
echo "  Directory : $DETECTOR_DIR"
echo "  Runtime   : $PYTHON_EXEC"
echo "========================================================================"

exec $PYTHON_EXEC main.py "$@"
