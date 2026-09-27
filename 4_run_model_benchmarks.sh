#!/usr/bin/env bash
# ==============================================================================
# 4_run_model_benchmarks.sh
# Master runner for deep learning model benchmarks across 5 model families.
# Evaluates models and saves best performing .pth weights to ./modelBenchmark/best_models/
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
echo "  [STAGE 4] EXECUTING DEEP LEARNING MODEL BENCHMARK SUITE"
echo "  Python Binary : $PYTHON_BIN"
echo "  Working Dir   : ./modelBenchmark/"
echo "  Data Source   : ./modelBenchmark/data/train_dataset.csv & test_dataset.csv"
echo "  Best Output   : ./modelBenchmark/best_models/"
echo "========================================================================"

cd "$SCRIPT_DIR/modelBenchmark"
"$PYTHON_BIN" benchmark_engine.py "$@"

# Copy top performing LSTM touch model to best_models if generated
if [ -f "$SCRIPT_DIR/modelBenchmark/weights/LSTM_All_Combined_cfg01.pth" ]; then
    cp -f "$SCRIPT_DIR/modelBenchmark/weights/LSTM_All_Combined_cfg01.pth" "$SCRIPT_DIR/modelBenchmark/best_models/best_finger_touch_lstm.pth"
    echo "[STAGE 4 INFO] Synchronized best model weight -> ./modelBenchmark/best_models/best_finger_touch_lstm.pth"
fi

echo "========================================================================"
echo "  [STAGE 4 COMPLETED] Evaluation results stored in ./modelBenchmark/results/"
echo "========================================================================"
