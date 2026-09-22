#!/usr/bin/env bash
# ==============================================================================
# run_realtime.sh: Launcher script for Real-Time MediaPipe Touch Detector
# ==============================================================================

# ==============================================================================
# Configuration Parameters (Exact process.sh Filtration Values)
# ==============================================================================
DEFAULT_MODEL="LSTM_All_Combined"
HAND_MOVEMENT_THRESHOLD=0.155
MIN_AVG_SCORE=0.65
MIN_FRAME_SCORE=0.45
MAX_SCORE_DROP=0.35

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PYTHON="$SCRIPT_DIR/.venv/bin/python3"

if [ -f "$VENV_PYTHON" ]; then
    PYTHON_BIN="$VENV_PYTHON"
else
    PYTHON_BIN="python3"
fi

echo "========================================================================"
echo "  STARTING REAL-TIME MEDIAPIPE TOUCH DETECTOR HUD"
echo "  Python Binary           : $PYTHON_BIN"
echo "  Defined Model           : $DEFAULT_MODEL"
echo "  Hand Movement Threshold : $HAND_MOVEMENT_THRESHOLD L_hand (Step 7 Filter)"
echo "  Min Average Hand Score  : $MIN_AVG_SCORE (Step 10 Quality Filter)"
echo "  Min Per-Frame Hand Score: $MIN_FRAME_SCORE (Step 10 Quality Filter)"
echo "  Max Score Fluctuation   : $MAX_SCORE_DROP (Step 10 Quality Filter)"
echo "  Sampling Standard       : 12 FPS Sub-sampling Pipeline"
echo "  5-Queue Architecture    : Active Multi-Queue Per-Finger Evaluation"
echo "========================================================================"

exec "$PYTHON_BIN" "$SCRIPT_DIR/realtimeprocess/main_realtime_ui.py" \
    --model "$DEFAULT_MODEL" \
    --threshold "$HAND_MOVEMENT_THRESHOLD" \
    --min-avg-score "$MIN_AVG_SCORE" \
    --min-frame-score "$MIN_FRAME_SCORE" \
    --max-score-drop "$MAX_SCORE_DROP" \
    "$@"
