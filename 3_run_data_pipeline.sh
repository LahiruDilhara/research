#!/usr/bin/env bash
# ==============================================================================
# 3_run_data_pipeline.sh
# End-to-end data filtration, 12 FPS windowing, normalization & train/test split.
# Outputs train_dataset.csv and test_dataset.csv directly to modelBenchmark/data/
# ==============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="python3"

if [ -f "$SCRIPT_DIR/virtualKeyboardSetup/detector/.venv/bin/python3" ]; then
    PYTHON_BIN="$SCRIPT_DIR/virtualKeyboardSetup/detector/.venv/bin/python3"
elif [ -f "$SCRIPT_DIR/mediapipeDetector/.venv/bin/python3" ]; then
    PYTHON_BIN="$SCRIPT_DIR/mediapipeDetector/.venv/bin/python3"
fi

HAND_MOVEMENT_THRESHOLD=0.155
WORK_DIR="$SCRIPT_DIR/dataPipeline/build_cache"

echo "========================================================================"
echo "  [STAGE 3] EXECUTING FEATURE PIPELINE & DATASET FILTRATION"
echo "  Python Binary           : $PYTHON_BIN"
echo "  Hand Movement Threshold : $HAND_MOVEMENT_THRESHOLD L_hand"
echo "  Source Directory        : ./videos/"
echo "  Output Directory        : ./dataPipeline/output/ & ./modelBenchmark/data/"
echo "========================================================================"

rm -rf "$WORK_DIR"
mkdir -p "$WORK_DIR/1_rawCSVFiles"
mkdir -p "$WORK_DIR/2_normalized_coordinates"
mkdir -p "$WORK_DIR/3_filtered_coordinates"
mkdir -p "$WORK_DIR/4_filtered_coordinates_and_annotations"
mkdir -p "$WORK_DIR/5_windowed_dataset"
mkdir -p "$WORK_DIR/6_merged_windowed_dataset"
mkdir -p "$WORK_DIR/7_hand_movement_filtered"
mkdir -p "$WORK_DIR/8_dataset_with_velocities"
mkdir -p "$WORK_DIR/9_cleaned_dataset"
mkdir -p "$WORK_DIR/10_quality_filtered_dataset"
mkdir -p "$WORK_DIR/11_per_finger_dataset"
mkdir -p "$WORK_DIR/12_split_touch_dataset"
mkdir -p "$WORK_DIR/13_train_test_split"
mkdir -p "$WORK_DIR/summaries"
mkdir -p "$SCRIPT_DIR/dataPipeline/output"
mkdir -p "$SCRIPT_DIR/modelBenchmark/data"

echo "[Step 1] Ingesting CSV files from ./videos/..."
cp -f "$SCRIPT_DIR/videos/"*.raw_landmarks.* "$WORK_DIR/1_rawCSVFiles/"
cp -f "$SCRIPT_DIR/videos/"*.window_annotations.* "$WORK_DIR/1_rawCSVFiles/"

echo "[Step 2] Applying unitless hand-length scale normalization (L_hand)..."
"$PYTHON_BIN" "$SCRIPT_DIR/dataPipeline/src/normalize_landmarks.py" \
    -i "$WORK_DIR/1_rawCSVFiles/"*.raw_landmarks.* \
    -o "$WORK_DIR/2_normalized_coordinates/"

echo "[Step 3] Coordinate passthrough filtering..."
"$PYTHON_BIN" "$SCRIPT_DIR/dataPipeline/src/filter_landmarks.py" --mode none \
    -i "$WORK_DIR/2_normalized_coordinates/"*.normalize_landmarks.* \
    -o "$WORK_DIR/3_filtered_coordinates/"

cp -f "$WORK_DIR/1_rawCSVFiles/"*.window_annotations.* "$WORK_DIR/4_filtered_coordinates_and_annotations/"
cp -f "$WORK_DIR/3_filtered_coordinates/"*.filtered_landmarks.* "$WORK_DIR/4_filtered_coordinates_and_annotations/"

echo "[Step 4] Assembling 5-frame temporal sliding windows (2-frame overlap)..."
"$PYTHON_BIN" "$SCRIPT_DIR/dataPipeline/src/create_windows.py" \
    -i "$WORK_DIR/4_filtered_coordinates_and_annotations/" \
    -o "$WORK_DIR/5_windowed_dataset/"

echo "[Step 5] Merging window chunks into single unified dataset..."
"$PYTHON_BIN" "$SCRIPT_DIR/dataPipeline/src/merge_windows.py" \
    -i "$WORK_DIR/5_windowed_dataset/" \
    -o "$WORK_DIR/6_merged_windowed_dataset/all_windowed_dataset.csv"

echo "[Step 6] Filtering whole-hand transit movement windows..."
"$PYTHON_BIN" "$SCRIPT_DIR/dataPipeline/src/filter_hand_movement.py" \
    -i "$WORK_DIR/6_merged_windowed_dataset/all_windowed_dataset.csv" \
    -o "$WORK_DIR/7_hand_movement_filtered/hand_movement_filtered_dataset.csv" \
    --raw-dir "$WORK_DIR/1_rawCSVFiles/" \
    --threshold "$HAND_MOVEMENT_THRESHOLD"

echo "[Step 7] Calculating 4-step landmark velocities and speeds (84-D vector)..."
"$PYTHON_BIN" "$SCRIPT_DIR/dataPipeline/src/calculate_velocities.py" \
    -i "$WORK_DIR/7_hand_movement_filtered/hand_movement_filtered_dataset.csv" \
    -o "$WORK_DIR/8_dataset_with_velocities/all_windowed_dataset_velocities.csv"

echo "[Step 8] Filtering zero velocity, invisible hands, and desync records..."
"$PYTHON_BIN" "$SCRIPT_DIR/dataPipeline/src/filter_dataset.py" \
    -i "$WORK_DIR/8_dataset_with_velocities/all_windowed_dataset_velocities.csv" \
    -o "$WORK_DIR/9_cleaned_dataset/cleaned_dataset.csv" \
    --remove-zero-vel-touch --remove-out-of-sync --remove-hand-invisible

echo "[Step 9] Quality & confidence filtering (MediaPipe tracking score thresholds)..."
"$PYTHON_BIN" "$SCRIPT_DIR/dataPipeline/src/filter_window_quality.py" \
    -i "$WORK_DIR/9_cleaned_dataset/cleaned_dataset.csv" \
    -o "$WORK_DIR/10_quality_filtered_dataset/quality_cleaned_dataset.csv" \
    --min-avg-score 0.65 --min-frame-score 0.45 --max-score-drop 0.35

echo "[Step 10] Unrolling sequences into per-finger touch records..."
"$PYTHON_BIN" "$SCRIPT_DIR/dataPipeline/src/split_fingers.py" \
    -i "$WORK_DIR/10_quality_filtered_dataset/quality_cleaned_dataset.csv" \
    -o "$WORK_DIR/11_per_finger_dataset/per_finger_dataset.csv"

echo "[Step 11] Separating touch vs. untouch candidate pools..."
"$PYTHON_BIN" "$SCRIPT_DIR/dataPipeline/src/split_touch.py" \
    -i "$WORK_DIR/11_per_finger_dataset/per_finger_dataset.csv" \
    -o "$WORK_DIR/12_split_touch_dataset/"

echo "[Step 12] Performing stratified train/test split (no video leak)..."
"$PYTHON_BIN" "$SCRIPT_DIR/dataPipeline/src/create_train_test_split.py" \
    --touch-in "$WORK_DIR/12_split_touch_dataset/touch_dataset.csv" \
    --untouch-in "$WORK_DIR/12_split_touch_dataset/untouch_dataset.csv" \
    --train-out "$WORK_DIR/13_train_test_split/training_dataset.csv" \
    --test-out "$WORK_DIR/13_train_test_split/testing_dataset.csv" \
    --touch-test-pct 20 --untouch-train-ratio-pct 130 --untouch-test-ratio-pct 100 --seed 50 --no-video-leak

echo "[Step 13] Exporting finalized train and test datasets..."
cp -f "$WORK_DIR/13_train_test_split/training_dataset.csv" "$SCRIPT_DIR/dataPipeline/output/train_dataset.csv"
cp -f "$WORK_DIR/13_train_test_split/testing_dataset.csv" "$SCRIPT_DIR/dataPipeline/output/test_dataset.csv"

# Sync with modelBenchmark/data/
cp -f "$SCRIPT_DIR/dataPipeline/output/train_dataset.csv" "$SCRIPT_DIR/modelBenchmark/data/train_dataset.csv"
cp -f "$SCRIPT_DIR/dataPipeline/output/test_dataset.csv" "$SCRIPT_DIR/modelBenchmark/data/test_dataset.csv"

echo "========================================================================"
echo "  [STAGE 3 COMPLETED SUCCESSFULLY]"
echo "  Output generated at:"
echo "    -> ./dataPipeline/output/train_dataset.csv"
echo "    -> ./dataPipeline/output/test_dataset.csv"
echo "  Synchronized with ./modelBenchmark/data/ for immediate training."
echo "========================================================================"
