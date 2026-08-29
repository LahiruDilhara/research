rm -r dataprocessing
rm -r training_testing_data

# Ensure all data processing directories exist
mkdir -p dataprocessing/1_rawCSVFiles
mkdir -p dataprocessing/2_normalized_coordinates
mkdir -p dataprocessing/3_euroFilter_coordinates
mkdir -p dataprocessing/4_filtered_coordinates_and_annotations
mkdir -p dataprocessing/5_windowed_dataset
mkdir -p dataprocessing/6_merged_windowed_dataset
mkdir -p dataprocessing/7_dataset_with_velocities
mkdir -p dataprocessing/8_cleaned_dataset
mkdir -p dataprocessing/9_per_finger_dataset
mkdir -p dataprocessing/10_split_touch_dataset
mkdir -p dataprocessing/11_train_test_split

# Copy CSV files to data processing directory
cp -f -r ./dataset/*.raw_landmarks.* dataprocessing/1_rawCSVFiles/
cp -f -r ./dataset/*.window_annotations.* dataprocessing/1_rawCSVFiles/

# Normalize landmarks
python3 datacreator/normalize_landmarks.py -i ./dataprocessing/1_rawCSVFiles/*.raw_landmarks.* -o ./dataprocessing/2_normalized_coordinates/

# Filter landmarks using 1Euro Filter
python3 datacreator/filter_landmarks.py -min 1.5 -beta 1.0 -d 1.0 -i ./dataprocessing/2_normalized_coordinates/*.normalize_landmarks.* -o ./dataprocessing/3_euroFilter_coordinates/

cp -f -r ./dataprocessing/1_rawCSVFiles/*.window_annotations.* ./dataprocessing/4_filtered_coordinates_and_annotations/
cp -f -r ./dataprocessing/3_euroFilter_coordinates/*.filtered_landmarks.* ./dataprocessing/4_filtered_coordinates_and_annotations/

# Create windowed sequence datasets
python3 datacreator/create_windows.py -i ./dataprocessing/4_filtered_coordinates_and_annotations/ -o ./dataprocessing/5_windowed_dataset/

# Merge all windowed datasets into a single combined CSV dataset
python3 datacreator/merge_windows.py -i ./dataprocessing/5_windowed_dataset/ -o ./dataprocessing/6_merged_windowed_dataset/all_windowed_dataset.csv

# Calculate 4-step velocities (vx, vy) & 2D speeds sqrt(vx^2 + vy^2) for all landmarks
python3 datacreator/calculate_velocities.py -i ./dataprocessing/6_merged_windowed_dataset/all_windowed_dataset.csv -o ./dataprocessing/7_dataset_with_velocities/all_windowed_dataset_velocities.csv

# Filter and clean windowed dataset based on configurable flags
python3 datacreator/filter_dataset.py -i ./dataprocessing/7_dataset_with_velocities/all_windowed_dataset_velocities.csv -o ./dataprocessing/8_cleaned_dataset/cleaned_dataset.csv --remove-zero-vel-touch --remove-out-of-sync --remove-hand-invisible

# Unroll sequence windows into per-finger dataset records (thumb, index, middle, ring, pinky)
python3 datacreator/split_fingers.py -i ./dataprocessing/8_cleaned_dataset/cleaned_dataset.csv -o ./dataprocessing/9_per_finger_dataset/per_finger_dataset.csv

# Separate per-finger dataset into touch_dataset.csv and untouch_dataset.csv
python3 datacreator/split_touch.py -i ./dataprocessing/9_per_finger_dataset/per_finger_dataset.csv -o ./dataprocessing/10_split_touch_dataset/

# Create balanced training and testing datasets
python3 datacreator/create_train_test_split.py --touch-in ./dataprocessing/10_split_touch_dataset/touch_dataset.csv --untouch-in ./dataprocessing/10_split_touch_dataset/untouch_dataset.csv --train-out ./dataprocessing/11_train_test_split/training_dataset.csv --test-out ./dataprocessing/11_train_test_split/testing_dataset.csv --touch-test-pct 15 --untouch-train-ratio-pct 100 --untouch-test-ratio-pct 100 --seed 50 --no-video-leak

# Copy training and testing data to root
mkdir training_testing_data
cp -f ./dataprocessing/11_train_test_split/training_dataset.csv training_testing_data/train_dataset.csv
cp -f ./dataprocessing/11_train_test_split/testing_dataset.csv training_testing_data/test_dataset.csv