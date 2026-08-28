# Ensure all data processing directories exist
mkdir -p dataprocessing/1_rawCSVFiles
mkdir -p dataprocessing/2_normalized_coordinates
mkdir -p dataprocessing/3_euroFilter_coordinates
mkdir -p dataprocessing/4_filtered_coordinates_and_annotations
mkdir -p dataprocessing/5_windowed_dataset
mkdir -p dataprocessing/6_merged_windowed_dataset
mkdir -p dataprocessing/7_dataset_with_velocities
mkdir -p dataprocessing/8_cleaned_dataset

# Copy CSV files to data processing directory
cp -f -r ./videos/*.raw_landmarks.* dataprocessing/1_rawCSVFiles/
cp -f -r ./videos/*.window_annotations.* dataprocessing/1_rawCSVFiles/

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