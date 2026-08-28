# Ensure all data processing directories exist
mkdir -p dataprocessing/1_rawCSVFiles
mkdir -p dataprocessing/2_normalized_coordinates
mkdir -p dataprocessing/3_euroFilter_coordinates
mkdir -p dataprocessing/4_filtered_coordinates_and_annotations
mkdir -p dataprocessing/5_windowed_dataset

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