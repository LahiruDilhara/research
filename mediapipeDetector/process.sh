# Copy CSV files to data processing directory
cp -f -r ./videos/*.raw_landmarks.* dataprocessing/1_rawCSVFiles/
cp -f -r ./videos/*.window_annotations.* dataprocessing/1_rawCSVFiles/

# Normalize landmarks
python3 datacreator/normalize_landmarks.py -i ./dataprocessing/1_rawCSVFiles/*.raw_landmarks.* -o ./dataprocessing/2_normalized_coordinates/

# Filter landmarks using 1Euro Filter
# python3 datacreator/filter_landmarks.py -i ./dataprocessing/1_rawCSVFiles/*.raw_landmarks.* -o ./dataprocessing/2_coordinationFilteredCSVFiles/