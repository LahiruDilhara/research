# On These

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
python3 datacreator/filter_landmarks.py -min 0.05 -beta 1.0 -d 1.0 -i ./dataprocessing/2_normalized_coordinates/*.normalize_landmarks.* -o ./dataprocessing/3_euroFilter_coordinates/

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


# Output

======================================================================
  ALL ARCHITECTURES COMPLETE
======================================================================
  ✓ DONE      LSTM  (vel only      4×8 )  (0m 14s)
  ✓ DONE      LSTM  (coords        5×8 )  (0m 22s)
  ✓ DONE      LSTM  (combined      4×16)  (0m 14s)
  ✓ DONE      LSTM  (vel + speed   4×12)  (0m 9s)
  ✓ DONE      LSTM  (all 9 joints  4×18)  (0m 10s)
  ✓ DONE      BiLSTM (combined     4×16)  (0m 15s)
  ✓ DONE      CNN1D (combined      4×16)  (0m 11s)
  ✓ DONE      ResNet1D (combined   4×16)  (0m 12s)
  ✓ DONE      Attention (combined  4×16)  (0m 14s)
  ✓ DONE      TCN   (combined      4×16)  (0m 20s)

  Total time: 2m 26s
======================================================================

  Generating results comparison report...


──────────────────────────────────────────────────────────────────────────────────────────
  ARCHITECTURE RANKING  (by best single config accuracy)
──────────────────────────────────────────────────────────────────────────────────────────
  #1  LSTM_Combined            87.39%  █████████████
  #2  LSTM_All_Joints_Vel      87.24%  █████████████
  #3  LSTM_Vel_Speed           87.09%  █████████████
  #4  LSTM_Velocities          86.94%  █████████████
  #5  TCN                      86.79%  █████████████
  #6  BiLSTM                   86.79%  █████████████
  #7  Attention                86.34%  ████████████
  #8  ResNet1D                 86.34%  ████████████
  #9  CNN1D                    86.19%  ████████████
  #10  LSTM_Coords              80.03%  ████████████
──────────────────────────────────────────────────────────────────────────────────────────

──────────────────────────────────────────────────────────────────────────────────────────
  BEST CONFIG PER ARCHITECTURE
──────────────────────────────────────────────────────────────────────────────────────────
  Attention               cfg01  Acc= 86.34%  F1=0.8542  |  embed_dim=32.0  num_heads=4.0  dropout=0.2  lr=0.001  batch_size=32
  BiLSTM                  cfg01  Acc= 86.79%  F1=0.8567  |  hidden=32.0  layers=2.0  dropout=0.2  lr=0.001  batch_size=32
  CNN1D                   cfg01  Acc= 86.19%  F1=0.8571  |  conv_ch=32.0  fc_hid=32.0  dropout=0.2  lr=0.001  batch_size=32
  LSTM_All_Joints_Vel     cfg01  Acc= 87.24%  F1=0.8644  |  hidden=32.0  layers=2.0  dropout=0.2  lr=0.001  batch_size=32
  LSTM_Combined           cfg01  Acc= 87.39%  F1=0.8659  |  hidden=32.0  layers=2.0  dropout=0.2  lr=0.001  batch_size=32
  LSTM_Coords             cfg01  Acc= 80.03%  F1=0.7919  |  hidden=32.0  layers=2.0  dropout=0.2  lr=0.001  batch_size=32
  LSTM_Vel_Speed          cfg01  Acc= 87.09%  F1=0.8571  |  hidden=32.0  layers=2.0  dropout=0.2  lr=0.001  batch_size=32
  LSTM_Velocities         cfg01  Acc= 86.94%  F1=0.8291  |  hidden=32.0  layers=2.0  dropout=0.2  lr=0.001  batch_size=32
  ResNet1D                cfg01  Acc= 86.34%  F1=0.8466  |  hidden_dim=32.0  dropout=0.2  lr=0.001  batch_size=32
  TCN                     cfg01  Acc= 86.79%  F1=0.8576  |  tcn_channels=32.0  num_levels=2.0  dropout=0.2  lr=0.001  batch_size=32
──────────────────────────────────────────────────────────────────────────────────────────

==========================================================================================
  TOP 5 CONFIGURATIONS OVERALL
==========================================================================================
  #1  LSTM_Combined           cfg01  →  Acc= 87.39%  |  F1=0.8659  |  Prec=0.8793  Rec=0.8529  |  11.5s
  #2  LSTM_All_Joints_Vel     cfg01  →  Acc= 87.24%  |  F1=0.8644  |  Prec=0.8580  Rec=0.8709  |  7.9s
  #3  LSTM_Vel_Speed          cfg01  →  Acc= 87.09%  |  F1=0.8571  |  Prec=0.8677  Rec=0.8468  |  6.3s
  #4  LSTM_Velocities         cfg01  →  Acc= 86.94%  |  F1=0.8291  |  Prec=0.8763  Rec=0.7868  |  11.7s
  #5  TCN                     cfg01  →  Acc= 86.79%  |  F1=0.8576  |  Prec=0.8654  Rec=0.8498  |  17.4s
==========================================================================================

==========================================================================================
  FULL BENCHMARK RESULTS  (sorted by Test Accuracy)
==========================================================================================
 rank                arch  config_id best_test_acc f1_touch precision_touch recall_touch train_time_s
    1       LSTM_Combined          1        87.39%   0.8659          0.8793       0.8529        11.5s
    2 LSTM_All_Joints_Vel          1        87.24%   0.8644          0.8580       0.8709         7.9s
    3      LSTM_Vel_Speed          1        87.09%   0.8571          0.8677       0.8468         6.3s
    4     LSTM_Velocities          1        86.94%   0.8291          0.8763       0.7868        11.7s
    5                 TCN          1        86.79%   0.8576          0.8654       0.8498        17.4s
    6              BiLSTM          1        86.79%   0.8567          0.8700       0.8438        12.1s
    7           Attention          1        86.34%   0.8542          0.8466       0.8619        11.5s
    8            ResNet1D          1        86.34%   0.8466          0.8652       0.8288         9.4s
    9               CNN1D          1        86.19%   0.8571          0.8584       0.8559         8.5s
   10         LSTM_Coords          1        80.03%   0.7919          0.8268       0.7598        19.4s
==========================================================================================




















# Iteration 2

## Parameters
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
python3 datacreator/filter_landmarks.py -min 0.05 -beta 1.0 -d 1.0 -i ./dataprocessing/2_normalized_coordinates/*.normalize_landmarks.* -o ./dataprocessing/3_euroFilter_coordinates/

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


## Result 

  ✓ DONE  —  TCN   (combined      4×16)  (0m 14s)


======================================================================
  ALL ARCHITECTURES COMPLETE
======================================================================
  ✓ DONE      LSTM  (vel only      4×8 )  (0m 10s)
  ✓ DONE      LSTM  (coords        5×8 )  (0m 19s)
  ✓ DONE      LSTM  (combined      4×16)  (0m 19s)
  ✓ DONE      LSTM  (vel + speed   4×12)  (0m 11s)
  ✓ DONE      LSTM  (all 9 joints  4×18)  (0m 15s)
  ✓ DONE      BiLSTM (combined     4×16)  (0m 16s)
  ✓ DONE      CNN1D (combined      4×16)  (0m 15s)
  ✓ DONE      ResNet1D (combined   4×16)  (0m 25s)
  ✓ DONE      Attention (combined  4×16)  (0m 17s)
  ✓ DONE      TCN   (combined      4×16)  (0m 14s)

  Total time: 2m 46s
======================================================================

  Generating results comparison report...


──────────────────────────────────────────────────────────────────────────────────────────
  ARCHITECTURE RANKING  (by best single config accuracy)
──────────────────────────────────────────────────────────────────────────────────────────
  #1  LSTM_Combined            84.53%  ████████████
  #2  ResNet1D                 84.38%  ████████████
  #3  LSTM_Vel_Speed           83.93%  ████████████
  #4  LSTM_All_Joints_Vel      83.48%  ████████████
  #5  TCN                      83.48%  ████████████
  #6  CNN1D                    83.18%  ████████████
  #7  BiLSTM                   82.43%  ████████████
  #8  LSTM_Velocities          82.13%  ████████████
  #9  Attention                80.18%  ████████████
  #10  LSTM_Coords              74.92%  ███████████
──────────────────────────────────────────────────────────────────────────────────────────

──────────────────────────────────────────────────────────────────────────────────────────
  BEST CONFIG PER ARCHITECTURE
──────────────────────────────────────────────────────────────────────────────────────────
  Attention               cfg01  Acc= 80.18%  F1=0.7982  |  embed_dim=32.0  num_heads=4.0  dropout=0.2  lr=0.001  batch_size=32
  BiLSTM                  cfg01  Acc= 82.43%  F1=0.8256  |  hidden=32.0  layers=2.0  dropout=0.2  lr=0.001  batch_size=32
  CNN1D                   cfg01  Acc= 83.18%  F1=0.8187  |  conv_ch=32.0  fc_hid=32.0  dropout=0.2  lr=0.001  batch_size=32
  LSTM_All_Joints_Vel     cfg01  Acc= 83.48%  F1=0.8172  |  hidden=32.0  layers=2.0  dropout=0.2  lr=0.001  batch_size=32
  LSTM_Combined           cfg01  Acc= 84.53%  F1=0.8383  |  hidden=32.0  layers=2.0  dropout=0.2  lr=0.001  batch_size=32
  LSTM_Coords             cfg01  Acc= 74.92%  F1=0.7113  |  hidden=32.0  layers=2.0  dropout=0.2  lr=0.001  batch_size=32
  LSTM_Vel_Speed          cfg01  Acc= 83.93%  F1=0.8346  |  hidden=32.0  layers=2.0  dropout=0.2  lr=0.001  batch_size=32
  LSTM_Velocities         cfg01  Acc= 82.13%  F1=0.8108  |  hidden=32.0  layers=2.0  dropout=0.2  lr=0.001  batch_size=32
  ResNet1D                cfg01  Acc= 84.38%  F1=0.8398  |  hidden_dim=32.0  dropout=0.2  lr=0.001  batch_size=32
  TCN                     cfg01  Acc= 83.48%  F1=0.8271  |  tcn_channels=32.0  num_levels=2.0  dropout=0.2  lr=0.001  batch_size=32
──────────────────────────────────────────────────────────────────────────────────────────

==========================================================================================
  TOP 5 CONFIGURATIONS OVERALL
==========================================================================================
  #1  LSTM_Combined           cfg01  →  Acc= 84.53%  |  F1=0.8383  |  Prec=0.8358  Rec=0.8408  |  16.5s
  #2  ResNet1D                cfg01  →  Acc= 84.38%  |  F1=0.8398  |  Prec=0.8083  Rec=0.8739  |  22.4s
  #3  LSTM_Vel_Speed          cfg01  →  Acc= 83.93%  |  F1=0.8346  |  Prec=0.8436  Rec=0.8258  |  8.7s
  #4  LSTM_All_Joints_Vel     cfg01  →  Acc= 83.48%  |  F1=0.8172  |  Prec=0.8682  Rec=0.7718  |  11.9s
  #5  TCN                     cfg01  →  Acc= 83.48%  |  F1=0.8271  |  Prec=0.8283  Rec=0.8258  |  11.4s
==========================================================================================

==========================================================================================
  FULL BENCHMARK RESULTS  (sorted by Test Accuracy)
==========================================================================================
 rank                arch  config_id best_test_acc f1_touch precision_touch recall_touch train_time_s
    1       LSTM_Combined          1        84.53%   0.8383          0.8358       0.8408        16.5s
    2            ResNet1D          1        84.38%   0.8398          0.8083       0.8739        22.4s
    3      LSTM_Vel_Speed          1        83.93%   0.8346          0.8436       0.8258         8.7s
    4 LSTM_All_Joints_Vel          1        83.48%   0.8172          0.8682       0.7718        11.9s
    5                 TCN          1        83.48%   0.8271          0.8283       0.8258        11.4s
    6               CNN1D          1        83.18%   0.8187          0.8534       0.7868        12.2s
    7              BiLSTM          1        82.43%   0.8256          0.8195       0.8318        13.3s
    8     LSTM_Velocities          1        82.13%   0.8108          0.8108       0.8108         7.1s
    9           Attention          1        80.18%   0.7982          0.7778       0.8198        14.5s
   10         LSTM_Coords          1        74.92%   0.7113          0.7786       0.6547        16.8s
==========================================================================================


















# Iteration 4

## Parameters
-d 0.0

## Process.sh
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
python3 datacreator/filter_landmarks.py -min 5 -beta 2.4 -d 1.0 -i ./dataprocessing/2_normalized_coordinates/*.normalize_landmarks.* -o ./dataprocessing/3_euroFilter_coordinates/

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

## Ouput


──────────────────────────────────────────────────────────────────────────────────────────
  ARCHITECTURE RANKING  (by best single config accuracy)
──────────────────────────────────────────────────────────────────────────────────────────
  #1  LSTM_Vel_Speed           89.64%  █████████████
  #2  LSTM_All_Joints_Vel      89.04%  █████████████
  #3  LSTM_Combined            88.74%  █████████████
  #4  BiLSTM                   88.44%  █████████████
  #5  TCN                      88.44%  █████████████
  #6  LSTM_Velocities          88.44%  █████████████
  #7  Attention                87.69%  █████████████
  #8  ResNet1D                 87.54%  █████████████
  #9  CNN1D                    87.24%  █████████████
  #10  LSTM_Coords              85.59%  ████████████
──────────────────────────────────────────────────────────────────────────────────────────

──────────────────────────────────────────────────────────────────────────────────────────
  BEST CONFIG PER ARCHITECTURE
──────────────────────────────────────────────────────────────────────────────────────────
  Attention               cfg01  Acc= 87.69%  F1=0.8660  |  embed_dim=32.0  num_heads=4.0  dropout=0.0  lr=0.001  batch_size=32
  BiLSTM                  cfg01  Acc= 88.44%  F1=0.8622  |  hidden=32.0  layers=2.0  dropout=0.0  lr=0.001  batch_size=32
  CNN1D                   cfg01  Acc= 87.24%  F1=0.8707  |  conv_ch=32.0  fc_hid=32.0  dropout=0.0  lr=0.001  batch_size=32
  LSTM_All_Joints_Vel     cfg01  Acc= 89.04%  F1=0.8735  |  hidden=32.0  layers=2.0  dropout=0.0  lr=0.001  batch_size=32
  LSTM_Combined           cfg01  Acc= 88.74%  F1=0.8668  |  hidden=32.0  layers=2.0  dropout=0.0  lr=0.001  batch_size=32
  LSTM_Coords             cfg01  Acc= 85.59%  F1=0.8484  |  hidden=32.0  layers=2.0  dropout=0.0  lr=0.001  batch_size=32
  LSTM_Vel_Speed          cfg01  Acc= 89.64%  F1=0.8889  |  hidden=32.0  layers=2.0  dropout=0.0  lr=0.001  batch_size=32
  LSTM_Velocities         cfg01  Acc= 88.44%  F1=0.8796  |  hidden=32.0  layers=2.0  dropout=0.0  lr=0.001  batch_size=32
  ResNet1D                cfg01  Acc= 87.54%  F1=0.8463  |  hidden_dim=32.0  dropout=0.0  lr=0.001  batch_size=32
  TCN                     cfg01  Acc= 88.44%  F1=0.8773  |  tcn_channels=32.0  num_levels=2.0  dropout=0.0  lr=0.001  batch_size=32
──────────────────────────────────────────────────────────────────────────────────────────

==========================================================================================
  TOP 5 CONFIGURATIONS OVERALL
==========================================================================================
  #1  LSTM_Vel_Speed          cfg01  →  Acc= 89.64%  |  F1=0.8889  |  Prec=0.8772  Rec=0.9009  |  12.8s
  #2  LSTM_All_Joints_Vel     cfg01  →  Acc= 89.04%  |  F1=0.8735  |  Prec=0.8984  Rec=0.8498  |  9.0s
  #3  LSTM_Combined           cfg01  →  Acc= 88.74%  |  F1=0.8668  |  Prec=0.8457  Rec=0.8889  |  8.2s
  #4  BiLSTM                  cfg01  →  Acc= 88.44%  |  F1=0.8622  |  Prec=0.8509  Rec=0.8739  |  11.9s
  #5  TCN                     cfg01  →  Acc= 88.44%  |  F1=0.8773  |  Prec=0.8444  Rec=0.9129  |  10.6s
==========================================================================================

==========================================================================================
  FULL BENCHMARK RESULTS  (sorted by Test Accuracy)
==========================================================================================
 rank                arch  config_id best_test_acc f1_touch precision_touch recall_touch train_time_s
    1      LSTM_Vel_Speed          1        89.64%   0.8889          0.8772       0.9009        12.8s
    2 LSTM_All_Joints_Vel          1        89.04%   0.8735          0.8984       0.8498         9.0s
    3       LSTM_Combined          1        88.74%   0.8668          0.8457       0.8889         8.2s
    4              BiLSTM          1        88.44%   0.8622          0.8509       0.8739        11.9s
    5                 TCN          1        88.44%   0.8773          0.8444       0.9129        10.6s
    6     LSTM_Velocities          1        88.44%   0.8796          0.9048       0.8559         7.8s
    7           Attention          1        87.69%   0.8660          0.8497       0.8829         8.8s
    8            ResNet1D          1        87.54%   0.8463          0.8580       0.8348         8.8s
    9               CNN1D          1        87.24%   0.8707          0.8618       0.8799         5.5s
   10         LSTM_Coords          1        85.59%   0.8484          0.8656       0.8318        26.1s
==========================================================================================

  Full ranked summary saved → /home/lahirukasunidilhara/Documents/university/research/mediapipeDetector/deepLearningModels/results/summary_all.csv
