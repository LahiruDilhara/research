# python3 analysis/merge_csvs.py -i ./dataset/*.csv -o data/all_merged.csv 
# python3 analysis/split_fingers.py -i ./data/all_merged.csv -o ./data/finger_split_all.csv 
# python3 analysis/filter_dataset.py -i ./data/finger_split_all.csv -o ./data/cleaned_data.csv --remove-zero-vel-touch --remove-hand-invisible --remove-out-of-sync
# python3 analysis/split_touch.py -i ./data/cleaned_data.csv --touch-out ./data/touch_dataset.csv --untouch-out ./data/untouch_dataset.csv
# python3 analysis/create_train_test_split.py --touch-in ./data/touch_dataset.csv --untouch-in ./data/untouch_dataset.csv --train-out ./data/training_data.csv --test-out ./data/test_data.csv --touch-test-pct 15 --untouch-train-ratio-pct 125 --untouch-test-ratio-pct 125 --seed 50 --no-video-leak


python3 analysis/merge_csvs.py -i ./dataset/*.csv -o data/all_merged.csv 
python3 analysis/split_fingers.py -i ./data/all_merged.csv -o ./data/finger_split_all.csv 
python3 analysis/filter_dataset.py -i ./data/finger_split_all.csv -o ./data/cleaned_data.csv --remove-zero-vel-touch --remove-hand-invisible --remove-out-of-sync
python3 analysis/split_touch.py -i ./data/cleaned_data.csv --touch-out ./data/touch_dataset.csv --untouch-out ./data/untouch_dataset.csv
python3 analysis/create_train_test_split.py --touch-in ./data/touch_dataset.csv --untouch-in ./data/untouch_dataset.csv --train-out ./data/training_data.csv --test-out ./data/test_data.csv --touch-test-pct 10 --untouch-train-ratio-pct 125 --untouch-test-ratio-pct 125 --seed 50 --no-video-leak