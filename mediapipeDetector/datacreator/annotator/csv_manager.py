"""
datacreator/annotator/csv_manager.py

Handles loading raw frame landmark CSVs (containing all 21 landmark coordinates per frame)
and saving window-based touch and environment annotations referencing frame ranges.
Only saves windows that have been visited/annotated when leaving a window.
Zero MediaPipe dependencies at runtime.
"""

import csv
import os

FINGERS = ["Thumb", "Index", "Middle", "Ring", "Pinky"]
ALL_21_LANDMARK_NAMES = [
    "wrist",
    "thumb_cmc", "thumb_mcp", "thumb_ip", "thumb_tip",
    "index_mcp", "index_pip", "index_dip", "index_tip",
    "middle_mcp", "middle_pip", "middle_dip", "middle_tip",
    "ring_mcp", "ring_pip", "ring_dip", "ring_tip",
    "pinky_mcp", "pinky_pip", "pinky_dip", "pinky_tip",
]


class WindowAnnotationCSVManager:
    def __init__(self, raw_csv_path: str, window_size: int = 5, window_overlap: int = 2):
        self.raw_csv_path = raw_csv_path
        self.window_size = window_size
        self.window_overlap = window_overlap
        self.window_step = max(1, window_size - window_overlap)

        self.raw_rows: list[dict] = []
        self.headers: list[str] = []
        self.video_hash: str = ""
        self.video_file: str = ""
        self.total_frames: int = 0
        self.total_windows: int = 0

        self.window_annotations: dict[int, dict] = {}

        self.load_raw_landmarks()

    def set_window_parameters(self, window_size: int, window_overlap: int):
        """Updates window parameters and re-calculates window count."""
        self.window_size = max(1, window_size)
        self.window_overlap = max(0, min(window_overlap, self.window_size - 1))
        self.window_step = max(1, self.window_size - self.window_overlap)

        if self.total_frames >= self.window_size:
            self.total_windows = 1 + (self.total_frames - self.window_size) // self.window_step
        else:
            self.total_windows = 0

    def load_raw_landmarks(self):
        """Loads all raw frame landmarks from the CSV file."""
        if not os.path.exists(self.raw_csv_path):
            raise FileNotFoundError(f"Raw landmarks CSV file not found: {self.raw_csv_path}")

        with open(self.raw_csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            self.headers = reader.fieldnames or []
            self.raw_rows = list(reader)

        self.total_frames = len(self.raw_rows)
        if self.raw_rows:
            self.video_hash = self.raw_rows[0].get("video_hash", "")
            self.video_file = self.raw_rows[0].get("video_file", "")

        self.set_window_parameters(self.window_size, self.window_overlap)

        # Load existing window annotations if annotation CSV exists
        annotation_csv_path = self.get_annotation_csv_path()
        if os.path.exists(annotation_csv_path):
            self.load_existing_annotations(annotation_csv_path)

    def get_annotation_csv_path(self) -> str:
        """Determines the window annotations CSV filepath in the same directory."""
        dir_name = os.path.dirname(os.path.abspath(self.raw_csv_path))
        base_name = os.path.basename(self.raw_csv_path)

        if ".raw_landmarks." in base_name:
            ann_name = base_name.replace(".raw_landmarks.", ".window_annotations.")
        else:
            name_no_ext = os.path.splitext(base_name)[0]
            ann_name = f"{name_no_ext}.window_annotations.csv"

        return os.path.join(dir_name, ann_name)

    def load_existing_annotations(self, annotation_csv_path: str):
        """Loads previously saved window touch and environment annotations."""
        try:
            with open(annotation_csv_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    w_idx = int(row.get("window_idx", -1))
                    if w_idx >= 0:
                        self.window_annotations[w_idx] = row
            print(f"[Info] Loaded {len(self.window_annotations)} existing window annotations from {annotation_csv_path}")
        except Exception as e:
            print(f"[Warning] Could not parse existing annotations CSV: {e}")

    def get_window_frame_indices(self, window_idx: int) -> tuple[int, int]:
        """Returns (start_frame, end_frame) for a 0-indexed window."""
        start_frame = window_idx * self.window_step
        end_frame = min(self.total_frames - 1, start_frame + self.window_size - 1)
        return start_frame, end_frame

    def get_window_frames_data(self, window_idx: int) -> list[dict]:
        """Returns the list of raw landmark dicts for frames in window_idx."""
        start_frame, end_frame = self.get_window_frame_indices(window_idx)
        return self.raw_rows[start_frame : end_frame + 1]

    def has_annotation(self, window_idx: int) -> bool:
        """Checks if window_idx has already been saved to CSV."""
        return window_idx in self.window_annotations

    def get_window_annotation(self, window_idx: int) -> dict:
        """Returns annotation dict for a given window_idx (inheriting environment from previous window if unannotated)."""
        if window_idx in self.window_annotations:
            return self.window_annotations[window_idx]

        start_frame, end_frame = self.get_window_frame_indices(window_idx)
        start_ms = self.raw_rows[start_frame].get("timestamp_ms", "0") if start_frame < len(self.raw_rows) else "0"
        end_ms = self.raw_rows[end_frame].get("timestamp_ms", "0") if end_frame < len(self.raw_rows) else "0"

        # Look up most recent saved/visited window to inherit environment controls
        prev_idx = window_idx - 1
        while prev_idx >= 0 and prev_idx not in self.window_annotations:
            prev_idx -= 1

        prev_ann = self.window_annotations.get(prev_idx, {}) if prev_idx >= 0 else {}

        raw_hand = "Right"
        if start_frame < len(self.raw_rows):
            raw_hand = self.raw_rows[start_frame].get("hand", "Right")
        default_right_hand = "1" if raw_hand == "Right" else "0"

        right_hand_val = prev_ann.get("right_hand", default_right_hand)
        hand_move_val = prev_ann.get("hand_move", "0")
        hand_closer_val = prev_ann.get("hand_closer", "0")
        hovering_val = prev_ann.get("hovering", "0")
        daylight_val = prev_ann.get("daylight", "1")
        hand_visible_val = prev_ann.get("hand_visible", "1")
        out_of_sync_val = prev_ann.get("out_of_sync", "0")

        default_dict = {
            "video_file": self.video_file,
            "raw_landmarks_csv": os.path.basename(self.raw_csv_path),
            "video_hash": self.video_hash,
            "window_idx": str(window_idx),
            "start_frame": str(start_frame),
            "end_frame": str(end_frame),
            "start_ms": str(start_ms),
            "end_ms": str(end_ms),
            "window_size": str(self.window_size),
            "window_overlap": str(self.window_overlap),
            "right_hand": right_hand_val,
            "hand_move": hand_move_val,
            "hand_closer": hand_closer_val,
            "hovering": hovering_val,
            "daylight": daylight_val,
            "hand_visible": hand_visible_val,
            "out_of_sync": out_of_sync_val,
            "thumb_touch": "0",
            "index_touch": "0",
            "middle_touch": "0",
            "ring_touch": "0",
            "pinky_touch": "0",
            "any_touch": "0",
        }
        return default_dict

    def update_window_annotation(
        self,
        window_idx: int,
        touch_dict: dict,
        right_hand: bool = True,
        hand_move: bool = False,
        hand_closer: bool = False,
        hovering: bool = False,
        daylight: bool = True,
        hand_visible: bool = True,
        out_of_sync: bool = False
    ):
        """Updates window-level touch and environment labels for window_idx."""
        ann = self.get_window_annotation(window_idx)
        any_touch = False
        for f in FINGERS:
            val = 1 if touch_dict.get(f, False) else 0
            ann[f"{f.lower()}_touch"] = str(val)
            if val == 1:
                any_touch = True

        ann["any_touch"] = "1" if any_touch else "0"
        ann["right_hand"] = "1" if right_hand else "0"
        ann["hand_move"] = "1" if hand_move else "0"
        ann["hand_closer"] = "1" if hand_closer else "0"
        ann["hovering"] = "1" if hovering else "0"
        ann["daylight"] = "1" if daylight else "0"
        ann["hand_visible"] = "1" if hand_visible else "0"
        ann["out_of_sync"] = "1" if out_of_sync else "0"

        self.window_annotations[window_idx] = ann

    def save_window_annotations(self, output_path: str = None, verbose: bool = False) -> str:
        """Saves only windows present in self.window_annotations to CSV."""
        save_path = output_path or self.get_annotation_csv_path()

        headers = [
            "video_file", "raw_landmarks_csv", "video_hash",
            "window_idx", "start_frame", "end_frame", "start_ms", "end_ms",
            "window_size", "window_overlap",
            "right_hand", "hand_move", "hand_closer", "hovering", "daylight", "hand_visible", "out_of_sync",
            "thumb_touch", "index_touch", "middle_touch", "ring_touch", "pinky_touch", "any_touch"
        ]

        sorted_indices = sorted(self.window_annotations.keys())
        rows_to_save = []
        for idx in sorted_indices:
            row = dict(self.window_annotations[idx])
            row.pop("is_annotated", None)
            rows_to_save.append(row)

        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        with open(save_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=headers, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows_to_save)

        if verbose:
            print(f"[Success] Saved {len(rows_to_save)} window annotations to: {save_path}")
        return save_path
