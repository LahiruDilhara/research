# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "customtkinter>=6.0.0",
#     "numpy>=2.5.2",
#     "opencv-python>=5.0.0.93",
#     "pillow",
# ]
# ///

"""
datacreator/hand_movement_analyzer_ui.py

Interactive Visual Analyzer for calibrating Hand Movement Displacement Threshold.
Loads pre-extracted 12 FPS MediaPipe landmark CSVs directly (matching pipeline 100%),
constructs 5-frame sliding windows (stride = 3 frames), and provides interactive
controls to tune the coordinate-based displacement threshold (relative to rigid palm scale L_hand).

Stationary reference landmarks:
  - Wrist (Joint 0)
  - Index MCP (Joint 5)
  - Middle MCP (Joint 9)
  - Ring MCP (Joint 13)
  - Pinky MCP (Joint 17)
"""

import argparse
import csv
import glob
import math
import os
import sys
from pathlib import Path
from tkinter import messagebox

import cv2
import customtkinter as ctk
import numpy as np
from PIL import Image, ImageTk

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datacreator.annotator.utils import open_csv_dialog, open_video_dialog

# Theme Colors (VS Code Dark+)
FF = "Helvetica"
BG = "#1e1e1e"
PANEL = "#252526"
BORDER = "#3c3c3c"
HDR_BG = "#333333"
TXT_PRI = "#ffffff"
TXT_SEC = "#cccccc"
TXT_MUT = "#858585"
BTN_PRI = "#007acc"
BTN_HVP = "#005999"
BTN_SEC = "#3a3d41"
GREEN = "#4ec9f0"
AMBER = "#ce9178"
RED = "#f44747"
PASS_BG = "#1e3a1e"
FAIL_BG = "#3a1e1e"

# Stationary Reference Landmark Indices
WRIST_INDEX = 0
STATIONARY_INDICES = [0, 5, 9, 13, 17]
STATIONARY_NAMES = {
    0: "Wrist",
    5: "Index MCP",
    9: "Middle MCP",
    13: "Ring MCP",
    17: "Pinky MCP",
}

ALL_21_LANDMARK_NAMES = [
    "wrist",
    "thumb_cmc", "thumb_mcp", "thumb_ip", "thumb_tip",
    "index_mcp", "index_pip", "index_dip", "index_tip",
    "middle_mcp", "middle_pip", "middle_dip", "middle_tip",
    "ring_mcp", "ring_pip", "ring_dip", "ring_tip",
    "pinky_mcp", "pinky_pip", "pinky_dip", "pinky_tip",
]

# 21 Hand Landmark Connections for Skeleton Drawing
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),          # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),          # Index
    (5, 9), (9, 10), (10, 11), (11, 12),     # Middle
    (9, 13), (13, 14), (14, 15), (15, 16),   # Ring
    (13, 17), (17, 18), (18, 19), (19, 20),  # Pinky
    (0, 17),                                  # Palm Base
]


def calculate_palm_scale(pts_px: list[tuple[float, float, float]]) -> float:
    """Calculates rigid palm scale L_hand using Root-Mean-Square (RMS) of 8 symmetric palm segments."""
    w_x, w_y, _ = pts_px[0]
    i_x, i_y, _ = pts_px[5]
    m_x, m_y, _ = pts_px[9]
    r_x, r_y, _ = pts_px[13]
    p_x, p_y, _ = pts_px[17]

    d_sq = [
        (i_x - w_x) ** 2 + (i_y - w_y) ** 2,
        (m_x - w_x) ** 2 + (m_y - w_y) ** 2,
        (r_x - w_x) ** 2 + (r_y - w_y) ** 2,
        (p_x - w_x) ** 2 + (p_y - w_y) ** 2,
        (m_x - i_x) ** 2 + (m_y - i_y) ** 2,
        (r_x - m_x) ** 2 + (r_y - m_y) ** 2,
        (p_x - r_x) ** 2 + (p_y - r_y) ** 2,
        (p_x - i_x) ** 2 + (p_y - i_y) ** 2,
    ]
    l_hand = math.sqrt(sum(d_sq) / 8.0)
    return max(1.0, l_hand)


def find_matching_raw_csv(video_path: str) -> str | None:
    """Finds matching *.raw_landmarks.*.csv for the given video file."""
    video_dir = os.path.dirname(os.path.abspath(video_path))
    base_name = os.path.splitext(os.path.basename(video_path))[0]

    search_dirs = [
        video_dir,
        os.path.join(PROJECT_ROOT, "videos"),
        os.path.join(PROJECT_ROOT, "dataset"),
        os.path.join(PROJECT_ROOT, "dataprocessing", "1_rawCSVFiles"),
    ]

    for s_dir in search_dirs:
        if not os.path.exists(s_dir):
            continue
        matches = glob.glob(os.path.join(s_dir, f"{base_name}.raw_landmarks.*.csv"))
        if matches:
            return matches[0]
        cand = os.path.join(s_dir, f"{base_name}.raw_landmarks.csv")
        if os.path.exists(cand):
            return cand

    return None


class HandMovementAnalyzerApp(ctk.CTk):
    def __init__(self, initial_video: str = ""):
        super().__init__()

        self.title("Hand Movement Threshold Analyzer (12 FPS)")
        self.geometry("1400x900")
        self.minsize(1200, 800)
        ctk.set_appearance_mode("Dark")
        ctk.set_default_color_theme("blue")

        # Video State
        self.video_path = initial_video
        self.csv_path = None
        self.cap = None
        self.video_width = 640
        self.video_height = 480
        self.total_frames = 0
        self.actual_fps = 12.0
        self.fps_valid = False

        # Landmark Cache: list of list of 21 (x_px, y_px, z_px) points per frame
        self.frames_landmarks = []
        self.frames_raw_images = []
        self.palm_scales = []

        # Windows: list of dicts with 5-frame sequence data
        self.windows = []
        self.current_window_idx = 0
        self.current_frame_in_window = 0  # 0..4, default to first frame (Frame 1) in window

        # Threshold (Default: 0.100 hand-lengths)
        self.disp_threshold = 0.100

        # Playback animation state
        self.is_playing = False
        self.play_job = None

        self._build_ui()

        # Keyboard shortcuts
        self.bind("<space>", lambda e: self.toggle_play())
        self.bind("<Left>", lambda e: self.prev_window())
        self.bind("<Right>", lambda e: self.next_window())
        self.bind("a", lambda e: self.prev_frame_in_window())
        self.bind("d", lambda e: self.next_frame_in_window())

        if initial_video and os.path.exists(initial_video):
            self.load_video(initial_video)

    def _build_ui(self):
        # ── Top Bar ──────────────────────────────────────────────────────────
        top_bar = ctk.CTkFrame(self, height=52, fg_color=PANEL, corner_radius=0)
        top_bar.pack(fill="x", side="top")

        btn_open = ctk.CTkButton(
            top_bar, text="📁 Open Video...", width=140, height=32,
            font=ctk.CTkFont(family=FF, size=13, weight="bold"),
            fg_color=BTN_PRI, hover_color=BTN_HVP,
            command=self._browse_video
        )
        btn_open.pack(side="left", padx=(16, 10), pady=10)

        btn_csv = ctk.CTkButton(
            top_bar, text="📄 Open Landmark CSV...", width=170, height=32,
            font=ctk.CTkFont(family=FF, size=12),
            fg_color=BTN_SEC, hover_color=BORDER,
            command=self._browse_csv
        )
        btn_csv.pack(side="left", padx=5, pady=10)

        self.lbl_fps_status = ctk.CTkLabel(
            top_bar, text="FPS: Not Loaded",
            font=ctk.CTkFont(family=FF, size=12, weight="bold"),
            text_color=TXT_MUT
        )
        self.lbl_fps_status.pack(side="left", padx=15, pady=10)

        self.lbl_video_title = ctk.CTkLabel(
            top_bar, text="No Video Selected",
            font=ctk.CTkFont(family=FF, size=13),
            text_color=TXT_SEC
        )
        self.lbl_video_title.pack(side="left", padx=10, pady=10)

        # ── Main Content Area (Split Left & Right) ───────────────────────────
        main_body = ctk.CTkFrame(self, fg_color=BG)
        main_body.pack(fill="both", expand=True, padx=12, pady=12)

        # Left Column: Video & Window Frame Player
        left_col = ctk.CTkFrame(main_body, fg_color=PANEL, corner_radius=8)
        left_col.pack(side="left", fill="both", expand=True, padx=(0, 8))

        # Video Canvas
        self.video_canvas = ctk.CTkLabel(left_col, text="", fg_color="#101010", corner_radius=6)
        self.video_canvas.pack(fill="both", expand=True, padx=12, pady=12)

        # In-Window Sub-Frame Control Bar
        sub_frame_ctrl = ctk.CTkFrame(left_col, height=44, fg_color=HDR_BG, corner_radius=6)
        sub_frame_ctrl.pack(fill="x", padx=12, pady=(0, 12))

        ctk.CTkLabel(
            sub_frame_ctrl, text="Window Step Frame:",
            font=ctk.CTkFont(family=FF, size=12, weight="bold")
        ).pack(side="left", padx=(12, 8))

        self.sub_frame_btns = []
        for f_idx in range(5):
            btn = ctk.CTkButton(
                sub_frame_ctrl, text=f"Frame {f_idx + 1}", width=75, height=28,
                font=ctk.CTkFont(family=FF, size=11),
                fg_color=BTN_PRI if f_idx == 0 else BTN_SEC,
                command=lambda i=f_idx: self._select_sub_frame(i)
            )
            btn.pack(side="left", padx=3)
            self.sub_frame_btns.append(btn)

        self.var_show_trails = ctk.BooleanVar(value=True)
        ctk.CTkCheckBox(
            sub_frame_ctrl, text="Show Motion Trails (Frame 1 → 5)", variable=self.var_show_trails,
            font=ctk.CTkFont(family=FF, size=11), command=self.render_current_view
        ).pack(side="right", padx=12)

        # Right Column: Controls, Threshold Slider & Metrics
        right_col = ctk.CTkFrame(main_body, width=460, fg_color=PANEL, corner_radius=8)
        right_col.pack(side="right", fill="y", padx=(8, 0))
        right_col.pack_propagate(False)

        # ── Section 1: Window Navigator ──────────────────────────────────────
        sec1 = ctk.CTkFrame(right_col, fg_color=HDR_BG, corner_radius=6)
        sec1.pack(fill="x", padx=12, pady=(12, 8))

        lbl_sec1 = ctk.CTkLabel(
            sec1, text="WINDOW NAVIGATION (5 FRAMES, STRIDE 3)",
            font=ctk.CTkFont(family=FF, size=11, weight="bold"), text_color=TXT_MUT
        )
        lbl_sec1.pack(anchor="w", padx=10, pady=(6, 2))

        win_ctrl_row = ctk.CTkFrame(sec1, fg_color="transparent")
        win_ctrl_row.pack(fill="x", padx=10, pady=6)

        self.btn_prev = ctk.CTkButton(
            win_ctrl_row, text="◀ Prev", width=70, height=28,
            fg_color=BTN_SEC, hover_color=BORDER, command=self.prev_window
        )
        self.btn_prev.pack(side="left", padx=(0, 4))

        self.btn_play = ctk.CTkButton(
            win_ctrl_row, text="▶ Play", width=70, height=28,
            fg_color=BTN_PRI, hover_color=BTN_HVP, command=self.toggle_play
        )
        self.btn_play.pack(side="left", padx=4)

        self.btn_next = ctk.CTkButton(
            win_ctrl_row, text="Next ▶", width=70, height=28,
            fg_color=BTN_SEC, hover_color=BORDER, command=self.next_window
        )
        self.btn_next.pack(side="left", padx=4)

        self.lbl_window_counter = ctk.CTkLabel(
            win_ctrl_row, text="Window: 0 / 0",
            font=ctk.CTkFont(family=FF, size=12, weight="bold"), text_color=GREEN
        )
        self.lbl_window_counter.pack(side="right", padx=6)

        self.slider_window = ctk.CTkSlider(
            sec1, from_=0, to=1, number_of_steps=1, command=self._on_window_slider
        )
        self.slider_window.pack(fill="x", padx=10, pady=(0, 8))

        # Quick Jump buttons
        jump_row = ctk.CTkFrame(sec1, fg_color="transparent")
        jump_row.pack(fill="x", padx=10, pady=(0, 8))
        ctk.CTkButton(
            jump_row, text="Jump Next Moving ❯", width=120, height=24,
            font=ctk.CTkFont(family=FF, size=10), fg_color="#5a2020", hover_color="#7a2828",
            command=lambda: self.jump_to_window(target_is_moving=True)
        ).pack(side="left")
        ctk.CTkButton(
            jump_row, text="Jump Next Stationary ❯", width=130, height=24,
            font=ctk.CTkFont(family=FF, size=10), fg_color="#205a20", hover_color="#287a28",
            command=lambda: self.jump_to_window(target_is_moving=False)
        ).pack(side="right")

        # ── Section 2: Stationary Coordinate Displacement ────────────────────
        sec2 = ctk.CTkFrame(right_col, fg_color=HDR_BG, corner_radius=6)
        sec2.pack(fill="x", padx=12, pady=6)

        lbl_sec2 = ctk.CTkLabel(
            sec2, text="STATIONARY DISPLACEMENT (WRIST & MCP JOINTS)",
            font=ctk.CTkFont(family=FF, size=11, weight="bold"), text_color=TXT_MUT
        )
        lbl_sec2.pack(anchor="w", padx=10, pady=(6, 2))

        self.lbl_disp_metric = ctk.CTkLabel(
            sec2, text="Wrist Disp: 0.0000 | Max MCP Disp: 0.0000",
            font=ctk.CTkFont(family=FF, size=11), text_color=TXT_SEC
        )
        self.lbl_disp_metric.pack(anchor="w", padx=10, pady=2)

        disp_slider_row = ctk.CTkFrame(sec2, fg_color="transparent")
        disp_slider_row.pack(fill="x", padx=10, pady=4)
        ctk.CTkLabel(disp_slider_row, text="Displacement Threshold:", font=ctk.CTkFont(family=FF, size=11)).pack(side="left")
        self.lbl_disp_thresh_val = ctk.CTkLabel(
            disp_slider_row, text=f"{self.disp_threshold:.3f} L_hand",
            font=ctk.CTkFont(family=FF, size=11, weight="bold"), text_color=GREEN
        )
        self.lbl_disp_thresh_val.pack(side="right")

        self.slider_disp = ctk.CTkSlider(
            sec2, from_=0.010, to=0.350, number_of_steps=68, command=self._on_disp_slider
        )
        self.slider_disp.set(self.disp_threshold)
        self.slider_disp.pack(fill="x", padx=10, pady=(0, 6))

        self.lbl_disp_status = ctk.CTkLabel(
            sec2, text="Status: PENDING",
            font=ctk.CTkFont(family=FF, size=11, weight="bold"), text_color=TXT_MUT
        )
        self.lbl_disp_status.pack(anchor="w", padx=10, pady=(0, 8))

        # ── Section 3: Joint Breakdown Card ──────────────────────────────────
        sec3 = ctk.CTkFrame(right_col, fg_color=HDR_BG, corner_radius=6)
        sec3.pack(fill="x", padx=12, pady=6)

        lbl_sec3 = ctk.CTkLabel(
            sec3, text="PER-JOINT DISPLACEMENT BREAKDOWN (L_HAND)",
            font=ctk.CTkFont(family=FF, size=11, weight="bold"), text_color=TXT_MUT
        )
        lbl_sec3.pack(anchor="w", padx=10, pady=(6, 4))

        self.lbl_joint_breakdown = ctk.CTkLabel(
            sec3, text="Wrist: 0.0000 | Index MCP: 0.0000\nMiddle MCP: 0.0000 | Ring MCP: 0.0000 | Pinky MCP: 0.0000",
            font=ctk.CTkFont(family=FF, size=11), justify="left", text_color=TXT_SEC
        )
        self.lbl_joint_breakdown.pack(anchor="w", padx=10, pady=(0, 8))

        # ── Section 4: Final Window Decision Badge ───────────────────────────
        self.box_decision = ctk.CTkFrame(right_col, height=75, fg_color=PANEL, corner_radius=8, border_width=2, border_color=BORDER)
        self.box_decision.pack(fill="x", padx=12, pady=8)
        self.box_decision.pack_propagate(False)

        self.lbl_decision_main = ctk.CTkLabel(
            self.box_decision, text="WINDOW DECISION",
            font=ctk.CTkFont(family=FF, size=14, weight="bold"), text_color=TXT_MUT
        )
        self.lbl_decision_main.pack(pady=(12, 2))

        self.lbl_decision_sub = ctk.CTkLabel(
            self.box_decision, text="No window loaded",
            font=ctk.CTkFont(family=FF, size=11), text_color=TXT_MUT
        )
        self.lbl_decision_sub.pack(pady=(0, 8))

        # ── Section 5: Global Dataset Statistics ─────────────────────────────
        sec5 = ctk.CTkFrame(right_col, fg_color=HDR_BG, corner_radius=6)
        sec5.pack(fill="both", expand=True, padx=12, pady=(6, 12))

        lbl_sec5 = ctk.CTkLabel(
            sec5, text="GLOBAL PIPELINE AUDIT (WINDOW RETENTION)",
            font=ctk.CTkFont(family=FF, size=11, weight="bold"), text_color=TXT_MUT
        )
        lbl_sec5.pack(anchor="w", padx=10, pady=(6, 4))

        self.lbl_stats_summary = ctk.CTkLabel(
            sec5, text="Total Windows: 0\nStationary (Kept): 0 (0.0%)\nHand Moving (Dropped): 0 (0.0%)",
            font=ctk.CTkFont(family=FF, size=11), justify="left", text_color=TXT_SEC
        )
        self.lbl_stats_summary.pack(anchor="w", padx=10, pady=(0, 8))

    # ── 12 FPS Validation & Video / CSV Loading ──────────────────────────────

    def _browse_video(self):
        v_path = open_video_dialog()
        if v_path:
            self.load_video(v_path)

    def _browse_csv(self):
        c_path = open_csv_dialog()
        if c_path:
            if not self.video_path:
                # If no video opened yet, try to find matching video
                base_name = os.path.basename(c_path).split(".raw_landmarks.")[0]
                v_cand = os.path.join(os.path.dirname(c_path), f"{base_name}.mp4")
                if os.path.exists(v_cand):
                    self.load_video(v_cand, csv_path=c_path)
                    return
            self.load_landmarks_from_csv(c_path)

    def validate_12fps(self, video_path: str) -> tuple[bool, float, int]:
        """Validates whether video is approximately 12 FPS (~11.4 to 12.6 FPS)."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return False, 0.0, 0

        header_fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        first_msec = None
        last_msec = None
        count = 0

        while True:
            ret, _ = cap.read()
            if not ret:
                break
            msec = cap.get(cv2.CAP_PROP_POS_MSEC)
            if first_msec is None:
                first_msec = msec
            last_msec = msec
            count += 1

        cap.release()

        if count > 1 and last_msec is not None and first_msec is not None and last_msec > first_msec:
            actual_fps = count / ((last_msec - first_msec) / 1000.0)
        else:
            actual_fps = header_fps if header_fps > 0 else 12.0

        is_valid = 11.4 <= actual_fps <= 12.6
        return is_valid, actual_fps, count

    def load_video(self, video_path: str, csv_path: str = None):
        if not os.path.exists(video_path):
            messagebox.showerror("Error", f"Video file not found:\n{video_path}")
            return

        self.video_path = video_path
        self.lbl_video_title.configure(text=os.path.basename(video_path))

        # 1. 12 FPS Validation
        is_valid, fps, frame_count = self.validate_12fps(video_path)
        self.actual_fps = fps
        self.fps_valid = is_valid
        self.total_frames = frame_count

        if is_valid:
            self.lbl_fps_status.configure(
                text=f"FPS: {fps:.1f} (12 FPS Valid ✓)", text_color=GREEN
            )
        else:
            self.lbl_fps_status.configure(
                text=f"FPS: {fps:.1f} (⚠️ Non-12 FPS)", text_color=AMBER
            )
            messagebox.showwarning(
                "FPS Warning",
                f"Video is running at {fps:.2f} FPS (Expected: ~12.0 FPS).\n\n"
                f"For accurate timing synchronization, resample to 12 FPS using:\n"
                f"python3 datacreator/resample_12fps.py -i '{video_path}' -o ./resampled/"
            )

        # 2. Resolve matching raw landmarks CSV
        if not csv_path:
            csv_path = find_matching_raw_csv(video_path)

        if not csv_path:
            messagebox.showinfo(
                "Select Landmarks CSV",
                f"Could not auto-find matching *.raw_landmarks.*.csv for:\n{os.path.basename(video_path)}\n\n"
                f"Please select the corresponding landmarks CSV file."
            )
            csv_path = open_csv_dialog(initial_path=os.path.dirname(video_path))

        if not csv_path or not os.path.exists(csv_path):
            messagebox.showerror("Error", "No landmarks CSV file provided. Cannot proceed without landmarks.")
            return

        self.load_landmarks_from_csv(csv_path)

    def load_landmarks_from_csv(self, csv_path: str):
        """Loads landmarks from an existing .raw_landmarks.csv file (exact pipeline match)."""
        if not self.video_path or not os.path.exists(self.video_path):
            messagebox.showerror("Error", "Please select a video file first.")
            return

        self.csv_path = csv_path
        cap = cv2.VideoCapture(self.video_path)
        self.video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
        self.video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480

        # Read frames into memory for smooth interactive scrubbing
        self.frames_raw_images = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            self.frames_raw_images.append(frame)
        cap.release()

        if not self.frames_raw_images:
            messagebox.showerror("Error", "Could not read frames from video file.")
            return

        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        self.frames_landmarks = []
        self.palm_scales = []

        w = float(self.video_width)
        h = float(self.video_height)

        for row in rows:
            pts = []
            for name in ALL_21_LANDMARK_NAMES:
                rx = float(row.get(f"{name}_x", 0.0))
                ry = float(row.get(f"{name}_y", 0.0))
                rz = float(row.get(f"{name}_z", 0.0))
                pts.append((rx * w, ry * h, rz * w))
            self.frames_landmarks.append(pts)
            self.palm_scales.append(calculate_palm_scale(pts))

        self.lbl_video_title.configure(
            text=f"{os.path.basename(self.video_path)} | CSV: {os.path.basename(csv_path)}"
        )

        self.build_sliding_windows()
        self.recompute_all()
        self.set_window(0)

    def build_sliding_windows(self):
        """Builds standard 5-frame sliding windows with 2-frame overlap (step = 3 frames)."""
        self.windows = []
        total_f = min(len(self.frames_raw_images), len(self.frames_landmarks))
        if total_f < 5:
            return

        win_size = 5
        win_step = 3

        for start_f in range(0, total_f - win_size + 1, win_step):
            end_f = start_f + win_size
            self.windows.append({
                "start_frame": start_f,
                "end_frame": end_f - 1,
                "frame_indices": list(range(start_f, end_f)),
            })

        self.slider_window.configure(
            from_=0, to=max(0, len(self.windows) - 1),
            number_of_steps=max(1, len(self.windows) - 1)
        )

    # ── Kinematic Displacement Metric Calculations ───────────────────────────

    def compute_window_metrics(self, win: dict) -> dict:
        """
        Computes stationary displacement metrics strictly within the 5-frame window:
        - Completely isolated within this window (zero dependencies on other windows).
        - Frame 1 of this window serves as the fresh starting reference point.
        - Reference palm scale win_l_hand is calculated from Frame 1 of this window.
        - Net displacement for wrist and all 4 MCPs from Frame 1 to Frame 5.
        """
        f_indices = win["frame_indices"]
        win_l_hand = max(1.0, self.palm_scales[f_indices[0]])

        disp_by_joint = {}
        for j_idx in STATIONARY_INDICES:
            p1 = self.frames_landmarks[f_indices[0]][j_idx]
            p5 = self.frames_landmarks[f_indices[4]][j_idx]
            net_dist = math.sqrt((p5[0] - p1[0]) ** 2 + (p5[1] - p1[1]) ** 2) / win_l_hand
            disp_by_joint[j_idx] = net_dist

        wrist_disp = disp_by_joint[WRIST_INDEX]
        mcp_disps = [disp_by_joint[idx] for idx in [5, 9, 13, 17]]
        max_mcp_disp = max(mcp_disps) if mcp_disps else 0.0
        max_stationary_disp = max(wrist_disp, max_mcp_disp)

        return {
            "win_l_hand": win_l_hand,
            "wrist_disp": wrist_disp,
            "index_mcp_disp": disp_by_joint[5],
            "middle_mcp_disp": disp_by_joint[9],
            "ring_mcp_disp": disp_by_joint[13],
            "pinky_mcp_disp": disp_by_joint[17],
            "max_mcp_disp": max_mcp_disp,
            "max_stationary_disp": max_stationary_disp,
        }

    def recompute_all(self):
        """Recomputes metrics and evaluates displacement threshold pass/fail for all windows."""
        if not self.windows:
            return

        total = len(self.windows)
        kept_count = 0
        disp_dropped = 0

        for win in self.windows:
            metrics = self.compute_window_metrics(win)
            win["metrics"] = metrics

            disp_fail = metrics["max_stationary_disp"] > self.disp_threshold
            win["is_moving"] = disp_fail

            if disp_fail:
                disp_dropped += 1
            else:
                kept_count += 1

        moving_count = total - kept_count
        kept_pct = (kept_count / total * 100.0) if total > 0 else 0.0
        moving_pct = (moving_count / total * 100.0) if total > 0 else 0.0

        self.lbl_stats_summary.configure(
            text=f"Total Windows Analyzed: {total}\n"
                 f"Stationary (Kept): {kept_count} ({kept_pct:.1f}%)\n"
                 f"Hand Moving (Dropped): {moving_count} ({moving_pct:.1f}%)\n"
                 f"  • Filtered by Displacement Threshold: {disp_dropped}"
        )

    # ── Slider & Navigation Callbacks ────────────────────────────────────────

    def _on_disp_slider(self, val):
        self.disp_threshold = round(val, 3)
        self.lbl_disp_thresh_val.configure(text=f"{self.disp_threshold:.3f} L_hand")
        self.recompute_all()
        self.render_current_view()

    def _on_window_slider(self, val):
        self.set_window(int(round(val)))

    def _select_sub_frame(self, sub_idx: int):
        self.current_frame_in_window = sub_idx
        for i, btn in enumerate(self.sub_frame_btns):
            btn.configure(fg_color=BTN_PRI if i == sub_idx else BTN_SEC)
        self.render_current_view()

    def prev_frame_in_window(self):
        new_idx = max(0, self.current_frame_in_window - 1)
        self._select_sub_frame(new_idx)

    def next_frame_in_window(self):
        new_idx = min(4, self.current_frame_in_window + 1)
        self._select_sub_frame(new_idx)

    def set_window(self, win_idx: int, reset_to_frame1: bool = True):
        if not self.windows:
            return
        self.current_window_idx = max(0, min(len(self.windows) - 1, win_idx))
        if reset_to_frame1:
            self.current_frame_in_window = 0
            for i, btn in enumerate(self.sub_frame_btns):
                btn.configure(fg_color=BTN_PRI if i == 0 else BTN_SEC)
        self.slider_window.set(self.current_window_idx)
        self.lbl_window_counter.configure(
            text=f"Window: {self.current_window_idx + 1} / {len(self.windows)}"
        )
        self.render_current_view()

    def prev_window(self):
        self.set_window(self.current_window_idx - 1)

    def next_window(self):
        self.set_window(self.current_window_idx + 1)

    def jump_to_window(self, target_is_moving: bool):
        if not self.windows:
            return
        start = self.current_window_idx + 1
        for i in range(start, len(self.windows)):
            if self.windows[i].get("is_moving", False) == target_is_moving:
                self.set_window(i)
                return
        for i in range(0, start):
            if self.windows[i].get("is_moving", False) == target_is_moving:
                self.set_window(i)
                return

    def toggle_play(self):
        self.is_playing = not self.is_playing
        if self.is_playing:
            self.btn_play.configure(text="⏸ Pause", fg_color=AMBER)
            self._play_loop()
        else:
            self.btn_play.configure(text="▶ Play", fg_color=BTN_PRI)
            if self.play_job:
                self.after_cancel(self.play_job)
                self.play_job = None

    def _play_loop(self):
        if not self.is_playing:
            return
        if self.current_window_idx < len(self.windows) - 1:
            self.next_window()
            self.play_job = self.after(250, self._play_loop)
        else:
            self.toggle_play()

    # ── Rendering & Visual Overlay ───────────────────────────────────────────

    def render_current_view(self):
        if not self.windows or self.current_window_idx >= len(self.windows):
            return

        win = self.windows[self.current_window_idx]
        metrics = win.get("metrics")
        if not metrics:
            metrics = self.compute_window_metrics(win)
            win["metrics"] = metrics

        f_indices = win["frame_indices"]
        active_f_idx = f_indices[self.current_frame_in_window]

        if active_f_idx >= len(self.frames_raw_images):
            return

        frame = self.frames_raw_images[active_f_idx].copy()
        pts = self.frames_landmarks[active_f_idx]

        # Draw hand skeleton
        if pts and any(p[0] > 0 for p in pts):
            for i1, i2 in HAND_CONNECTIONS:
                p1 = (int(pts[i1][0]), int(pts[i1][1]))
                p2 = (int(pts[i2][0]), int(pts[i2][1]))
                if p1 != (0, 0) and p2 != (0, 0):
                    cv2.line(frame, p1, p2, (80, 80, 80), 2, cv2.LINE_AA)

            # Draw non-stationary joints (fingers) in gold
            for idx, p in enumerate(pts):
                if idx not in STATIONARY_INDICES:
                    px, py = int(p[0]), int(p[1])
                    if px > 0 and py > 0:
                        cv2.circle(frame, (px, py), 4, (240, 200, 78), -1, cv2.LINE_AA)

            # Draw motion trails strictly from Frame 1 of THIS window to the currently displayed sub-frame
            # When viewing Frame 1 (current_frame_in_window == 0), NO trail is drawn because Frame 1 is the fresh starting baseline
            if self.var_show_trails.get() and self.current_frame_in_window > 0:
                pts_f1 = self.frames_landmarks[f_indices[0]]
                pts_cur = self.frames_landmarks[active_f_idx]

                for st_idx in STATIONARY_INDICES:
                    p_start = (int(pts_f1[st_idx][0]), int(pts_f1[st_idx][1]))
                    p_end = (int(pts_cur[st_idx][0]), int(pts_cur[st_idx][1]))

                    if p_start != (0, 0) and p_end != (0, 0) and p_start != p_end:
                        trail_color = (0, 0, 255) if win.get("is_moving") else (0, 255, 120)
                        cv2.arrowedLine(frame, p_start, p_end, trail_color, 2, tipLength=0.25)
                        cv2.circle(frame, p_start, 3, (200, 200, 200), -1)

            # Highlight Stationary Landmarks (Wrist and MCP knuckles)
            for st_idx in STATIONARY_INDICES:
                px, py = int(pts[st_idx][0]), int(pts[st_idx][1])
                if px > 0 and py > 0:
                    st_color = (0, 0, 255) if win.get("is_moving") else (0, 255, 0)
                    cv2.circle(frame, (px, py), 7, st_color, -1, cv2.LINE_AA)
                    cv2.circle(frame, (px, py), 9, (255, 255, 255), 1, cv2.LINE_AA)

                    label_name = STATIONARY_NAMES[st_idx]
                    cv2.putText(
                        frame, label_name, (px + 10, py - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA
                    )

        # On-screen HUD banner
        status_text = "HAND MOVING [DROPPED]" if win.get("is_moving") else "STATIONARY [KEPT]"
        status_bgr = (0, 0, 220) if win.get("is_moving") else (0, 180, 0)
        cv2.rectangle(frame, (10, 10), (320, 50), (20, 20, 20), -1)
        cv2.rectangle(frame, (10, 10), (320, 50), status_bgr, 2)
        cv2.putText(
            frame, status_text, (20, 36),
            cv2.FONT_HERSHEY_SIMPLEX, 0.65, status_bgr, 2, cv2.LINE_AA
        )

        if self.current_frame_in_window == 0:
            frame_info = f"Window {self.current_window_idx + 1} | Frame 1/5 [Fresh Baseline] (Video Frame #{active_f_idx})"
        else:
            frame_info = f"Window {self.current_window_idx + 1} | Frame {self.current_frame_in_window + 1}/5 (Video Frame #{active_f_idx})"
        cv2.putText(
            frame, frame_info, (14, frame.shape[0] - 14),
            cv2.FONT_HERSHEY_SIMPLEX, 0.50, (220, 220, 220), 1, cv2.LINE_AA
        )

        # Resize for display
        disp_w = max(400, self.video_canvas.winfo_width() - 24)
        disp_h = max(300, self.video_canvas.winfo_height() - 24)
        scale = min(disp_w / frame.shape[1], disp_h / frame.shape[0])
        nw = int(frame.shape[1] * scale)
        nh = int(frame.shape[0] * scale)

        resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA)
        rgb_img = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb_img)
        ctk_img = ctk.CTkImage(light_image=pil_img, dark_image=pil_img, size=(nw, nh))

        self.video_canvas.configure(image=ctk_img)
        self.video_canvas.image = ctk_img

        # Update Metrics Text Readouts
        w_disp = metrics["wrist_disp"]
        m_disp = metrics["max_mcp_disp"]
        max_s_disp = metrics["max_stationary_disp"]
        disp_pass = max_s_disp <= self.disp_threshold

        self.lbl_disp_metric.configure(
            text=f"Wrist Disp: {w_disp:.4f} | Max MCP Disp: {m_disp:.4f} | Peak: {max_s_disp:.4f}"
        )
        if disp_pass:
            self.lbl_disp_status.configure(
                text=f"✓ PASSED: Stationary (Peak {max_s_disp:.4f} <= {self.disp_threshold:.3f} L_hand)",
                text_color=GREEN
            )
        else:
            self.lbl_disp_status.configure(
                text=f"✗ FAILED: Moving (Peak {max_s_disp:.4f} > {self.disp_threshold:.3f} L_hand)",
                text_color=RED
            )

        # Update Detailed Joint Breakdown
        i_mcp = metrics["index_mcp_disp"]
        m_mcp = metrics["middle_mcp_disp"]
        r_mcp = metrics["ring_mcp_disp"]
        p_mcp = metrics["pinky_mcp_disp"]
        self.lbl_joint_breakdown.configure(
            text=f"Wrist: {w_disp:.4f} | Index: {i_mcp:.4f} | Middle: {m_mcp:.4f}\n"
                 f"Ring: {r_mcp:.4f} | Pinky: {p_mcp:.4f}"
        )

        # Update Decision Badge
        if win.get("is_moving"):
            self.box_decision.configure(fg_color=FAIL_BG, border_color=RED)
            self.lbl_decision_main.configure(
                text="❌ HAND MOVING (DISCARD WINDOW)", text_color=RED
            )
            self.lbl_decision_sub.configure(
                text=f"Displacement exceeded threshold ({max_s_disp:.4f} > {self.disp_threshold:.3f} L_hand)",
                text_color=TXT_PRI
            )
        else:
            self.box_decision.configure(fg_color=PASS_BG, border_color=GREEN)
            self.lbl_decision_main.configure(
                text="✅ HAND STATIONARY (PROCESS FOR TOUCH)", text_color=GREEN
            )
            self.lbl_decision_sub.configure(
                text="Wrist & MCP knuckles are stationary. Safe to pass to model.",
                text_color=TXT_PRI
            )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Interactive Hand Movement Displacement Threshold Analyzer for 12 FPS Videos"
    )
    parser.add_argument(
        "-v", "--video", default="", help="Path to 12 FPS video file"
    )
    parser.add_argument(
        "-c", "--csv", default="", help="Path to raw landmarks CSV file (optional, auto-detected if empty)"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    app = HandMovementAnalyzerApp(initial_video=args.video)
    if args.csv and os.path.exists(args.csv):
        app.load_landmarks_from_csv(args.csv)
    app.mainloop()


if __name__ == "__main__":
    main()
