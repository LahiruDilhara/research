# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "customtkinter>=6.0.0",
#     "mediapipe>=1.0.0",
#     "numpy>=2.5.2",
#     "opencv-python>=5.0.0.93",
#     "pillow",
# ]
# ///

"""
datacreator/hand_movement_analyzer_ui.py

Interactive Visual Analyzer for calibrating Hand Movement & Velocity Thresholds.
Analyzes 12 FPS video frames, constructs 5-frame sliding windows, and provides
real-time interactive controls to tune:
  1. Coordinate-based displacement threshold (relative to rigid palm scale L_hand)
  2. Velocity threshold for stationary palm joints (hand-lengths per second)

Stationary reference landmarks:
  - Wrist (Joint 0)
  - Index MCP (Joint 5)
  - Middle MCP (Joint 9)
  - Ring MCP (Joint 13)
  - Pinky MCP (Joint 17)
"""

import argparse
import csv
import math
import os
import sys
import threading
import time
from pathlib import Path
from tkinter import messagebox

import cv2
import customtkinter as ctk
import mediapipe as mp
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

# Landmark Indices
WRIST_INDEX = 0
STATIONARY_INDICES = [0, 5, 9, 13, 17]  # Wrist, Index MCP, Middle MCP, Ring MCP, Pinky MCP
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


class HandMovementAnalyzerApp(ctk.CTk):
    def __init__(self, initial_video: str = ""):
        super().__init__()

        self.title("Hand Movement & Velocity Threshold Analyzer (12 FPS)")
        self.geometry("1400x900")
        self.minsize(1200, 800)
        ctk.set_appearance_mode("Dark")
        ctk.set_default_color_theme("blue")

        # Video State
        self.video_path = initial_video
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
        self.current_frame_in_window = 4  # 0..4, default to last frame in window

        # Thresholds (Default values)
        self.disp_threshold = 0.100   # in unitless hand-lengths
        self.vel_threshold = 1.00     # in hand-lengths per second

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
            top_bar, text="📄 Load Raw CSV...", width=140, height=32,
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
                fg_color=BTN_PRI if f_idx == 4 else BTN_SEC,
                command=lambda i=f_idx: self._select_sub_frame(i)
            )
            btn.pack(side="left", padx=3)
            self.sub_frame_btns.append(btn)

        self.var_show_trails = ctk.BooleanVar(value=True)
        ctk.CTkCheckBox(
            sub_frame_ctrl, text="Show Motion Trails", variable=self.var_show_trails,
            font=ctk.CTkFont(family=FF, size=11), command=self.render_current_view
        ).pack(side="right", padx=12)

        # Right Column: Controls, Sliders & Metrics
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

        # Jump buttons
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

        # ── Section 2: Threshold 1 — Coordinate Displacement ─────────────────
        sec2 = ctk.CTkFrame(right_col, fg_color=HDR_BG, corner_radius=6)
        sec2.pack(fill="x", padx=12, pady=6)

        lbl_sec2 = ctk.CTkLabel(
            sec2, text="FILTER 1: COORDINATE DISPLACEMENT (PRE-VELOCITY)",
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
        ctk.CTkLabel(disp_slider_row, text="Threshold:", font=ctk.CTkFont(family=FF, size=11)).pack(side="left")
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

        # ── Section 3: Threshold 2 — Kinematic Velocity ──────────────────────
        sec3 = ctk.CTkFrame(right_col, fg_color=HDR_BG, corner_radius=6)
        sec3.pack(fill="x", padx=12, pady=6)

        lbl_sec3 = ctk.CTkLabel(
            sec3, text="FILTER 2: PEAK VELOCITY (POST-VELOCITY)",
            font=ctk.CTkFont(family=FF, size=11, weight="bold"), text_color=TXT_MUT
        )
        lbl_sec3.pack(anchor="w", padx=10, pady=(6, 2))

        self.lbl_vel_metric = ctk.CTkLabel(
            sec3, text="Wrist Peak Vel: 0.000 | Max MCP Peak Vel: 0.000",
            font=ctk.CTkFont(family=FF, size=11), text_color=TXT_SEC
        )
        self.lbl_vel_metric.pack(anchor="w", padx=10, pady=2)

        vel_slider_row = ctk.CTkFrame(sec3, fg_color="transparent")
        vel_slider_row.pack(fill="x", padx=10, pady=4)
        ctk.CTkLabel(vel_slider_row, text="Threshold:", font=ctk.CTkFont(family=FF, size=11)).pack(side="left")
        self.lbl_vel_thresh_val = ctk.CTkLabel(
            vel_slider_row, text=f"{self.vel_threshold:.2f} L_hand/s",
            font=ctk.CTkFont(family=FF, size=11, weight="bold"), text_color=GREEN
        )
        self.lbl_vel_thresh_val.pack(side="right")

        self.slider_vel = ctk.CTkSlider(
            sec3, from_=0.10, to=3.00, number_of_steps=58, command=self._on_vel_slider
        )
        self.slider_vel.set(self.vel_threshold)
        self.slider_vel.pack(fill="x", padx=10, pady=(0, 6))

        self.lbl_vel_status = ctk.CTkLabel(
            sec3, text="Status: PENDING",
            font=ctk.CTkFont(family=FF, size=11, weight="bold"), text_color=TXT_MUT
        )
        self.lbl_vel_status.pack(anchor="w", padx=10, pady=(0, 8))

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
            sec5, text="GLOBAL PIPELINE IMPACT AUDIT",
            font=ctk.CTkFont(family=FF, size=11, weight="bold"), text_color=TXT_MUT
        )
        lbl_sec5.pack(anchor="w", padx=10, pady=(6, 4))

        self.lbl_stats_summary = ctk.CTkLabel(
            sec5, text="Total Windows: 0\nStationary (Kept): 0 (0.0%)\nHand Moving (Dropped): 0 (0.0%)\n- By Displacement: 0\n- By Velocity: 0",
            font=ctk.CTkFont(family=FF, size=11), justify="left", text_color=TXT_SEC
        )
        self.lbl_stats_summary.pack(anchor="w", padx=10, pady=(0, 8))

    # ── 12 FPS Validation & Video Loading ────────────────────────────────────

    def _browse_video(self):
        v_path = open_video_dialog()
        if v_path:
            self.load_video(v_path)

    def _browse_csv(self):
        c_path = open_csv_dialog()
        if c_path and self.video_path:
            self.load_landmarks_from_csv(c_path)

    def validate_12fps(self, video_path: str) -> tuple[bool, float, int]:
        """Validates whether video is approximately 12 FPS (~11.4 to 12.6 FPS)."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return False, 0.0, 0

        header_fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration_ms = 0.0

        # Read first and last timestamps to get true duration
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

        # Check if actual FPS is within 11.4 to 12.6 FPS
        is_valid = 11.4 <= actual_fps <= 12.6
        return is_valid, actual_fps, count

    def load_video(self, video_path: str):
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
                f"For accurate kinematic velocity normalization, resample to 12 FPS using:\n"
                f"python3 datacreator/resample_12fps.py -i '{video_path}' -o ./resampled/"
            )

        # Check for existing matching raw landmarks CSV
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        csv_candidates = [
            os.path.join(os.path.dirname(video_path), f"{base_name}.raw_landmarks.csv"),
            os.path.join(PROJECT_ROOT, "dataset", f"{base_name}.raw_landmarks.csv"),
            os.path.join(PROJECT_ROOT, "dataprocessing", "1_rawCSVFiles", f"{base_name}.raw_landmarks.csv"),
        ]

        found_csv = None
        for cand in csv_candidates:
            if os.path.exists(cand):
                found_csv = cand
                break

        if found_csv:
            self.load_landmarks_from_csv(found_csv)
        else:
            self.extract_landmarks_with_mediapipe(video_path)

    def load_landmarks_from_csv(self, csv_path: str):
        """Loads landmarks from an existing .raw_landmarks.csv file."""
        if not self.video_path or not os.path.exists(self.video_path):
            return

        cap = cv2.VideoCapture(self.video_path)
        self.video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
        self.video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480

        # Read frames into memory
        self.frames_raw_images = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            self.frames_raw_images.append(frame)
        cap.release()

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

        self.build_sliding_windows()
        self.recompute_all()
        self.set_window(0)

    def extract_landmarks_with_mediapipe(self, video_path: str):
        """Extracts MediaPipe landmarks on the video frames."""
        cap = cv2.VideoCapture(video_path)
        self.video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
        self.video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480

        self.frames_raw_images = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            self.frames_raw_images.append(frame)
        cap.release()

        if not self.frames_raw_images:
            messagebox.showerror("Error", "Could not read frames from video.")
            return

        # Setup MediaPipe HandLandmarker
        import urllib.request
        from mediapipe.tasks.python import BaseOptions
        from mediapipe.tasks.python.vision import (
            HandLandmarker, HandLandmarkerOptions, RunningMode,
        )

        model_path = str(PROJECT_ROOT / "hand_landmarker.task")
        if not os.path.exists(model_path):
            try:
                urllib.request.urlretrieve(
                    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
                    model_path
                )
            except Exception as e:
                messagebox.showerror("Error", f"Failed to download MediaPipe model: {e}")
                return

        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=RunningMode.VIDEO,
            num_hands=1,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        landmarker = HandLandmarker.create_from_options(options)

        self.frames_landmarks = []
        self.palm_scales = []
        w = float(self.video_width)
        h = float(self.video_height)

        try:
            for idx, frame in enumerate(self.frames_raw_images):
                t_ms = int((idx / 12.0) * 1000)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                res = landmarker.detect_for_video(mp_image, t_ms)

                pts = []
                if res.hand_landmarks and len(res.hand_landmarks) > 0:
                    hand = res.hand_landmarks[0]
                    for lm in hand:
                        pts.append((lm.x * w, lm.y * h, lm.z * w))
                else:
                    pts = [(0.0, 0.0, 0.0)] * 21

                self.frames_landmarks.append(pts)
                self.palm_scales.append(calculate_palm_scale(pts))
        finally:
            try:
                landmarker.close()
            except Exception:
                pass

        self.build_sliding_windows()
        self.recompute_all()
        self.set_window(0)

    def build_sliding_windows(self):
        """Builds standard 5-frame sliding windows with 2-frame overlap (step = 3 frames)."""
        self.windows = []
        total_f = len(self.frames_landmarks)
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

    # ── Kinematic Calculations & Recomputation ───────────────────────────────

    def compute_window_metrics(self, win: dict) -> dict:
        """
        Computes stationary displacement and velocity metrics for a 5-frame window:
        - Displacement of wrist & MCPs relative to average L_hand.
        - Velocity (hand-lengths / s) of wrist & MCPs across 4 frame intervals.
        """
        f_indices = win["frame_indices"]
        l_hands = [self.palm_scales[i] for i in f_indices]
        avg_l_hand = max(1.0, sum(l_hands) / len(l_hands))

        # 1. Displacement across window (net travel between frame 1 and frame 5)
        # and max displacement from frame 1
        disp_by_joint = {}
        for j_idx in STATIONARY_INDICES:
            p1 = self.frames_landmarks[f_indices[0]][j_idx]
            p5 = self.frames_landmarks[f_indices[4]][j_idx]
            net_dist = math.sqrt((p5[0] - p1[0]) ** 2 + (p5[1] - p1[1]) ** 2) / avg_l_hand

            # Also check max frame deviation from frame 1
            max_dev = 0.0
            for k in range(1, 5):
                pk = self.frames_landmarks[f_indices[k]][j_idx]
                d_k = math.sqrt((pk[0] - p1[0]) ** 2 + (pk[1] - p1[1]) ** 2) / avg_l_hand
                if d_k > max_dev:
                    max_dev = d_k

            disp_by_joint[j_idx] = {
                "net": net_dist,
                "max_dev": max_dev,
            }

        wrist_disp = disp_by_joint[WRIST_INDEX]["net"]
        mcp_disps = [disp_by_joint[idx]["net"] for idx in [5, 9, 13, 17]]
        max_mcp_disp = max(mcp_disps) if mcp_disps else 0.0
        max_stationary_disp = max(wrist_disp, max_mcp_disp)

        # 2. Kinematic Velocities across 4 intervals (v = 1..4)
        # dt = 1 / 12.0 seconds
        dt = 1.0 / 12.0
        peak_vel_by_joint = {}

        for j_idx in STATIONARY_INDICES:
            max_speed = 0.0
            for k in range(4):
                f_a = f_indices[k]
                f_b = f_indices[k + 1]
                pa = self.frames_landmarks[f_a][j_idx]
                pb = self.frames_landmarks[f_b][j_idx]

                # Velocity in hand-lengths per second
                vx = (pb[0] - pa[0]) / (avg_l_hand * dt)
                vy = (pb[1] - pa[1]) / (avg_l_hand * dt)
                speed = math.sqrt(vx * vx + vy * vy)
                if speed > max_speed:
                    max_speed = speed
            peak_vel_by_joint[j_idx] = max_speed

        wrist_vel = peak_vel_by_joint[WRIST_INDEX]
        mcp_vels = [peak_vel_by_joint[idx] for idx in [5, 9, 13, 17]]
        max_mcp_vel = max(mcp_vels) if mcp_vels else 0.0
        max_stationary_vel = max(wrist_vel, max_mcp_vel)

        return {
            "avg_l_hand": avg_l_hand,
            "wrist_disp": wrist_disp,
            "max_mcp_disp": max_mcp_disp,
            "max_stationary_disp": max_stationary_disp,
            "wrist_vel": wrist_vel,
            "max_mcp_vel": max_mcp_vel,
            "max_stationary_vel": max_stationary_vel,
        }

    def recompute_all(self):
        """Recomputes metrics and evaluates threshold pass/fail for all windows."""
        if not self.windows:
            return

        total = len(self.windows)
        kept_count = 0
        disp_dropped = 0
        vel_dropped = 0

        for win in self.windows:
            metrics = self.compute_window_metrics(win)
            win["metrics"] = metrics

            disp_fail = metrics["max_stationary_disp"] > self.disp_threshold
            vel_fail = metrics["max_stationary_vel"] > self.vel_threshold

            win["disp_fail"] = disp_fail
            win["vel_fail"] = vel_fail
            win["is_moving"] = disp_fail or vel_fail

            if win["is_moving"]:
                if disp_fail:
                    disp_dropped += 1
                if vel_fail:
                    vel_dropped += 1
            else:
                kept_count += 1

        moving_count = total - kept_count
        kept_pct = (kept_count / total * 100.0) if total > 0 else 0.0
        moving_pct = (moving_count / total * 100.0) if total > 0 else 0.0

        self.lbl_stats_summary.configure(
            text=f"Total Windows Analyzed: {total}\n"
                 f"Stationary (Kept): {kept_count} ({kept_pct:.1f}%)\n"
                 f"Hand Moving (Dropped): {moving_count} ({moving_pct:.1f}%)\n"
                 f"  • Flagged by Displacement: {disp_dropped}\n"
                 f"  • Flagged by Velocity: {vel_dropped}"
        )

    # ── Slider Callbacks ─────────────────────────────────────────────────────

    def _on_disp_slider(self, val):
        self.disp_threshold = round(val, 3)
        self.lbl_disp_thresh_val.configure(text=f"{self.disp_threshold:.3f} L_hand")
        self.recompute_all()
        self.render_current_view()

    def _on_vel_slider(self, val):
        self.vel_threshold = round(val, 2)
        self.lbl_vel_thresh_val.configure(text=f"{self.vel_threshold:.2f} L_hand/s")
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

    def set_window(self, win_idx: int):
        if not self.windows:
            return
        self.current_window_idx = max(0, min(len(self.windows) - 1, win_idx))
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
        # Wrap around to beginning
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
            # 12 FPS window step interval is 3 frames = 250 ms
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
        avg_l_hand = metrics["avg_l_hand"]

        # Draw hand skeleton
        if pts and any(p[0] > 0 for p in pts):
            for i1, i2 in HAND_CONNECTIONS:
                p1 = (int(pts[i1][0]), int(pts[i1][1]))
                p2 = (int(pts[i2][0]), int(pts[i2][1]))
                if p1 != (0, 0) and p2 != (0, 0):
                    cv2.line(frame, p1, p2, (80, 80, 80), 2, cv2.LINE_AA)

            # Draw non-stationary joints (fingers) in light cyan
            for idx, p in enumerate(pts):
                if idx not in STATIONARY_INDICES:
                    px, py = int(p[0]), int(p[1])
                    if px > 0 and py > 0:
                        cv2.circle(frame, (px, py), 4, (240, 200, 78), -1, cv2.LINE_AA)

            # Draw motion trails between Frame 1 and Frame 5 if enabled
            if self.var_show_trails.get():
                pts_f1 = self.frames_landmarks[f_indices[0]]
                pts_f5 = self.frames_landmarks[f_indices[4]]

                for st_idx in STATIONARY_INDICES:
                    p_start = (int(pts_f1[st_idx][0]), int(pts_f1[st_idx][1]))
                    p_end = (int(pts_f5[st_idx][0]), int(pts_f5[st_idx][1]))

                    if p_start != (0, 0) and p_end != (0, 0):
                        # Draw displacement arrow/line
                        trail_color = (0, 0, 255) if win.get("is_moving") else (0, 255, 120)
                        cv2.arrowedLine(frame, p_start, p_end, trail_color, 2, tipLength=0.25)
                        cv2.circle(frame, p_start, 3, (200, 200, 200), -1)

            # Highlight Stationary Landmarks (Wrist and MCPs)
            for st_idx in STATIONARY_INDICES:
                px, py = int(pts[st_idx][0]), int(pts[st_idx][1])
                if px > 0 and py > 0:
                    st_color = (0, 0, 255) if win.get("is_moving") else (0, 255, 0)
                    cv2.circle(frame, (px, py), 7, st_color, -1, cv2.LINE_AA)
                    cv2.circle(frame, (px, py), 9, (255, 255, 255), 1, cv2.LINE_AA)

                    # Label
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

        frame_info = f"Window {self.current_window_idx + 1} | Step {self.current_frame_in_window + 1}/5 (Frame #{active_f_idx})"
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
            text=f"Wrist: {w_disp:.4f} | Max MCP: {m_disp:.4f} | Max: {max_s_disp:.4f}"
        )
        if disp_pass:
            self.lbl_disp_status.configure(
                text=f"✓ PASSED (<= {self.disp_threshold:.3f})", text_color=GREEN
            )
        else:
            self.lbl_disp_status.configure(
                text=f"✗ FAILED (> {self.disp_threshold:.3f} L_hand)", text_color=RED
            )

        w_vel = metrics["wrist_vel"]
        m_vel = metrics["max_mcp_vel"]
        max_s_vel = metrics["max_stationary_vel"]
        vel_pass = max_s_vel <= self.vel_threshold

        self.lbl_vel_metric.configure(
            text=f"Wrist: {w_vel:.3f} | Max MCP: {m_vel:.3f} | Max: {max_s_vel:.3f}"
        )
        if vel_pass:
            self.lbl_vel_status.configure(
                text=f"✓ PASSED (<= {self.vel_threshold:.2f})", text_color=GREEN
            )
        else:
            self.lbl_vel_status.configure(
                text=f"✗ FAILED (> {self.vel_threshold:.2f} L_hand/s)", text_color=RED
            )

        # Update Decision Badge
        if win.get("is_moving"):
            reasons = []
            if not disp_pass:
                reasons.append("Displacement Exceeded")
            if not vel_pass:
                reasons.append("Velocity Exceeded")
            reason_str = " & ".join(reasons)

            self.box_decision.configure(fg_color=FAIL_BG, border_color=RED)
            self.lbl_decision_main.configure(
                text="❌ HAND MOVING (DISCARD WINDOW)", text_color=RED
            )
            self.lbl_decision_sub.configure(
                text=f"Window dropped from touch evaluation ({reason_str})", text_color=TXT_PRI
            )
        else:
            self.box_decision.configure(fg_color=PASS_BG, border_color=GREEN)
            self.lbl_decision_main.configure(
                text="✅ HAND STATIONARY (PROCESS FOR TOUCH)", text_color=GREEN
            )
            self.lbl_decision_sub.configure(
                text="Wrist & MCPs stable. Passed forward to neural network.", text_color=TXT_PRI
            )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Interactive Hand Movement & Velocity Threshold Analyzer for 12 FPS Videos"
    )
    parser.add_argument(
        "-v", "--video", default="", help="Path to 12 FPS video file"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    app = HandMovementAnalyzerApp(initial_video=args.video)
    app.mainloop()


if __name__ == "__main__":
    main()
