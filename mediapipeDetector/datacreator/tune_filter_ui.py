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
datacreator/tune_filter.py

Interactive Visual Parameter Tuning GUI for 1€ (One Euro) Landmark Filtering.
Renders side-by-side / overlay comparison of Raw MediaPipe landmarks (Red) vs
1€ Filtered landmarks (Green) on video frames in real-time.

Allows live adjustment of min_cutoff, beta, and d_cutoff parameters to visually find
the optimal balance between jitter removal and motion responsiveness.
"""

import argparse
import csv
import math
import os
import sys
import threading
import time
from pathlib import Path
from tkinter import filedialog, messagebox

import cv2
import customtkinter as ctk
from PIL import Image

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

# 21 Hand Landmark Connections for Skeleton Drawing
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),          # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),          # Index
    (5, 9), (9, 10), (10, 11), (11, 12),     # Middle
    (9, 13), (13, 14), (14, 15), (15, 16),   # Ring
    (13, 17), (17, 18), (18, 19), (19, 20),  # Pinky
    (0, 17),                                  # Palm base
]

ALL_21_LANDMARK_NAMES = [
    "wrist",
    "thumb_cmc", "thumb_mcp", "thumb_ip", "thumb_tip",
    "index_mcp", "index_pip", "index_dip", "index_tip",
    "middle_mcp", "middle_pip", "middle_dip", "middle_tip",
    "ring_mcp", "ring_pip", "ring_dip", "ring_tip",
    "pinky_mcp", "pinky_pip", "pinky_dip", "pinky_tip",
]


class OneEuroFilter1D:
    def __init__(self, t0: float, x0: float, min_cutoff: float = 1.5, beta: float = 1.0, d_cutoff: float = 1.0):
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self.x_prev = float(x0)
        self.dx_prev = 0.0
        self.t_prev = float(t0)

    def _smoothing_factor(self, t_elapsed: float, cutoff: float) -> float:
        r = 2.0 * math.pi * cutoff * t_elapsed
        return r / (r + 1.0)

    def _exponential_smoothing(self, alpha: float, x: float, x_prev: float) -> float:
        return alpha * x + (1.0 - alpha) * x_prev

    def filter(self, t: float, x: float) -> float:
        t_elapsed = t - self.t_prev
        if t_elapsed <= 0:
            return self.x_prev

        a_d = self._smoothing_factor(t_elapsed, self.d_cutoff)
        dx = (x - self.x_prev) / t_elapsed
        dx_hat = self._exponential_smoothing(a_d, dx, self.dx_prev)

        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = self._smoothing_factor(t_elapsed, cutoff)
        x_hat = self._exponential_smoothing(a, x, self.x_prev)

        self.x_prev, self.dx_prev, self.t_prev = x_hat, dx_hat, t
        return x_hat


class FilterTunerApp(ctk.CTk):
    def __init__(self, video_path: str = "", csv_path: str = ""):
        super().__init__()

        self.title("🎛️ 1€ Filter Parameter Visual Tuner")
        self.geometry("1280x880")
        self.configure(fg_color=BG)

        self.video_path = video_path
        self.csv_path = csv_path
        self.cap = None
        self.raw_rows = []
        self.filtered_rows = []
        self.headers = []

        self.current_frame_idx = 0
        self.is_playing = False
        self.play_job = None

        # Filter Parameters State
        self.min_cutoff_var = ctk.DoubleVar(value=1.5)
        self.beta_var = ctk.DoubleVar(value=1.0)
        self.d_cutoff_var = ctk.DoubleVar(value=1.0)

        # Display Toggles
        self.show_raw_var = ctk.BooleanVar(value=True)
        self.show_filtered_var = ctk.BooleanVar(value=True)

        self._build_ui()
        self._bind_shortcuts()

        if video_path and csv_path and os.path.exists(video_path) and os.path.exists(csv_path):
            self.load_session(video_path, csv_path)

    def _build_ui(self):
        # ── Top Bar ─────────────────────────────────────────────────────────────
        top_bar = ctk.CTkFrame(self, height=48, corner_radius=0, fg_color=HDR_BG)
        top_bar.pack(fill="x", side="top")
        top_bar.pack_propagate(False)

        ctk.CTkLabel(
            top_bar, text="🎛️ 1€ Filter Visual Tuner",
            font=ctk.CTkFont(family=FF, size=16, weight="bold"), text_color=TXT_PRI
        ).pack(side="left", padx=16)

        ctk.CTkButton(
            top_bar, text="📁 Open Files...", width=120, height=32,
            fg_color=BTN_SEC, hover_color="#4f5258",
            font=ctk.CTkFont(family=FF, size=12),
            command=self._browse_files
        ).pack(side="right", padx=(6, 16))

        ctk.CTkButton(
            top_bar, text="💾 Export Filtered CSV", width=160, height=32,
            fg_color="#2e7d32", hover_color="#1b5e20",
            font=ctk.CTkFont(family=FF, size=12, weight="bold"),
            command=self.export_filtered_csv
        ).pack(side="right", padx=6)

        # ── Main Content Container ──────────────────────────────────────────────
        main_content = ctk.CTkFrame(self, fg_color=BG)
        main_content.pack(fill="both", expand=True, padx=12, pady=12)

        # Left Column: Video Player Canvas
        left_col = ctk.CTkFrame(main_content, fg_color=PANEL, corner_radius=6, border_width=1, border_color=BORDER)
        left_col.pack(side="left", fill="both", expand=True, padx=(0, 8))

        self.video_label = ctk.CTkLabel(left_col, text="Please load Video and Raw Landmarks CSV...", text_color=TXT_MUT)
        self.video_label.pack(fill="both", expand=True, padx=12, pady=12)

        # Video Player Controls
        ctrl_bar = ctk.CTkFrame(left_col, fg_color="transparent")
        ctrl_bar.pack(fill="x", padx=12, pady=(0, 12))

        self.btn_prev = ctk.CTkButton(ctrl_bar, text="Step ◄", width=80, height=32, fg_color=BTN_SEC, command=self.prev_frame)
        self.btn_prev.pack(side="left", padx=4)

        self.btn_play = ctk.CTkButton(
            ctrl_bar, text="▶ Play (Space)", width=130, height=32, fg_color=BTN_PRI, hover_color=BTN_HVP,
            font=ctk.CTkFont(family=FF, size=12, weight="bold"),
            command=self.toggle_play
        )
        self.btn_play.pack(side="left", padx=4)

        self.btn_next = ctk.CTkButton(ctrl_bar, text="Step ►", width=80, height=32, fg_color=BTN_SEC, command=self.next_frame)
        self.btn_next.pack(side="left", padx=4)

        self.slider_frame = ctk.CTkSlider(ctrl_bar, from_=0, to=100, number_of_steps=100, command=self._on_slider_move)
        self.slider_frame.pack(side="left", fill="x", expand=True, padx=12)

        self.lbl_frame_info = ctk.CTkLabel(ctrl_bar, text="Frame: 0 / 0", font=ctk.CTkFont(family=FF, size=12, weight="bold"), text_color=GREEN)
        self.lbl_frame_info.pack(side="right", padx=6)

        # Right Column: Filter Control Panel
        right_col = ctk.CTkFrame(main_content, width=380, fg_color=PANEL, corner_radius=6, border_width=1, border_color=BORDER)
        right_col.pack(side="right", fill="y", padx=(4, 0))
        right_col.pack_propagate(False)

        ctk.CTkLabel(
            right_col, text="⚙  1€ Filter Tuning Parameters",
            font=ctk.CTkFont(family=FF, size=16, weight="bold"), text_color=TXT_PRI
        ).pack(anchor="w", padx=16, pady=(16, 12))

        # Slider 1: min_cutoff
        c1 = ctk.CTkFrame(right_col, fg_color="#2d2d2d", corner_radius=6, border_width=1, border_color=BORDER)
        c1.pack(fill="x", padx=16, pady=6)
        hdr1 = ctk.CTkFrame(c1, fg_color="transparent")
        hdr1.pack(fill="x", padx=12, pady=(10, 2))
        ctk.CTkLabel(hdr1, text="min_cutoff (Hz):", font=ctk.CTkFont(family=FF, size=13, weight="bold"), text_color=TXT_PRI).pack(side="left")
        self.lbl_min_val = ctk.CTkLabel(hdr1, text="1.50 Hz", font=ctk.CTkFont(family=FF, size=13, weight="bold"), text_color=GREEN)
        self.lbl_min_val.pack(side="right")
        ctk.CTkLabel(
            c1, text="Minimum cutoff frequency for still hand.\nLower (0.5-1.5 Hz) = smooths idle jitter.",
            font=ctk.CTkFont(family=FF, size=11), text_color=TXT_MUT, justify="left", anchor="w"
        ).pack(fill="x", padx=12, pady=(0, 6))
        self.slider_min = ctk.CTkSlider(c1, from_=0.05, to=5.0, number_of_steps=99, command=self._on_param_change)
        self.slider_min.set(1.5)
        self.slider_min.pack(fill="x", padx=12, pady=(0, 12))

        # Slider 2: beta
        c2 = ctk.CTkFrame(right_col, fg_color="#2d2d2d", corner_radius=6, border_width=1, border_color=BORDER)
        c2.pack(fill="x", padx=16, pady=6)
        hdr2 = ctk.CTkFrame(c2, fg_color="transparent")
        hdr2.pack(fill="x", padx=12, pady=(10, 2))
        ctk.CTkLabel(hdr2, text="beta (Speed Slope):", font=ctk.CTkFont(family=FF, size=13, weight="bold"), text_color=TXT_PRI).pack(side="left")
        self.lbl_beta_val = ctk.CTkLabel(hdr2, text="1.00", font=ctk.CTkFont(family=FF, size=13, weight="bold"), text_color=GREEN)
        self.lbl_beta_val.pack(side="right")
        ctk.CTkLabel(
            c2, text="Speed responsiveness coefficient.\nHigher (1.0-2.0) = zero lag during fast motion.",
            font=ctk.CTkFont(family=FF, size=11), text_color=TXT_MUT, justify="left", anchor="w"
        ).pack(fill="x", padx=12, pady=(0, 6))
        self.slider_beta = ctk.CTkSlider(c2, from_=0.0, to=5.0, number_of_steps=100, command=self._on_param_change)
        self.slider_beta.set(1.0)
        self.slider_beta.pack(fill="x", padx=12, pady=(0, 12))

        # Slider 3: d_cutoff
        c3 = ctk.CTkFrame(right_col, fg_color="#2d2d2d", corner_radius=6, border_width=1, border_color=BORDER)
        c3.pack(fill="x", padx=16, pady=6)
        hdr3 = ctk.CTkFrame(c3, fg_color="transparent")
        hdr3.pack(fill="x", padx=12, pady=(10, 2))
        ctk.CTkLabel(hdr3, text="d_cutoff (Velocity Hz):", font=ctk.CTkFont(family=FF, size=13, weight="bold"), text_color=TXT_PRI).pack(side="left")
        self.lbl_d_val = ctk.CTkLabel(hdr3, text="1.00 Hz", font=ctk.CTkFont(family=FF, size=13, weight="bold"), text_color=GREEN)
        self.lbl_d_val.pack(side="right")
        ctk.CTkLabel(
            c3, text="Velocity derivative cutoff frequency.\nStandard = 1.0 Hz (stabilizes adaptive speed).",
            font=ctk.CTkFont(family=FF, size=11), text_color=TXT_MUT, justify="left", anchor="w"
        ).pack(fill="x", padx=12, pady=(0, 6))
        self.slider_d = ctk.CTkSlider(c3, from_=0.1, to=5.0, number_of_steps=98, command=self._on_param_change)
        self.slider_d.set(1.0)
        self.slider_d.pack(fill="x", padx=12, pady=(0, 12))

        # Preset Buttons
        p_row = ctk.CTkFrame(right_col, fg_color="transparent")
        p_row.pack(fill="x", padx=16, pady=8)
        ctk.CTkButton(p_row, text="Default (1.5, 1.0)", width=110, fg_color=BTN_SEC, command=lambda: self.set_preset(1.5, 1.0, 1.0)).pack(side="left", padx=2)
        ctk.CTkButton(p_row, text="High Smooth (0.5, 0.3)", width=110, fg_color=BTN_SEC, command=lambda: self.set_preset(0.5, 0.3, 1.0)).pack(side="left", padx=2)
        ctk.CTkButton(p_row, text="Fast Motion (2.5, 2.0)", width=110, fg_color=BTN_SEC, command=lambda: self.set_preset(2.5, 2.0, 1.0)).pack(side="left", padx=2)

        # Skeleton Overlay Toggles & Legend
        ctk.CTkFrame(right_col, height=1, fg_color=BORDER).pack(fill="x", padx=16, pady=10)

        ctk.CTkLabel(right_col, text="🎨 Visual Skeleton Legend & Toggles", font=ctk.CTkFont(family=FF, size=13, weight="bold"), text_color=TXT_SEC).pack(anchor="w", padx=16, pady=(0, 6))

        ctk.CTkCheckBox(
            right_col, text="🔴 Raw MediaPipe Skeleton (Red Jitter)", variable=self.show_raw_var,
            text_color="#ff5252", font=ctk.CTkFont(family=FF, size=12), command=self._refresh_frame
        ).pack(anchor="w", padx=16, pady=4)

        ctk.CTkCheckBox(
            right_col, text="🟢 1€ Filtered Skeleton (Emerald Smooth)", variable=self.show_filtered_var,
            text_color=GREEN, font=ctk.CTkFont(family=FF, size=12), command=self._refresh_frame
        ).pack(anchor="w", padx=16, pady=4)

        # File Metadata Info Box
        ctk.CTkFrame(right_col, height=1, fg_color=BORDER).pack(fill="x", padx=16, pady=10)
        self.lbl_metadata = ctk.CTkLabel(
            right_col, text="Metadata info...",
            font=ctk.CTkFont(family=FF, size=11), text_color=TXT_MUT,
            justify="left", anchor="w", wraplength=340
        )
        self.lbl_metadata.pack(fill="both", expand=True, padx=16, pady=(0, 12))

    def _bind_shortcuts(self):
        self.bind("<KeyPress>", self._on_key_press)

    def _on_key_press(self, event):
        key = event.keysym.lower()
        char = event.char.lower() if event.char else ""

        if key == "space":
            self.toggle_play()
        elif key == "left" or char == ",":
            self.prev_frame()
        elif key == "right" or char == ".":
            self.next_frame()

    def _browse_files(self):
        v = open_video_dialog()
        if not v:
            return
        c = open_csv_dialog()
        if not c:
            return
        self.load_session(v, c)

    def load_session(self, video_path: str, csv_path: str):
        if not os.path.exists(video_path) or not os.path.exists(csv_path):
            messagebox.showerror("Error", "Selected Video or CSV file does not exist!")
            return

        self.video_path = video_path
        self.csv_path = csv_path

        if self.cap:
            self.cap.release()

        self.cap = cv2.VideoCapture(video_path)
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            self.headers = reader.fieldnames or []
            self.raw_rows = list(reader)

        total_frames = len(self.raw_rows)
        self.slider_frame.configure(from_=0, to=max(1, total_frames - 1), number_of_steps=max(1, total_frames - 1))

        w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.lbl_metadata.configure(
            text=f"Video File: {os.path.basename(video_path)}\n"
                 f"Resolution: {w}x{h} px\n"
                 f"Total Frames: {total_frames}\n"
                 f"Raw Landmarks CSV: {os.path.basename(csv_path)}"
        )

        self.current_frame_idx = 0
        self.recompute_filtering()
        self.show_frame(0)

    def set_preset(self, min_cutoff: float, beta: float, d_cutoff: float):
        self.slider_min.set(min_cutoff)
        self.slider_beta.set(beta)
        self.slider_d.set(d_cutoff)
        self._on_param_change(None)

    def _on_param_change(self, val):
        min_c = round(self.slider_min.get(), 2)
        beta_c = round(self.slider_beta.get(), 2)
        d_c = round(self.slider_d.get(), 2)

        self.lbl_min_val.configure(text=f"{min_c:.2f} Hz")
        self.lbl_beta_val.configure(text=f"{beta_c:.2f}")
        self.lbl_d_val.configure(text=f"{d_c:.2f} Hz")

        self.recompute_filtering()
        self._refresh_frame()

    def recompute_filtering(self):
        """Re-filters all video frames in memory instantly when parameters change."""
        if not self.raw_rows:
            return

        min_c = self.slider_min.get()
        beta_c = self.slider_beta.get()
        d_c = self.slider_d.get()

        filters = {}
        self.filtered_rows = []

        for row in self.raw_rows:
            out_row = dict(row)
            t_sec = float(row.get("timestamp_ms", "0")) / 1000.0
            hand_type = row.get("hand", "None")

            if hand_type == "None":
                filters.clear()
            else:
                for lm_name in ALL_21_LANDMARK_NAMES:
                    x_col = f"{lm_name}_x"
                    y_col = f"{lm_name}_y"
                    raw_x = float(row.get(x_col, 0.0))
                    raw_y = float(row.get(y_col, 0.0))

                    if raw_x == 0.0 and raw_y == 0.0:
                        continue

                    if lm_name not in filters:
                        fx = OneEuroFilter1D(t_sec, raw_x, min_cutoff=min_c, beta=beta_c, d_cutoff=d_c)
                        fy = OneEuroFilter1D(t_sec, raw_y, min_cutoff=min_c, beta=beta_c, d_cutoff=d_c)
                        filters[lm_name] = (fx, fy)
                        out_row[x_col] = raw_x
                        out_row[y_col] = raw_y
                    else:
                        fx, fy = filters[lm_name]
                        out_row[x_col] = fx.filter(t_sec, raw_x)
                        out_row[y_col] = fy.filter(t_sec, raw_y)

            self.filtered_rows.append(out_row)

    def _refresh_frame(self):
        self.show_frame(self.current_frame_idx)

    def show_frame(self, frame_idx: int):
        if not self.cap or not self.raw_rows:
            return

        frame_idx = max(0, min(frame_idx, len(self.raw_rows) - 1))
        self.current_frame_idx = frame_idx

        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = self.cap.read()
        if not ret:
            return

        raw_rec = self.raw_rows[frame_idx]
        filt_rec = self.filtered_rows[frame_idx] if frame_idx < len(self.filtered_rows) else raw_rec

        h, w, _ = frame.shape

        # Render RAW Skeleton Overlay (Red)
        if self.show_raw_var.get():
            frame = self._draw_skeleton(frame, raw_rec, color_bgr=(50, 50, 255), label="RAW MediaPipe (Jitter)", dot_radius=5)

        # Render FILTERED Skeleton Overlay (Emerald Green)
        if self.show_filtered_var.get():
            frame = self._draw_skeleton(frame, filt_rec, color_bgr=(240, 201, 78), label="1€ FILTERED (Smooth)", dot_radius=7)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        ctk_img = ctk.CTkImage(light_image=pil_img, dark_image=pil_img, size=(640, 480))

        self.video_label.configure(image=ctk_img, text="")
        self.slider_frame.set(frame_idx)
        self.lbl_frame_info.configure(text=f"Frame: {frame_idx + 1} / {len(self.raw_rows)} ({raw_rec.get('timestamp_ms', '0')} ms)")

    def _draw_skeleton(self, frame: cv2.Mat, record: dict, color_bgr: tuple, label: str, dot_radius: int) -> cv2.Mat:
        h, w, _ = frame.shape
        hand_type = record.get("hand", "None")

        if hand_type == "None":
            return frame

        pts_px = []
        for lm_name in ALL_21_LANDMARK_NAMES:
            nx = float(record.get(f"{lm_name}_x", 0.0))
            ny = float(record.get(f"{lm_name}_y", 0.0))
            pts_px.append((int(round(nx * w)), int(round(ny * h))))

        # Draw Connections
        for a, b in HAND_CONNECTIONS:
            cv2.line(frame, pts_px[a], pts_px[b], color_bgr, 2)

        # Draw Joint Circles
        for pt in pts_px:
            cv2.circle(frame, pt, dot_radius, color_bgr, -1)

        return frame

    def prev_frame(self):
        self._stop_play()
        if self.current_frame_idx > 0:
            self.show_frame(self.current_frame_idx - 1)

    def next_frame(self):
        self._stop_play()
        if self.current_frame_idx < len(self.raw_rows) - 1:
            self.show_frame(self.current_frame_idx + 1)

    def _on_slider_move(self, val):
        self._stop_play()
        self.show_frame(int(val))

    def toggle_play(self):
        if self.is_playing:
            self._stop_play()
        else:
            self._start_play()

    def _start_play(self):
        self.is_playing = True
        self.btn_play.configure(text="⏸ Pause (Space)", fg_color="#d32f2f")
        self._tick_play()

    def _stop_play(self):
        self.is_playing = False
        if self.play_job:
            self.after_cancel(self.play_job)
            self.play_job = None
        self.btn_play.configure(text="▶ Play (Space)", fg_color=BTN_PRI)

    def _tick_play(self):
        if not self.is_playing or not self.raw_rows:
            return

        if self.current_frame_idx < len(self.raw_rows) - 1:
            self.show_frame(self.current_frame_idx + 1)
            self.play_job = self.after(83, self._tick_play)
        else:
            self._stop_play()

    def export_filtered_csv(self):
        if not self.filtered_rows or not self.csv_path:
            messagebox.showerror("Error", "No landmark data loaded to export!")
            return

        dir_name = os.path.dirname(os.path.abspath(self.csv_path))
        base_name = os.path.basename(self.csv_path)

        if ".raw_landmarks." in base_name:
            out_name = base_name.replace(".raw_landmarks.", ".filtered_landmarks.")
        else:
            name_no_ext = os.path.splitext(base_name)[0]
            out_name = f"{name_no_ext}.filtered_landmarks.csv"

        out_path = os.path.join(dir_name, out_name)

        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.headers)
            writer.writeheader()
            writer.writerows(self.filtered_rows)

        min_c = round(self.slider_min.get(), 2)
        beta_c = round(self.slider_beta.get(), 2)
        d_c = round(self.slider_d.get(), 2)

        messagebox.showinfo(
            "Export Successful",
            f"Successfully exported 1€ filtered landmarks to:\n{out_path}\n\n"
            f"Tuned Parameters:\n"
            f"• min_cutoff = {min_c} Hz\n"
            f"• beta = {beta_c}\n"
            f"• d_cutoff = {d_c} Hz"
        )


def main():
    parser = argparse.ArgumentParser(description="Interactive Visual Parameter Tuning GUI for 1€ Landmark Filter")
    parser.add_argument("-v", "--video", default="", help="Path to 12 FPS video file")
    parser.add_argument("-c", "--csv", default="", help="Path to raw landmarks CSV file")

    args = parser.parse_args()

    app = FilterTunerApp(video_path=args.video, csv_path=args.csv)
    app.mainloop()


if __name__ == "__main__":
    main()
