# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "customtkinter>=6.0.0",
#     "mediapipe>=0.10.14",
#     "numpy>=2.5.2",
#     "opencv-python>=5.0.0.93",
#     "pillow",
# ]
# ///

"""
datacreator/live_tune_filter_ui.py

Live 12 FPS Webcam Interactive Parameter Tuner for 1€ Landmark Filtering.

Replicates exact process.sh dataset preprocessing sequence:
1. 12 FPS temporal sampling rate (1/12.0s frame interval).
2. Per-frame pixel space conversion, 8-distance palm RMS scale normalization (L_hand), and wrist centering (0,0,0).
3. 1€ Filtering applied directly ON scale-normalized coordinates (process.sh Step 3).
4. Reconstructs smooth pixel space for silk-smooth visual skeleton comparison.

Allows tuning min_cutoff, beta, and d_cutoff parameters to export directly to process.sh.
"""

import math
import os
import sys
import threading
import time
from pathlib import Path

import cv2
import customtkinter as ctk
from PIL import Image, ImageTk

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from realtimeprocess.stages.stage1_normalizer import HandScaleNormalizer, WRIST_INDEX, INDEX_MCP_INDEX, MIDDLE_MCP_INDEX, RING_MCP_INDEX, PINKY_MCP_INDEX
from realtimeprocess.stages.stage2_euro_filter import OneEuroFilter1D, OneEuroFilterBank, ALL_21_LANDMARK_NAMES

# Theme Colors (Professional Dark GUI)
BG = "#181818"
PANEL = "#222222"
BORDER = "#333333"
HDR_BG = "#2a2a2a"
TXT_PRI = "#ffffff"
TXT_SEC = "#cccccc"
TXT_MUT = "#888888"
BTN_PRI = "#007acc"
BTN_HVP = "#005999"
BTN_SEC = "#33373b"
CYAN = "#00e5ff"
RED = "#ff4444"
AMBER = "#ffaa00"

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),          # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),          # Index
    (5, 9), (9, 10), (10, 11), (11, 12),     # Middle
    (9, 13), (13, 14), (14, 15), (15, 16),   # Ring
    (13, 17), (17, 18), (18, 19), (19, 20),  # Pinky
    (0, 17),                                  # Palm base
]


class LiveFilterTunerApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("🎛️ Live 12 FPS 1€ Filter Parameter Visual Tuner (process.sh Pipeline)")
        self.geometry("1360x860")
        self.configure(fg_color=BG)

        # Threading & Shared State
        self.running = False
        self.lock = threading.Lock()
        self.latest_frame = None
        self.latest_fps = 0.0
        self.latest_score = 0.0
        self.latest_l_hand = 0.0
        self.hand_detected = False

        # Filter Parameters State (matching process.sh defaults)
        self.min_cutoff_var = ctk.DoubleVar(value=3.0)
        self.beta_var = ctk.DoubleVar(value=1.4)
        self.d_cutoff_var = ctk.DoubleVar(value=1.0)

        # Display Mode Variables
        self.show_raw_var = ctk.BooleanVar(value=True)
        self.show_filtered_var = ctk.BooleanVar(value=True)
        self.side_by_side_var = ctk.BooleanVar(value=False)

        # Telemetry Labels
        self.fps_var = ctk.StringVar(value="12.0 FPS Target")
        self.hand_score_var = ctk.StringVar(value="Score: --")
        self.l_hand_var = ctk.StringVar(value="L_hand: --")

        self._build_ui()
        self._start_camera_thread()

        # Tkinter Main Thread Update Loop (~50 Hz)
        self.after(20, self._update_gui_loop)

    def _build_ui(self):
        # ── Top Bar ─────────────────────────────────────────────────────────────
        top_bar = ctk.CTkFrame(self, height=48, corner_radius=0, fg_color=HDR_BG)
        top_bar.pack(fill="x", side="top")
        top_bar.pack_propagate(False)

        title_lbl = ctk.CTkLabel(
            top_bar,
            text="🎛️ Live 12 FPS 1€ Filter Parameter Tuner (Exact process.sh Normalization)",
            font=("Helvetica", 15, "bold"),
            text_color=TXT_PRI
        )
        title_lbl.pack(side="left", padx=16)

        fps_lbl = ctk.CTkLabel(
            top_bar,
            textvariable=self.fps_var,
            font=("Helvetica", 12, "bold"),
            text_color=CYAN,
            fg_color="#1e1e1e",
            corner_radius=4,
            padx=10,
            pady=4
        )
        fps_lbl.pack(side="right", padx=16)

        l_hand_lbl = ctk.CTkLabel(
            top_bar,
            textvariable=self.l_hand_var,
            font=("Helvetica", 12, "bold"),
            text_color=AMBER,
            fg_color="#1e1e1e",
            corner_radius=4,
            padx=10,
            pady=4
        )
        l_hand_lbl.pack(side="right", padx=4)

        score_lbl = ctk.CTkLabel(
            top_bar,
            textvariable=self.hand_score_var,
            font=("Helvetica", 12, "bold"),
            text_color="#00dc64",
            fg_color="#1e1e1e",
            corner_radius=4,
            padx=10,
            pady=4
        )
        score_lbl.pack(side="right", padx=4)

        # ── Main Content Container ─────────────────────────────────────────────
        main_container = ctk.CTkFrame(self, fg_color="transparent")
        main_container.pack(fill="both", expand=True, padx=12, pady=12)

        # Left Container: Clean Video Feed Canvas
        self.video_container = ctk.CTkFrame(main_container, fg_color="#101010", corner_radius=8, border_width=1, border_color=BORDER)
        self.video_container.pack(side="left", fill="both", expand=True, padx=(0, 8))
        self.video_container.pack_propagate(False)

        self.video_label = ctk.CTkLabel(self.video_container, text="Initializing 12 FPS Camera Stream...", text_color=TXT_MUT)
        self.video_label.pack(fill="both", expand=True, padx=2, pady=2)

        # Right Sidebar: Controls & Parameter Sliders (350px)
        sidebar = ctk.CTkFrame(main_container, width=350, fg_color=PANEL, corner_radius=8, border_width=1, border_color=BORDER)
        sidebar.pack(side="right", fill="y")
        sidebar.pack_propagate(False)

        # Section 1: Parameter Sliders
        param_hdr = ctk.CTkLabel(sidebar, text="⚙️ 1€ FILTER PARAMETERS (process.sh)", font=("Helvetica", 13, "bold"), text_color=TXT_PRI)
        param_hdr.pack(anchor="w", padx=16, pady=(16, 8))

        # min_cutoff Slider
        mc_lbl_frame = ctk.CTkFrame(sidebar, fg_color="transparent")
        mc_lbl_frame.pack(fill="x", padx=16, pady=(4, 0))
        ctk.CTkLabel(mc_lbl_frame, text="min_cutoff (-min):", font=("Helvetica", 12), text_color=TXT_SEC).pack(side="left")
        self.mc_val_lbl = ctk.CTkLabel(mc_lbl_frame, text=f"{self.min_cutoff_var.get():.2f} Hz", font=("Helvetica", 12, "bold"), text_color=CYAN)
        self.mc_val_lbl.pack(side="right")

        self.mc_slider = ctk.CTkSlider(
            sidebar,
            from_=0.1,
            to=10.0,
            number_of_steps=99,
            variable=self.min_cutoff_var,
            command=self._on_param_change
        )
        self.mc_slider.pack(fill="x", padx=16, pady=(0, 12))

        # beta Slider
        b_lbl_frame = ctk.CTkFrame(sidebar, fg_color="transparent")
        b_lbl_frame.pack(fill="x", padx=16, pady=(4, 0))
        ctk.CTkLabel(b_lbl_frame, text="beta (-beta):", font=("Helvetica", 12), text_color=TXT_SEC).pack(side="left")
        self.b_val_lbl = ctk.CTkLabel(b_lbl_frame, text=f"{self.beta_var.get():.2f}", font=("Helvetica", 12, "bold"), text_color=CYAN)
        self.b_val_lbl.pack(side="right")

        self.b_slider = ctk.CTkSlider(
            sidebar,
            from_=0.0,
            to=10.0,
            number_of_steps=200,
            variable=self.beta_var,
            command=self._on_param_change
        )
        self.b_slider.pack(fill="x", padx=16, pady=(0, 12))

        # d_cutoff Slider
        dc_lbl_frame = ctk.CTkFrame(sidebar, fg_color="transparent")
        dc_lbl_frame.pack(fill="x", padx=16, pady=(4, 0))
        ctk.CTkLabel(dc_lbl_frame, text="d_cutoff (-d):", font=("Helvetica", 12), text_color=TXT_SEC).pack(side="left")
        self.dc_val_lbl = ctk.CTkLabel(dc_lbl_frame, text=f"{self.d_cutoff_var.get():.2f} Hz", font=("Helvetica", 12, "bold"), text_color=CYAN)
        self.dc_val_lbl.pack(side="right")

        self.dc_slider = ctk.CTkSlider(
            sidebar,
            from_=0.1,
            to=10.0,
            number_of_steps=99,
            variable=self.d_cutoff_var,
            command=self._on_param_change
        )
        self.dc_slider.pack(fill="x", padx=16, pady=(0, 16))

        # Section 2: Presets
        preset_hdr = ctk.CTkLabel(sidebar, text="⚡ TUNING PRESETS", font=("Helvetica", 13, "bold"), text_color=TXT_PRI)
        preset_hdr.pack(anchor="w", padx=16, pady=(4, 8))

        preset_grid = ctk.CTkFrame(sidebar, fg_color="transparent")
        preset_grid.pack(fill="x", padx=16, pady=(0, 16))

        btn_default = ctk.CTkButton(preset_grid, text="process.sh Default\n(3.0, 1.4, 1.0)", width=140, height=38, fg_color=BTN_SEC, hover_color="#45494e", command=lambda: self.apply_preset(3.0, 1.4, 1.0))
        btn_default.grid(row=0, column=0, padx=(0, 4), pady=4)

        btn_high_smooth = ctk.CTkButton(preset_grid, text="High Smooth\n(1.0, 0.5, 1.0)", width=140, height=38, fg_color=BTN_SEC, hover_color="#45494e", command=lambda: self.apply_preset(1.0, 0.5, 1.0))
        btn_high_smooth.grid(row=0, column=1, padx=(4, 0), pady=4)

        btn_fast_motion = ctk.CTkButton(preset_grid, text="Fast Responsiveness\n(5.0, 2.5, 1.0)", width=140, height=38, fg_color=BTN_SEC, hover_color="#45494e", command=lambda: self.apply_preset(5.0, 2.5, 1.0))
        btn_fast_motion.grid(row=1, column=0, padx=(0, 4), pady=4)

        btn_zero_filter = ctk.CTkButton(preset_grid, text="Raw Unfiltered\n(10.0, 0.0, 1.0)", width=140, height=38, fg_color=BTN_SEC, hover_color="#45494e", command=lambda: self.apply_preset(10.0, 0.0, 1.0))
        btn_zero_filter.grid(row=1, column=1, padx=(4, 0), pady=4)

        # Section 3: Visual Modes
        vis_hdr = ctk.CTkLabel(sidebar, text="👁️ VISUAL PREVIEW MODES", font=("Helvetica", 13, "bold"), text_color=TXT_PRI)
        vis_hdr.pack(anchor="w", padx=16, pady=(4, 8))

        chk_raw = ctk.CTkCheckBox(sidebar, text="Show Raw Landmarks (Red)", variable=self.show_raw_var, text_color=RED, fg_color=RED, hover_color="#cc0000")
        chk_raw.pack(anchor="w", padx=16, pady=4)

        chk_filt = ctk.CTkCheckBox(sidebar, text="Show 1€ Filtered (Cyan/Amber)", variable=self.show_filtered_var, text_color=CYAN, fg_color=BTN_PRI, hover_color=BTN_HVP)
        chk_filt.pack(anchor="w", padx=16, pady=4)

        chk_side = ctk.CTkCheckBox(sidebar, text="Side-by-Side Split Canvas", variable=self.side_by_side_var, text_color=TXT_SEC)
        chk_side.pack(anchor="w", padx=16, pady=4)

        # Section 4: Export Helper Command
        export_box = ctk.CTkFrame(sidebar, fg_color="#141414", corner_radius=6, border_width=1, border_color=BORDER)
        export_box.pack(fill="x", padx=16, pady=(16, 16))

        ctk.CTkLabel(export_box, text="📋 Target process.sh Command:", font=("Helvetica", 11, "bold"), text_color=TXT_SEC).pack(anchor="w", padx=10, pady=(8, 2))
        self.cmd_lbl = ctk.CTkLabel(
            export_box,
            text=f"./process.sh -min 3.0 -beta 1.4 -d 1.0",
            font=("Courier", 11, "bold"),
            text_color=CYAN,
            justify="left"
        )
        self.cmd_lbl.pack(anchor="w", padx=10, pady=(0, 8))

    def _on_param_change(self, value=None):
        mc = self.min_cutoff_var.get()
        b = self.beta_var.get()
        dc = self.d_cutoff_var.get()

        self.mc_val_lbl.configure(text=f"{mc:.2f} Hz")
        self.b_val_lbl.configure(text=f"{b:.2f}")
        self.dc_val_lbl.configure(text=f"{dc:.2f} Hz")
        self.cmd_lbl.configure(text=f"./process.sh -min {mc:.1f} -beta {b:.1f} -d {dc:.1f}")

    def apply_preset(self, min_cutoff: float, beta: float, d_cutoff: float):
        self.min_cutoff_var.set(min_cutoff)
        self.beta_var.set(beta)
        self.d_cutoff_var.set(d_cutoff)
        self._on_param_change()

    def _start_camera_thread(self):
        self.running = True
        self.thread = threading.Thread(target=self._camera_loop, daemon=True)
        self.thread.start()

    def _camera_loop(self):
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions
        from mediapipe.tasks.python.vision import (
            HandLandmarker, HandLandmarkerOptions, RunningMode
        )

        model_path = os.path.join(PROJECT_ROOT, "hand_landmarker.task")
        if not os.path.exists(model_path):
            print(f"[LiveFilterTuner] Downloading hand_landmarker.task model...")
            import urllib.request
            urllib.request.urlretrieve(
                "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
                model_path
            )

        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=RunningMode.IMAGE,
            num_hands=1
        )
        landmarker = HandLandmarker.create_from_options(options)

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("[Error] Could not open webcam source 0.")
            return

        normalizer = HandScaleNormalizer()
        euro_filter_bank = OneEuroFilterBank(
            min_cutoff=self.min_cutoff_var.get(),
            beta=self.beta_var.get(),
            d_cutoff=self.d_cutoff_var.get()
        )

        # Enforce exact 12 FPS temporal frame sampling rate (1/12.0s interval)
        target_fps = 12.0
        frame_interval = 1.0 / target_fps
        last_capture_time = time.perf_counter()
        stream_start_time = time.perf_counter()
        frame_count = 0
        fps_start_time = time.perf_counter()

        while self.running:
            now = time.perf_counter()
            elapsed = now - last_capture_time
            if elapsed < frame_interval:
                time.sleep(max(0.001, frame_interval - elapsed))
                continue

            last_capture_time = time.perf_counter()
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.01)
                continue

            # Dynamically update 1€ filter parameters from GUI sliders
            mc = float(self.min_cutoff_var.get())
            b = float(self.beta_var.get())
            dc = float(self.d_cutoff_var.get())

            if hasattr(euro_filter_bank, "update_params"):
                euro_filter_bank.update_params(mc, b, dc)
            else:
                euro_filter_bank.min_cutoff = mc
                euro_filter_bank.beta = b
                euro_filter_bank.d_cutoff = dc
                for f in getattr(euro_filter_bank, "filters", {}).values():
                    f.min_cutoff = mc
                    f.beta = b
                    f.d_cutoff = dc

            h, w, _ = frame.shape
            frame_mirror = cv2.flip(frame, 1)
            frame_rgb = cv2.cvtColor(frame_mirror, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            result = landmarker.detect(mp_image)

            t_sec = now - stream_start_time

            if self.side_by_side_var.get():
                raw_canvas = frame_mirror.copy()
                filt_canvas = frame_mirror.copy()
            else:
                raw_canvas = frame_mirror.copy()
                filt_canvas = raw_canvas

            hand_score = 0.0
            l_hand = 0.0
            detected = False

            if result and result.hand_landmarks and len(result.hand_landmarks) > 0:
                detected = True
                landmarks = result.hand_landmarks[0]
                if result.handedness and len(result.handedness) > 0 and len(result.handedness[0]) > 0:
                    hand_score = float(result.handedness[0][0].score)

                # Step 1: Pixel space conversion
                pts_px = [(lm.x * w, lm.y * h, lm.z * w) for lm in landmarks]

                # Step 2: Wrist Centering (0,0,0) & 8-Distance Palm RMS Scale Normalization L_hand (process.sh Step 2)
                norm_pts = normalizer.normalize(pts_px, center_wrist=True)

                # Extract L_hand for telemetry display
                w_x, w_y, _ = pts_px[WRIST_INDEX]
                i_x, i_y, _ = pts_px[INDEX_MCP_INDEX]
                m_x, m_y, _ = pts_px[MIDDLE_MCP_INDEX]
                r_x, r_y, _ = pts_px[RING_MCP_INDEX]
                p_x, p_y, _ = pts_px[PINKY_MCP_INDEX]
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

                # Step 3: Apply 1€ filter ON SCALE-NORMALIZED wrist-centered coordinates (process.sh Step 3)
                filtered_norm_pts = euro_filter_bank.filter_frame(t_sec, norm_pts)

                # Filter wrist origin (w_x, w_y) and palm scale (L_hand) to eliminate raw MediaPipe wrist jitter from screen preview
                if "wrist_px_x" not in euro_filter_bank.filters:
                    euro_filter_bank.filters["wrist_px_x"] = OneEuroFilter1D(t_sec, w_x, mc, b, dc)
                    euro_filter_bank.filters["wrist_px_y"] = OneEuroFilter1D(t_sec, w_y, mc, b, dc)
                    euro_filter_bank.filters["l_hand"] = OneEuroFilter1D(t_sec, l_hand, mc, b, dc)
                    filtered_w_x = w_x
                    filtered_w_y = w_y
                    filtered_l_hand = l_hand
                else:
                    euro_filter_bank.filters["wrist_px_x"].min_cutoff = mc
                    euro_filter_bank.filters["wrist_px_x"].beta = b
                    euro_filter_bank.filters["wrist_px_x"].d_cutoff = dc

                    euro_filter_bank.filters["wrist_px_y"].min_cutoff = mc
                    euro_filter_bank.filters["wrist_px_y"].beta = b
                    euro_filter_bank.filters["wrist_px_y"].d_cutoff = dc

                    euro_filter_bank.filters["l_hand"].min_cutoff = mc
                    euro_filter_bank.filters["l_hand"].beta = b
                    euro_filter_bank.filters["l_hand"].d_cutoff = dc

                    filtered_w_x = euro_filter_bank.filters["wrist_px_x"].filter(t_sec, w_x)
                    filtered_w_y = euro_filter_bank.filters["wrist_px_y"].filter(t_sec, w_y)
                    filtered_l_hand = euro_filter_bank.filters["l_hand"].filter(t_sec, l_hand)

                # Reconstruct silk-smooth 2D pixel space for visual skeleton preview
                smooth_pts_px = [
                    (filtered_w_x + nx * filtered_l_hand, filtered_w_y + ny * filtered_l_hand)
                    for (nx, ny, _) in filtered_norm_pts
                ]

                # Render Raw Skeleton (Red)
                if self.show_raw_var.get() or self.side_by_side_var.get():
                    c_target = raw_canvas if self.side_by_side_var.get() else filt_canvas
                    for s_idx, e_idx in HAND_CONNECTIONS:
                        x1, y1 = int(pts_px[s_idx][0]), int(pts_px[s_idx][1])
                        x2, y2 = int(pts_px[e_idx][0]), int(pts_px[e_idx][1])
                        cv2.line(c_target, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    for (rx, ry, _) in pts_px:
                        cv2.circle(c_target, (int(rx), int(ry)), 4, (0, 0, 255), -1)

                # Render 1€ Filtered Skeleton (Cyan/Amber)
                if self.show_filtered_var.get() or self.side_by_side_var.get():
                    for s_idx, e_idx in HAND_CONNECTIONS:
                        x1, y1 = int(smooth_pts_px[s_idx][0]), int(smooth_pts_px[s_idx][1])
                        x2, y2 = int(smooth_pts_px[e_idx][0]), int(smooth_pts_px[e_idx][1])
                        cv2.line(filt_canvas, (x1, y1), (x2, y2), (255, 200, 0), 2)
                    for idx, (sx, sy) in enumerate(smooth_pts_px):
                        px, py = int(sx), int(sy)
                        if idx in [4, 8, 12, 16, 20]:
                            cv2.circle(filt_canvas, (px, py), 6, (0, 255, 255), -1)
                        else:
                            cv2.circle(filt_canvas, (px, py), 4, (0, 165, 255), -1)
            else:
                euro_filter_bank.reset()

            # Final Display Frame Assembly
            if self.side_by_side_var.get():
                cv2.putText(raw_canvas, "RAW MEDIAPIPE (RED)", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                cv2.putText(filt_canvas, f"1 EUR FILTER (min={self.min_cutoff_var.get():.1f}, b={self.beta_var.get():.1f})", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                combined_frame = cv2.hconcat([raw_canvas, filt_canvas])
            else:
                combined_frame = filt_canvas

            # Calculate actual 12 FPS telemetry
            frame_count += 1
            fps_dur = now - fps_start_time
            curr_fps = self.latest_fps
            if fps_dur >= 1.0:
                curr_fps = frame_count / fps_dur
                frame_count = 0
                fps_start_time = now

            with self.lock:
                self.latest_frame = combined_frame
                self.latest_fps = curr_fps
                self.latest_score = hand_score
                self.latest_l_hand = l_hand
                self.hand_detected = detected

        cap.release()

    def _update_gui_loop(self):
        """Runs on the main Tkinter thread at ~50 Hz to update video display without distortion."""
        if not self.running:
            return

        frame = None
        fps = 0.0
        score = 0.0
        l_hand = 0.0

        with self.lock:
            if self.latest_frame is not None:
                frame = self.latest_frame.copy()
                fps = self.latest_fps
                score = self.latest_score
                l_hand = self.latest_l_hand

        if frame is not None:
            container_w = max(100, self.video_container.winfo_width() - 8)
            container_h = max(100, self.video_container.winfo_height() - 8)

            h, w, _ = frame.shape
            scale = min(container_w / float(w), container_h / float(h))
            nw, nh = max(1, int(w * scale)), max(1, int(h * scale))

            resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)
            rgb_img = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb_img)

            ctk_img = ctk.CTkImage(light_image=pil_img, dark_image=pil_img, size=(nw, nh))
            self.video_label.configure(image=ctk_img, text="")

            self.fps_var.set(f"{fps:.1f} FPS (Target: 12.0)")
            self.hand_score_var.set(f"Score: {score:.2f}")
            self.l_hand_var.set(f"L_hand: {l_hand:.1f}px")

        self.after(20, self._update_gui_loop)

    def on_closing(self):
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.0)
        self.destroy()


def main():
    ctk.set_appearance_mode("dark")
    ctk.set_default_color_theme("blue")

    app = LiveFilterTunerApp()
    app.protocol("WM_DELETE_WINDOW", app.on_closing)
    app.mainloop()


if __name__ == "__main__":
    main()
