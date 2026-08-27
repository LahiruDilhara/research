"""
datacreator/annotator/annotation_screen.py

Main Annotation Screen UI.
Renders pre-calculated 21-joint skeleton on 12 FPS video frames.
Supports subframe stepping, window looping, shortcut customization (persisted to shortcuts.json),
and automated silent saving.
"""

import os
import sys
from pathlib import Path
from tkinter import messagebox

import cv2
import customtkinter as ctk
from PIL import Image

from datacreator.annotator.csv_manager import WindowAnnotationCSVManager
from datacreator.annotator.shortcuts import ShortcutEditorDialog, ShortcutManager

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
BTN_GHO = "#2d2d2d"
BTN_GHH = "#3e3e42"
AMBER = "#ce9178"
GREEN = "#4ec9f0"

FINGERS = ["Thumb", "Index", "Middle", "Ring", "Pinky"]
FINGER_COLORS_HEX = {
    "Thumb":  "#ff8c00",  # Dark Orange
    "Index":  "#00c8ff",  # Cyan
    "Middle": "#00e676",  # Emerald Green
    "Ring":   "#ff00ff",  # Magenta
    "Pinky":  "#3d7eff",  # Bright Blue
}

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


class AnnotationScreen(ctk.CTkFrame):
    def __init__(self, parent, app, video_path: str, csv_path: str, window_size: int = 5, window_overlap: int = 2):
        super().__init__(parent, fg_color=BG)
        self.app = app
        self.video_path = video_path
        self.csv_path = csv_path
        self.window_size = window_size
        self.window_overlap = window_overlap

        self.cap = None
        self.csv_mgr: WindowAnnotationCSVManager = None
        self.shortcut_mgr = ShortcutManager()
        self.current_window_idx = 0
        self.current_subframe_offset = 0

        self._loop_running = False
        self._loop_job = None

        # State Variables
        self.touch_vars = {f: ctk.BooleanVar(value=False) for f in FINGERS}
        self.right_hand_var = ctk.BooleanVar(value=True)
        self.hand_move_var = ctk.BooleanVar(value=False)
        self.hand_closer_var = ctk.BooleanVar(value=False)
        self.hovering_var = ctk.BooleanVar(value=False)
        self.daylight_var = ctk.BooleanVar(value=True)
        self.hand_visible_var = ctk.BooleanVar(value=True)
        self.out_of_sync_var = ctk.BooleanVar(value=False)
        self.auto_save_var = ctk.BooleanVar(value=True)

        self._build_ui()
        self._bind_keyboard_shortcuts()
        self.load_session(video_path, csv_path)

    def _build_ui(self):
        # ── Top Bar ─────────────────────────────────────────────────────────────
        top_bar = ctk.CTkFrame(self, height=48, corner_radius=0, fg_color=HDR_BG)
        top_bar.pack(fill="x", side="top")
        top_bar.pack_propagate(False)

        ctk.CTkLabel(
            top_bar, text="🖐 Lightweight Window Annotator",
            font=ctk.CTkFont(family=FF, size=15, weight="bold"),
            text_color=TXT_PRI
        ).pack(side="left", padx=16)

        # Status Badge (New vs Overriding Annotation)
        self.lbl_override_badge = ctk.CTkLabel(
            top_bar, text="🆕 NEW ANNOTATION",
            font=ctk.CTkFont(family=FF, size=11, weight="bold"),
            fg_color="#1b5e20", text_color="#ffffff", corner_radius=4, padx=8, pady=2
        )
        self.lbl_override_badge.pack(side="left", padx=12)

        ctk.CTkButton(
            top_bar, text="⚙ Settings", width=100, height=32,
            fg_color=BTN_SEC, hover_color="#4f5258",
            font=ctk.CTkFont(family=FF, size=12),
            command=self._on_change_settings
        ).pack(side="right", padx=(4, 16))

        ctk.CTkButton(
            top_bar, text="⌨ Shortcuts", width=110, height=32,
            fg_color=BTN_GHO, hover_color=BTN_GHH, border_width=1, border_color=BORDER,
            font=ctk.CTkFont(family=FF, size=12),
            command=self._open_shortcuts_dialog
        ).pack(side="right", padx=4)

        ctk.CTkButton(
            top_bar, text="💾 Save CSV", width=110, height=32,
            fg_color="#2e7d32", hover_color="#1b5e20",
            font=ctk.CTkFont(family=FF, size=12, weight="bold"),
            command=lambda: self.save_csv(show_toast=True)
        ).pack(side="right", padx=4)

        # ── Main Content Area ───────────────────────────────────────────────────
        main_content = ctk.CTkFrame(self, fg_color=BG)
        main_content.pack(fill="both", expand=True, padx=12, pady=12)

        # Left Column: Video & Controls
        left_col = ctk.CTkFrame(main_content, fg_color=PANEL, corner_radius=6, border_width=1, border_color=BORDER)
        left_col.pack(side="left", fill="both", expand=True, padx=(0, 8))

        # Video Display Canvas / Label
        self.video_label = ctk.CTkLabel(left_col, text="Loading Video...", text_color=TXT_MUT)
        self.video_label.pack(fill="both", expand=True, padx=12, pady=12)

        # Subframe & Playback Control Bar
        subframe_bar = ctk.CTkFrame(left_col, fg_color="transparent")
        subframe_bar.pack(fill="x", padx=12, pady=(0, 6))

        self.btn_prev_frame = ctk.CTkButton(
            subframe_bar, text="Step ◄ (Left / ,)", width=140, height=30, fg_color=BTN_GHO, hover_color=BTN_GHH,
            text_color=TXT_SEC, font=ctk.CTkFont(family=FF, size=11),
            command=self.prev_subframe
        )
        self.btn_prev_frame.pack(side="left", padx=4)

        self.btn_play = ctk.CTkButton(
            subframe_bar, text="▶ Play Window (Space)", width=170, height=30, fg_color=BTN_PRI, hover_color=BTN_HVP,
            font=ctk.CTkFont(family=FF, size=12, weight="bold"),
            command=self.toggle_play
        )
        self.btn_play.pack(side="left", padx=4)

        self.btn_next_frame = ctk.CTkButton(
            subframe_bar, text="Step ► (Right / .)", width=140, height=30, fg_color=BTN_GHO, hover_color=BTN_GHH,
            text_color=TXT_SEC, font=ctk.CTkFont(family=FF, size=11),
            command=self.next_subframe
        )
        self.btn_next_frame.pack(side="left", padx=4)

        # Speed Slider & Label
        ctk.CTkLabel(subframe_bar, text="Speed:", font=ctk.CTkFont(family=FF, size=11), text_color=TXT_SEC).pack(side="left", padx=(12, 4))
        self.lbl_speed_val = ctk.CTkLabel(subframe_bar, text="12 FPS", font=ctk.CTkFont(family=FF, size=11, weight="bold"), text_color=GREEN, width=48)
        self.lbl_speed_val.pack(side="left", padx=2)

        self.slider_speed = ctk.CTkSlider(
            subframe_bar, from_=1, to=30, number_of_steps=29, width=110,
            command=self._on_speed_changed
        )
        self.slider_speed.set(12)
        self.slider_speed.pack(side="left", padx=4)

        self.lbl_subframe_info = ctk.CTkLabel(
            subframe_bar, text="Position 1 / 5",
            font=ctk.CTkFont(family=FF, size=12, weight="bold"), text_color=AMBER
        )
        self.lbl_subframe_info.pack(side="right", padx=8)

        # Window Navigation Slider Bar
        ctrl_bar = ctk.CTkFrame(left_col, fg_color="transparent")
        ctrl_bar.pack(fill="x", padx=12, pady=(0, 12))

        self.btn_prev_win = ctk.CTkButton(
            ctrl_bar, text="⏮ Prev Window", width=130, height=32, fg_color=BTN_SEC,
            command=self.prev_window
        )
        self.btn_prev_win.pack(side="left", padx=4)

        self.btn_next_win = ctk.CTkButton(
            ctrl_bar, text="Next Window ⏭", width=130, height=32, fg_color=BTN_SEC,
            command=self.next_window
        )
        self.btn_next_win.pack(side="left", padx=4)

        self.slider_window = ctk.CTkSlider(
            ctrl_bar, from_=0, to=100, number_of_steps=100,
            command=self._on_slider_move
        )
        self.slider_window.pack(side="left", fill="x", expand=True, padx=12)

        self.lbl_window_info = ctk.CTkLabel(ctrl_bar, text="Window: 0 / 0", font=ctk.CTkFont(family=FF, size=12, weight="bold"), text_color=GREEN)
        self.lbl_window_info.pack(side="right", padx=6)

        # Right Column: Finger Touch & Environment Annotation Panel
        right_col = ctk.CTkFrame(main_content, width=360, fg_color=PANEL, corner_radius=6, border_width=1, border_color=BORDER)
        right_col.pack(side="right", fill="y", padx=(4, 0))
        right_col.pack_propagate(False)

        ctk.CTkLabel(
            right_col, text="Window Touch & Environment",
            font=ctk.CTkFont(family=FF, size=16, weight="bold"), text_color=TXT_PRI
        ).pack(anchor="w", padx=16, pady=(14, 10))

        # Finger Touch Toggle Buttons (1 to 5)
        self.touch_buttons = {}
        for finger in FINGERS:
            f_color = FINGER_COLORS_HEX[finger]
            key_name = self.shortcut_mgr.get(finger.lower()).upper()
            btn = ctk.CTkButton(
                right_col,
                text=f"{finger} Touch ({key_name})  [OFF]",
                font=ctk.CTkFont(family=FF, size=13, weight="bold"),
                fg_color=BTN_GHO, hover_color=BTN_GHH,
                text_color=f_color,
                height=38, corner_radius=6, border_width=1, border_color=f_color,
                command=lambda f=finger: self.toggle_finger_touch(f)
            )
            btn.pack(fill="x", padx=16, pady=4)
            self.touch_buttons[finger] = btn

        # Environmental & Hand Motion Toggles
        ctk.CTkFrame(right_col, height=1, fg_color=BORDER).pack(fill="x", padx=16, pady=12)

        ctk.CTkLabel(
            right_col, text="Hand & Environment Controls",
            font=ctk.CTkFont(family=FF, size=13, weight="bold"), text_color=TXT_SEC
        ).pack(anchor="w", padx=16, pady=(0, 6))

        # Right vs Left Hand Toggle Switch
        ctk.CTkSwitch(
            right_col, text="Right Hand (ON) / Left Hand (OFF)", variable=self.right_hand_var,
            progress_color=BTN_PRI, font=ctk.CTkFont(family=FF, size=12), text_color=TXT_PRI,
            command=self._on_annotation_change
        ).pack(anchor="w", padx=16, pady=4)

        ctk.CTkCheckBox(
            right_col, text="Hand Motion / Moving", variable=self.hand_move_var,
            text_color=TXT_SEC, font=ctk.CTkFont(family=FF, size=12),
            command=self._on_annotation_change
        ).pack(anchor="w", padx=16, pady=4)

        ctk.CTkCheckBox(
            right_col, text="Hand Moving Closer (Zooming)", variable=self.hand_closer_var,
            text_color=TXT_SEC, font=ctk.CTkFont(family=FF, size=12),
            command=self._on_annotation_change
        ).pack(anchor="w", padx=16, pady=4)

        ctk.CTkCheckBox(
            right_col, text="Hand Hovering", variable=self.hovering_var,
            text_color=TXT_SEC, font=ctk.CTkFont(family=FF, size=12),
            command=self._on_annotation_change
        ).pack(anchor="w", padx=16, pady=4)

        ctk.CTkCheckBox(
            right_col, text="Daylight / Normal Light", variable=self.daylight_var,
            text_color=TXT_SEC, font=ctk.CTkFont(family=FF, size=12),
            command=self._on_annotation_change
        ).pack(anchor="w", padx=16, pady=4)

        ctk.CTkCheckBox(
            right_col, text="Hand Visible in Window", variable=self.hand_visible_var,
            text_color=TXT_SEC, font=ctk.CTkFont(family=FF, size=12),
            command=self._on_annotation_change
        ).pack(anchor="w", padx=16, pady=4)

        ctk.CTkCheckBox(
            right_col, text="Out of Sync (Frame Lag)", variable=self.out_of_sync_var,
            text_color=AMBER, font=ctk.CTkFont(family=FF, size=12),
            command=self._on_annotation_change
        ).pack(anchor="w", padx=16, pady=4)

        ctk.CTkSwitch(
            right_col, text="Auto-Save on Change / Navigation", variable=self.auto_save_var,
            progress_color=BTN_PRI, font=ctk.CTkFont(family=FF, size=12),
            text_color=GREEN
        ).pack(anchor="w", padx=16, pady=(12, 4))

        # Metadata Information Box at Bottom Right
        ctk.CTkFrame(right_col, height=1, fg_color=BORDER).pack(fill="x", padx=16, pady=12)

        self.lbl_metadata = ctk.CTkLabel(
            right_col, text="Metadata...",
            font=ctk.CTkFont(family=FF, size=11), text_color=TXT_MUT,
            justify="left", anchor="w", wraplength=320
        )
        self.lbl_metadata.pack(fill="both", expand=True, padx=16, pady=(0, 12))

    def _on_speed_changed(self, val):
        if hasattr(self, "lbl_speed_val"):
            self.lbl_speed_val.configure(text=f"{int(float(val))} FPS")

    def _bind_keyboard_shortcuts(self):
        self.app.bind("<KeyPress>", self._on_key_press)

    def _on_key_press(self, event):
        widget_class = getattr(event.widget, "winfo_class", lambda: "")()
        if widget_class in ("Entry", "Text", "TCombobox"):
            return

        key = event.keysym.lower()
        char = event.char.lower() if event.char else ""

        # Match strictly against customizable ShortcutManager
        def is_match(action: str) -> bool:
            bound = self.shortcut_mgr.get(action).lower()
            if not bound:
                return False
            return key == bound or char == bound

        if is_match("play_window"):
            self.toggle_play()
        elif is_match("step_back"):
            self.prev_subframe()
        elif is_match("step_forward"):
            self.next_subframe()
        elif is_match("prev_window"):
            self.prev_window()
        elif is_match("next_window"):
            self.next_window()
        elif is_match("thumb"):
            self.toggle_finger_touch("Thumb")
        elif is_match("index"):
            self.toggle_finger_touch("Index")
        elif is_match("middle"):
            self.toggle_finger_touch("Middle")
        elif is_match("ring"):
            self.toggle_finger_touch("Ring")
        elif is_match("pinky"):
            self.toggle_finger_touch("Pinky")
        elif is_match("save") or ((event.state & 0x4) and key == "s"):
            self.save_csv(show_toast=True)

    def _open_shortcuts_dialog(self):
        ShortcutEditorDialog(self.app, self.shortcut_mgr, on_save_callback=self._refresh_shortcut_labels)

    def _refresh_shortcut_labels(self):
        """Refreshes button labels when shortcuts are edited."""
        for finger in FINGERS:
            key_name = self.shortcut_mgr.get(finger.lower()).upper()
            val = self.touch_vars[finger].get()
            f_color = FINGER_COLORS_HEX[finger]
            btn = self.touch_buttons[finger]
            if val:
                btn.configure(text=f"{finger} Touch ({key_name})  [ON]", fg_color=f_color, text_color="#ffffff")
            else:
                btn.configure(text=f"{finger} Touch ({key_name})  [OFF]", fg_color=BTN_GHO, text_color=f_color)

    def _on_change_settings(self):
        self._stop_loop()
        self.app.show_setup_screen()

    def load_session(self, video_path: str, csv_path: str):
        if not os.path.exists(video_path) or not os.path.exists(csv_path):
            return

        self.video_path = video_path
        self.csv_path = csv_path

        self.csv_mgr = WindowAnnotationCSVManager(csv_path, window_size=self.window_size, window_overlap=self.window_overlap)

        if self.cap:
            self.cap.release()

        self.cap = cv2.VideoCapture(video_path)

        self.slider_window.configure(
            from_=0,
            to=max(1, self.csv_mgr.total_windows - 1),
            number_of_steps=max(1, self.csv_mgr.total_windows - 1)
        )

        w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.lbl_metadata.configure(
            text=f"Video: {os.path.basename(video_path)}\n"
                 f"Resolution: {w}x{h}\n"
                 f"Total Frames: {self.csv_mgr.total_frames}\n"
                 f"Total Windows: {self.csv_mgr.total_windows}\n"
                 f"Window Size: {self.window_size} frames (Overlap: {self.window_overlap})\n"
                 f"Raw CSV: {os.path.basename(csv_path)}"
        )

        self.current_window_idx = 0
        self.current_subframe_offset = 0
        self.show_window(0)

    def show_window(self, window_idx: int, subframe_offset: int = 0, auto_play: bool = False):
        if not self.cap or not self.csv_mgr or self.csv_mgr.total_windows == 0:
            return

        window_idx = max(0, min(window_idx, self.csv_mgr.total_windows - 1))
        self.current_window_idx = window_idx

        # Check if window is already annotated
        is_overriding = self.csv_mgr.has_annotation(window_idx)
        if is_overriding:
            self.lbl_override_badge.configure(
                text=f"⚠️ OVERRIDING ANNOTATION (Win #{window_idx + 1})",
                fg_color="#b71c1c"
            )
        else:
            self.lbl_override_badge.configure(
                text=f"🆕 NEW ANNOTATION (Win #{window_idx + 1})",
                fg_color="#1b5e20"
            )

        start_frame, end_frame = self.csv_mgr.get_window_frame_indices(window_idx)
        win_frames_data = self.csv_mgr.get_window_frames_data(window_idx)

        subframe_offset = max(0, min(subframe_offset, len(win_frames_data) - 1))
        self.current_subframe_offset = subframe_offset

        target_frame_idx = start_frame + subframe_offset
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame_idx)

        ret, frame = self.cap.read()
        if not ret:
            return

        frame_record = win_frames_data[subframe_offset]
        window_ann = self.csv_mgr.get_window_annotation(window_idx)

        # Render skeleton on video frame
        frame = self._draw_skeleton_overlay(frame, frame_record, window_ann)
        self._update_controls_from_annotation(window_ann)

        # Convert to PIL Image for CustomTkinter Display
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        ctk_img = ctk.CTkImage(light_image=pil_img, dark_image=pil_img, size=(640, 480))

        self.video_label.configure(image=ctk_img, text="")
        self.slider_window.set(window_idx)
        self.lbl_window_info.configure(
            text=f"Window: {window_idx + 1} / {self.csv_mgr.total_windows}  (Frames {start_frame}..{end_frame})"
        )
        self.lbl_subframe_info.configure(
            text=f"Position {subframe_offset + 1} / {len(win_frames_data)} (Frame {target_frame_idx})"
        )

        if auto_play:
            self._start_play()

    def _draw_skeleton_overlay(self, frame, record: dict, window_ann: dict) -> cv2.Mat:
        """Draws pre-calculated 21 hand landmarks and skeleton connections onto video frame."""
        h, w, _ = frame.shape
        hand_type = record.get("hand", "None")

        if hand_type == "None":
            cv2.putText(frame, "No Hand Detected", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            return frame

        cv2.putText(frame, f"Hand: {hand_type} (Pre-calculated)", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Extract 21 points from raw CSV record (nx, ny in [0.0, 1.0])
        pts_px = []
        for lm_name in ALL_21_LANDMARK_NAMES:
            nx = float(record.get(f"{lm_name}_x", 0.0))
            ny = float(record.get(f"{lm_name}_y", 0.0))
            pts_px.append((int(round(nx * w)), int(round(ny * h))))

        # Draw Skeleton Connections
        for a, b in HAND_CONNECTIONS:
            pt_a = pts_px[a]
            pt_b = pts_px[b]
            cv2.line(frame, pt_a, pt_b, (200, 200, 200), 2)

        # Draw Wrist (Yellow)
        cv2.circle(frame, pts_px[0], 6, (0, 255, 255), -1)

        # Draw Finger Joints with Finger Colors
        finger_indices = {
            "Thumb":  [1, 2, 3, 4],
            "Index":  [5, 6, 7, 8],
            "Middle": [9, 10, 11, 12],
            "Ring":   [13, 14, 15, 16],
            "Pinky":  [17, 18, 19, 20],
        }

        bgr_colors = {
            "Thumb":  (0, 140, 255),   # Orange
            "Index":  (255, 200, 0),   # Cyan
            "Middle": (118, 230, 0),   # Emerald Green
            "Ring":   (255, 0, 255),   # Magenta
            "Pinky":  (255, 126, 61),  # Blue
        }

        for finger, indices in finger_indices.items():
            color = bgr_colors[finger]
            is_touching = window_ann.get(f"{finger.lower()}_touch", "0") == "1"

            for idx in indices:
                radius = 7 if idx in (4, 8, 12, 16, 20) else 4
                cv2.circle(frame, pts_px[idx], radius, color, -1)

            # Draw Touch Indicator Badge on Fingertip if window has touch active
            tip_pt = pts_px[indices[-1]]
            if is_touching:
                cv2.circle(frame, tip_pt, 12, (0, 0, 255), 2)
                cv2.putText(frame, f"{finger} TOUCH", (tip_pt[0] + 10, tip_pt[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

        return frame

    def _update_controls_from_annotation(self, window_ann: dict):
        for finger in FINGERS:
            val = window_ann.get(f"{finger.lower()}_touch", "0") == "1"
            self.touch_vars[finger].set(val)
            f_color = FINGER_COLORS_HEX[finger]
            key_name = self.shortcut_mgr.get(finger.lower()).upper()
            btn = self.touch_buttons[finger]
            if val:
                btn.configure(text=f"{finger} Touch ({key_name})  [ON]", fg_color=f_color, text_color="#ffffff")
            else:
                btn.configure(text=f"{finger} Touch ({key_name})  [OFF]", fg_color=BTN_GHO, text_color=f_color)

        self.right_hand_var.set(window_ann.get("right_hand", "1") == "1")
        self.hand_move_var.set(window_ann.get("hand_move", "0") == "1")
        self.hand_closer_var.set(window_ann.get("hand_closer", "0") == "1")
        self.hovering_var.set(window_ann.get("hovering", "0") == "1")
        self.daylight_var.set(window_ann.get("daylight", "1") == "1")
        self.hand_visible_var.set(window_ann.get("hand_visible", "1") == "1")
        self.out_of_sync_var.set(window_ann.get("out_of_sync", "0") == "1")

    def toggle_finger_touch(self, finger: str):
        curr = self.touch_vars[finger].get()
        self.touch_vars[finger].set(not curr)
        self._on_annotation_change()

    def _on_annotation_change(self):
        if not self.csv_mgr:
            return
        touch_dict = {f: self.touch_vars[f].get() for f in FINGERS}
        self.csv_mgr.update_window_annotation(
            self.current_window_idx,
            touch_dict,
            right_hand=self.right_hand_var.get(),
            hand_move=self.hand_move_var.get(),
            hand_closer=self.hand_closer_var.get(),
            hovering=self.hovering_var.get(),
            daylight=self.daylight_var.get(),
            hand_visible=self.hand_visible_var.get(),
            out_of_sync=self.out_of_sync_var.get()
        )
        self.show_window(self.current_window_idx, self.current_subframe_offset)

        if self.auto_save_var.get():
            self.save_csv(show_toast=False)

    def save_csv(self, show_toast: bool = False):
        """Automated silent background saving of CSV annotations."""
        if self.csv_mgr:
            save_path = self.csv_mgr.save_window_annotations()
            if show_toast:
                messagebox.showinfo("Saved", f"Window annotations saved successfully to:\n{save_path}")

    def prev_subframe(self):
        """Step backward frame-by-frame inside the current window."""
        self._stop_loop()
        win_frames_data = self.csv_mgr.get_window_frames_data(self.current_window_idx)
        if self.current_subframe_offset > 0:
            self.show_window(self.current_window_idx, self.current_subframe_offset - 1)

    def next_subframe(self):
        """Step forward frame-by-frame inside the current window."""
        self._stop_loop()
        win_frames_data = self.csv_mgr.get_window_frames_data(self.current_window_idx)
        if self.current_subframe_offset < len(win_frames_data) - 1:
            self.show_window(self.current_window_idx, self.current_subframe_offset + 1)

    def _save_leaving_window(self):
        """Saves current window annotation to CSV upon leaving the window."""
        if not self.csv_mgr:
            return
        touch_dict = {f: self.touch_vars[f].get() for f in FINGERS}
        self.csv_mgr.update_window_annotation(
            self.current_window_idx,
            touch_dict,
            right_hand=self.right_hand_var.get(),
            hand_move=self.hand_move_var.get(),
            hand_closer=self.hand_closer_var.get(),
            hovering=self.hovering_var.get(),
            daylight=self.daylight_var.get(),
            hand_visible=self.hand_visible_var.get(),
            out_of_sync=self.out_of_sync_var.get()
        )
        if self.auto_save_var.get():
            self.csv_mgr.save_window_annotations(verbose=False)

    def prev_window(self):
        self._stop_loop()
        if self.current_window_idx > 0:
            self._save_leaving_window()
            next_win = self.current_window_idx - 1
            if not self.csv_mgr.has_annotation(next_win):
                for f in FINGERS:
                    self.touch_vars[f].set(False)
            self.show_window(next_win, 0, auto_play=True)

    def next_window(self):
        self._stop_loop()
        if self.current_window_idx < self.csv_mgr.total_windows - 1:
            self._save_leaving_window()
            next_win = self.current_window_idx + 1
            if not self.csv_mgr.has_annotation(next_win):
                for f in FINGERS:
                    self.touch_vars[f].set(False)
            self.show_window(next_win, 0, auto_play=True)

    def _on_slider_move(self, val):
        self._stop_loop()
        target_win = int(val)
        if target_win != self.current_window_idx:
            self._save_leaving_window()
            if not self.csv_mgr.has_annotation(target_win):
                for f in FINGERS:
                    self.touch_vars[f].set(False)
            self.show_window(target_win, 0, auto_play=True)

    def toggle_play(self):
        if self._loop_running:
            self._stop_loop()
        else:
            self._start_play()

    def _start_play(self):
        self._stop_loop()
        self._loop_running = True
        self.btn_play.configure(text="⏸ Playing...", fg_color="#d32f2f")
        self.show_window(self.current_window_idx, 0, auto_play=False)

        fps = max(1, int(self.slider_speed.get()))
        delay_ms = max(30, int(1000 / fps))
        self._loop_job = self.after(delay_ms, self._tick_play)

    def _stop_loop(self):
        self._loop_running = False
        if self._loop_job:
            self.after_cancel(self._loop_job)
            self._loop_job = None
        self.btn_play.configure(text="▶ Play Window (Space)", fg_color=BTN_PRI)

    def _tick_play(self):
        """Plays through frames 1 to N of the window once at target FPS then stops."""
        if not self._loop_running or not self.csv_mgr:
            return

        win_frames_data = self.csv_mgr.get_window_frames_data(self.current_window_idx)
        n = len(win_frames_data)
        if n == 0:
            self._stop_loop()
            return

        if self.current_subframe_offset < n - 1:
            next_offset = self.current_subframe_offset + 1
            self.show_window(self.current_window_idx, next_offset, auto_play=False)
            fps = max(1, int(self.slider_speed.get()))
            delay_ms = max(30, int(1000 / fps))
            self._loop_job = self.after(delay_ms, self._tick_play)
        else:
            # Reached last frame of the window -> stop playback
            self._stop_loop()
