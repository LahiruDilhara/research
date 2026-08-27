"""
datacreator/annotator/setup_screen.py

Setup and Window Initialization screen styled with VS Code Dark+ palette.
Allows setting video file, raw landmarks CSV file, window size (default: 5 frames), and window overlap (default: 2 frames).
"""

import os
from tkinter import filedialog, messagebox

import customtkinter as ctk

# Colors
FF = "Helvetica"
BG = "#1e1e1e"
PANEL = "#252526"
BORDER = "#3c3c3c"
TXT_PRI = "#ffffff"
TXT_SEC = "#cccccc"
TXT_MUT = "#858585"
BTN_PRI = "#007acc"
BTN_HVP = "#005999"
BTN_SEC = "#3a3d41"
GREEN = "#4ec9f0"


class SetupScreen(ctk.CTkFrame):
    def __init__(self, parent, app, initial_video: str = "", initial_csv: str = "", window_size: int = 5, window_overlap: int = 2):
        super().__init__(parent, fg_color=BG)
        self.app = app

        self.video_path_var = ctk.StringVar(value=initial_video)
        self.csv_path_var = ctk.StringVar(value=initial_csv)
        self.window_size_var = ctk.StringVar(value=str(window_size))
        self.window_overlap_var = ctk.StringVar(value=str(window_overlap))

        self._build_ui()

    def _build_ui(self):
        center = ctk.CTkFrame(self, fg_color="transparent")
        center.place(relx=0.5, rely=0.5, anchor="center")

        # Header Title
        ctk.CTkLabel(
            center, text="🖐 Lightweight Hand Landmark Annotator",
            font=ctk.CTkFont(family=FF, size=24, weight="bold"),
            text_color=TXT_PRI
        ).pack(pady=(0, 4))

        ctk.CTkLabel(
            center, text="Window-Based Touch Annotations with Pre-Calculated Raw Landmarks",
            font=ctk.CTkFont(family=FF, size=13),
            text_color=TXT_SEC
        ).pack(pady=(0, 20))

        # ── Card 1: Input Files Setup ───────────────────────────────────────────
        file_card = ctk.CTkFrame(center, corner_radius=6, border_width=1, border_color=BORDER, fg_color=PANEL)
        file_card.pack(fill="x", pady=(0, 16), ipadx=10, ipady=4)

        ctk.CTkLabel(
            file_card, text="📁 Select Input Files",
            font=ctk.CTkFont(family=FF, size=14, weight="bold"), text_color=TXT_PRI
        ).pack(anchor="w", padx=20, pady=(16, 12))

        # Video Row
        v_row = ctk.CTkFrame(file_card, fg_color="transparent")
        v_row.pack(fill="x", padx=20, pady=4)
        ctk.CTkLabel(v_row, text="12 FPS Video File:", width=140, anchor="w", text_color=TXT_SEC, font=ctk.CTkFont(family=FF, size=12)).pack(side="left")
        ctk.CTkEntry(v_row, textvariable=self.video_path_var, font=ctk.CTkFont(family=FF, size=12), width=320).pack(side="left", padx=8)
        ctk.CTkButton(v_row, text="Browse...", width=90, fg_color=BTN_SEC, command=self._browse_video).pack(side="left")

        # CSV Row
        c_row = ctk.CTkFrame(file_card, fg_color="transparent")
        c_row.pack(fill="x", padx=20, pady=(4, 16))
        ctk.CTkLabel(c_row, text="Raw Landmarks CSV:", width=140, anchor="w", text_color=TXT_SEC, font=ctk.CTkFont(family=FF, size=12)).pack(side="left")
        ctk.CTkEntry(c_row, textvariable=self.csv_path_var, font=ctk.CTkFont(family=FF, size=12), width=320).pack(side="left", padx=8)
        ctk.CTkButton(c_row, text="Browse...", width=90, fg_color=BTN_SEC, command=self._browse_csv).pack(side="left")

        # ── Card 2: Window Configuration Setup ──────────────────────────────────
        win_card = ctk.CTkFrame(center, corner_radius=6, border_width=1, border_color=BORDER, fg_color=PANEL)
        win_card.pack(fill="x", pady=(0, 24), ipadx=10, ipady=4)

        ctk.CTkLabel(
            win_card, text="⚙  Sliding Window Configuration",
            font=ctk.CTkFont(family=FF, size=14, weight="bold"), text_color=TXT_PRI
        ).pack(anchor="w", padx=20, pady=(16, 12))

        w_row = ctk.CTkFrame(win_card, fg_color="transparent")
        w_row.pack(fill="x", padx=20, pady=(0, 16))

        # Window Size
        ctk.CTkLabel(w_row, text="Window Size (Frames):", text_color=TXT_SEC, font=ctk.CTkFont(family=FF, size=12)).pack(side="left", padx=(0, 8))
        self.ent_wsize = ctk.CTkEntry(w_row, textvariable=self.window_size_var, width=70, font=ctk.CTkFont(family=FF, size=12))
        self.ent_wsize.pack(side="left", padx=(0, 24))

        # Overlap
        ctk.CTkLabel(w_row, text="Overlap (Frames):", text_color=TXT_SEC, font=ctk.CTkFont(family=FF, size=12)).pack(side="left", padx=(0, 8))
        self.ent_overlap = ctk.CTkEntry(w_row, textvariable=self.window_overlap_var, width=70, font=ctk.CTkFont(family=FF, size=12))
        self.ent_overlap.pack(side="left", padx=(0, 16))

        self.lbl_step_info = ctk.CTkLabel(
            w_row, text="(Default: 5 frames size, 2 overlap = 3 frames step)",
            text_color=GREEN, font=ctk.CTkFont(family=FF, size=11)
        )
        self.lbl_step_info.pack(side="left")

        # ── Start Button ────────────────────────────────────────────────────────
        ctk.CTkButton(
            center, text="🚀 Start Window Annotation ▶",
            font=ctk.CTkFont(family=FF, size=14, weight="bold"),
            fg_color=BTN_PRI, hover_color=BTN_HVP,
            height=44, corner_radius=6,
            command=self._on_start_click
        ).pack(fill="x", pady=(0, 10))

    def _browse_video(self):
        from datacreator.annotator.utils import open_video_dialog
        p = open_video_dialog()
        if p:
            self.video_path_var.set(p)

    def _browse_csv(self):
        from datacreator.annotator.utils import open_csv_dialog
        p = open_csv_dialog()
        if p:
            self.csv_path_var.set(p)

    def _on_start_click(self):
        v_path = self.video_path_var.get().strip()
        c_path = self.csv_path_var.get().strip()

        if not v_path or not os.path.exists(v_path):
            messagebox.showerror("Error", "Please select a valid 12 FPS video file!")
            return
        if not c_path or not os.path.exists(c_path):
            messagebox.showerror("Error", "Please select a valid Raw Landmarks CSV file!")
            return

        try:
            w_size = int(self.window_size_var.get().strip())
            overlap = int(self.window_overlap_var.get().strip())
            if w_size <= 0 or overlap < 0 or overlap >= w_size:
                raise ValueError()
        except ValueError:
            messagebox.showerror("Error", "Window Size must be > 0, and Overlap must be between 0 and Window Size - 1!")
            return

        self.app.start_annotation_session(v_path, c_path, w_size, overlap)
