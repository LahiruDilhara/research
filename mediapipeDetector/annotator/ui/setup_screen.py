"""
annotator/ui/setup_screen.py

Professional, modern launch screen using CustomTkinter widgets.
Clean typography, subtle card borders, neutral dark palette, no game-like elements.
"""
import logging
import os
from tkinter import messagebox

import customtkinter as ctk

from annotator.utils import (
    extract_hash_from_csv_filename, open_csv_dialog, open_video_dialog,
    save_csv_dialog,
)

logger = logging.getLogger("Annotator.SetupScreen")


class SetupScreen(ctk.CTkFrame):
    def __init__(self, parent, app) -> None:
        super().__init__(parent, fg_color="transparent")
        self.app = app
        self._build()
        logger.info("SetupScreen built successfully.")

    def _build(self) -> None:
        # Center container
        center = ctk.CTkFrame(self, fg_color="transparent")
        center.place(relx=0.5, rely=0.5, anchor="center")

        # Application Title & Header
        title_lbl = ctk.CTkLabel(
            center,
            text="Touch Gesture Data Annotator",
            font=ctk.CTkFont(size=28, weight="bold"),
            text_color=("gray10", "gray95"),
        )
        title_lbl.pack(pady=(0, 6))

        subtitle_lbl = ctk.CTkLabel(
            center,
            text="MediaPipe 5-Frame Sliding-Window Feature Extractor & Touch Annotator",
            font=ctk.CTkFont(size=13),
            text_color=("gray50", "gray60"),
        )
        subtitle_lbl.pack(pady=(0, 36))

        # Cards Layout Container
        cards_frame = ctk.CTkFrame(center, fg_color="transparent")
        cards_frame.pack(fill="x", expand=True)

        # ── Card 1: Create New Dataset ─────────────────────────────────────────
        card_new = ctk.CTkFrame(
            cards_frame,
            width=420,
            corner_radius=10,
            border_width=1,
            border_color=("gray80", "gray25"),
            fg_color=("gray95", "gray14"),
        )
        card_new.pack(side="left", padx=12, pady=8, fill="both", expand=True)

        new_inner = ctk.CTkFrame(card_new, fg_color="transparent")
        new_inner.pack(padx=24, pady=24, fill="both", expand=True)

        ctk.CTkLabel(
            new_inner,
            text="New Dataset",
            font=ctk.CTkFont(size=18, weight="bold"),
            anchor="w",
        ).pack(anchor="w", pady=(0, 6))

        ctk.CTkLabel(
            new_inner,
            text="Select a video file to run the MediaPipe pipeline and specify a CSV location to record new 83-feature touch gesture data.",
            font=ctk.CTkFont(size=12),
            text_color=("gray40", "gray60"),
            justify="left",
            wraplength=360,
        ).pack(anchor="w", pady=(0, 24))

        ctk.CTkButton(
            new_inner,
            text="Create New Dataset",
            height=40,
            corner_radius=6,
            font=ctk.CTkFont(size=13, weight="bold"),
            fg_color="#2563eb",
            hover_color="#1d4ed8",
            command=self._create_new,
        ).pack(fill="x")

        # ── Card 2: Resume Existing Session ───────────────────────────────────
        card_open = ctk.CTkFrame(
            cards_frame,
            width=420,
            corner_radius=10,
            border_width=1,
            border_color=("gray80", "gray25"),
            fg_color=("gray95", "gray14"),
        )
        card_open.pack(side="left", padx=12, pady=8, fill="both", expand=True)

        open_inner = ctk.CTkFrame(card_open, fg_color="transparent")
        open_inner.pack(padx=24, pady=24, fill="both", expand=True)

        ctk.CTkLabel(
            open_inner,
            text="Resume Session",
            font=ctk.CTkFont(size=18, weight="bold"),
            anchor="w",
        ).pack(anchor="w", pady=(0, 6))

        ctk.CTkLabel(
            open_inner,
            text="Open an existing CSV file to resume annotation from your last saved state. Video fingerprint integrity will be verified automatically.",
            font=ctk.CTkFont(size=12),
            text_color=("gray40", "gray60"),
            justify="left",
            wraplength=360,
        ).pack(anchor="w", pady=(0, 24))

        ctk.CTkButton(
            open_inner,
            text="Open Existing CSV",
            height=40,
            corner_radius=6,
            font=ctk.CTkFont(size=13, weight="bold"),
            fg_color=("gray80", "gray28"),
            hover_color=("gray70", "gray36"),
            text_color=("gray10", "gray95"),
            command=self._open_existing,
        ).pack(fill="x")

        # Pipeline Footer Specs
        ctk.CTkLabel(
            center,
            text="Pipeline Spec: 5-frame window  ·  2-frame overlap  ·  1€ filter  ·  deadband threshold 0.4",
            font=ctk.CTkFont(size=11),
            text_color=("gray50", "gray50"),
        ).pack(pady=(32, 0))

    # ── Actions ───────────────────────────────────────────────────────────────

    def _create_new(self) -> None:
        logger.info("User clicked 'Create New Dataset'")
        video_path = open_video_dialog()
        if not video_path or not os.path.isfile(video_path):
            logger.info("No video selected or invalid path. Aborting setup.")
            return

        raw_path = save_csv_dialog(initial_name="annotation_data")
        if not raw_path:
            logger.info("No CSV save path selected. Aborting setup.")
            return

        csv_base = os.path.splitext(os.path.basename(raw_path))[0]
        csv_dir = os.path.dirname(raw_path) or os.getcwd()

        logger.info(f"Proceeding to ProcessingScreen: video='{video_path}', csv_base='{csv_base}', csv_dir='{csv_dir}'")
        self.app.show_processing(
            video_path=video_path, mode="new",
            csv_dir=csv_dir, csv_base=csv_base,
        )

    def _open_existing(self) -> None:
        logger.info("User clicked 'Resume Session'")
        csv_path = open_csv_dialog()
        if not csv_path or not os.path.isfile(csv_path):
            logger.info("No CSV selected. Aborting resume setup.")
            return

        csv_hash_expected = extract_hash_from_csv_filename(csv_path)
        if csv_hash_expected is None:
            logger.warning("Could not extract hash from CSV filename.")
            messagebox.showwarning(
                "Unknown Format",
                "Cannot extract video hash from CSV filename.\n"
                "Hash verification will be skipped.",
            )

        video_path = open_video_dialog()
        if not video_path or not os.path.isfile(video_path):
            logger.info("No video selected for resume. Aborting resume setup.")
            return

        logger.info(f"Proceeding to ProcessingScreen (resume): video='{video_path}', csv='{csv_path}'")
        self.app.show_processing(
            video_path=video_path, mode="resume",
            csv_path=csv_path,
            csv_hash_expected=csv_hash_expected,
        )
