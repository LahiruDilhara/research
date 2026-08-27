"""
annotator/ui/setup_screen.py

Professional setup screen styled with exact VS Code Dark+ charcoal palette.
Background: #1e1e1e | Cards: #252526 | Borders: #3c3c3c | Buttons: #007acc & #3c3c3c
Smooth vector typography with crisp 4px rectangular corner geometry.
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
FONT_FAMILY = "Helvetica"


class SetupScreen(ctk.CTkFrame):
    def __init__(self, parent, app) -> None:
        super().__init__(parent, fg_color="#1e1e1e")  # VS Code Main Dark Background
        self.app = app
        self._build()
        logger.info("SetupScreen built successfully with VS Code Dark palette.")

    def _build(self) -> None:
        # Center container
        center = ctk.CTkFrame(self, fg_color="transparent")
        center.place(relx=0.5, rely=0.5, anchor="center")

        # Application Title & Header
        title_lbl = ctk.CTkLabel(
            center,
            text="Touch Gesture Data Annotator",
            font=ctk.CTkFont(family=FONT_FAMILY, size=26, weight="bold"),
            text_color="#ffffff",
        )
        title_lbl.pack(pady=(0, 6))

        subtitle_lbl = ctk.CTkLabel(
            center,
            text="MediaPipe Sliding-Window Feature Extractor & Touch Annotator",
            font=ctk.CTkFont(family=FONT_FAMILY, size=13),
            text_color="#cccccc",
        )
        subtitle_lbl.pack(pady=(0, 20))

        # ── Sliding Window Configuration Card ─────────────────────────────────
        config_card = ctk.CTkFrame(
            center,
            corner_radius=4,
            border_width=1,
            border_color="#3c3c3c",
            fg_color="#252526",
        )
        config_card.pack(fill="x", pady=(0, 20), padx=14)

        config_inner = ctk.CTkFrame(config_card, fg_color="transparent")
        config_inner.pack(padx=20, pady=16, fill="x")

        ctk.CTkLabel(
            config_inner,
            text="⚙  Sliding Window Configuration",
            font=ctk.CTkFont(family=FONT_FAMILY, size=14, weight="bold"),
            text_color="#ffffff",
        ).pack(anchor="w", pady=(0, 10))

        inputs_row = ctk.CTkFrame(config_inner, fg_color="transparent")
        inputs_row.pack(fill="x")

        # Window Size Input
        ctk.CTkLabel(
            inputs_row,
            text="Window Size (Frames):",
            font=ctk.CTkFont(family=FONT_FAMILY, size=12),
            text_color="#cccccc",
        ).pack(side="left", padx=(0, 8))

        from annotator.constants import WINDOW_SIZE, WINDOW_OVERLAP, set_window_config
        self._ent_window_size = ctk.CTkEntry(
            inputs_row, width=70, height=28,
            font=ctk.CTkFont(family=FONT_FAMILY, size=12),
        )
        self._ent_window_size.insert(0, str(WINDOW_SIZE))
        self._ent_window_size.pack(side="left", padx=(0, 24))

        # Window Overlap Input
        ctk.CTkLabel(
            inputs_row,
            text="Overlap (Frames):",
            font=ctk.CTkFont(family=FONT_FAMILY, size=12),
            text_color="#cccccc",
        ).pack(side="left", padx=(0, 8))

        self._ent_window_overlap = ctk.CTkEntry(
            inputs_row, width=70, height=28,
            font=ctk.CTkFont(family=FONT_FAMILY, size=12),
        )
        self._ent_window_overlap.insert(0, str(WINDOW_OVERLAP))
        self._ent_window_overlap.pack(side="left", padx=(0, 24))

        self._lbl_config_status = ctk.CTkLabel(
            config_inner,
            text=f"Active: {WINDOW_SIZE}-frame window  ·  {WINDOW_OVERLAP}-frame overlap  ·  step={WINDOW_SIZE - WINDOW_OVERLAP} frames",
            font=ctk.CTkFont(family=FONT_FAMILY, size=11),
            text_color="#00c8ff",
        )
        self._lbl_config_status.pack(anchor="w", pady=(8, 0))

        # Cards Layout Container
        cards_frame = ctk.CTkFrame(center, fg_color="transparent")
        cards_frame.pack(fill="x", expand=True)

        # ── Card 1: Create New Dataset ─────────────────────────────────────────
        card_new = ctk.CTkFrame(
            cards_frame,
            width=400,
            corner_radius=4,
            border_width=1,
            border_color="#3c3c3c",   # VS Code Border
            fg_color="#252526",       # VS Code Panel Background
        )
        card_new.pack(side="left", padx=14, pady=8, fill="both", expand=True)

        new_inner = ctk.CTkFrame(card_new, fg_color="transparent")
        new_inner.pack(padx=26, pady=26, fill="both", expand=True)

        ctk.CTkLabel(
            new_inner,
            text="New Dataset",
            font=ctk.CTkFont(family=FONT_FAMILY, size=17, weight="bold"),
            text_color="#ffffff",
            anchor="w",
        ).pack(anchor="w", pady=(0, 8))

        ctk.CTkLabel(
            new_inner,
            text="Select a video file to run the MediaPipe pipeline and specify a CSV location to record new sequence feature dataset.",
            font=ctk.CTkFont(family=FONT_FAMILY, size=12),
            text_color="#cccccc",
            justify="left",
            wraplength=340,
        ).pack(anchor="w", pady=(0, 24))

        ctk.CTkButton(
            new_inner,
            text="Create New Dataset",
            height=40,
            corner_radius=4,
            border_width=0,
            font=ctk.CTkFont(family=FONT_FAMILY, size=13, weight="bold"),
            fg_color="#007acc",       # VS Code Blue Primary
            hover_color="#0062a3",
            text_color="#ffffff",
            command=self._create_new,
        ).pack(fill="x")

        # ── Card 2: Resume Existing Session ───────────────────────────────────
        card_open = ctk.CTkFrame(
            cards_frame,
            width=400,
            corner_radius=4,
            border_width=1,
            border_color="#3c3c3c",
            fg_color="#252526",
        )
        card_open.pack(side="left", padx=14, pady=8, fill="both", expand=True)

        open_inner = ctk.CTkFrame(card_open, fg_color="transparent")
        open_inner.pack(padx=26, pady=26, fill="both", expand=True)

        ctk.CTkLabel(
            open_inner,
            text="Resume Session",
            font=ctk.CTkFont(family=FONT_FAMILY, size=17, weight="bold"),
            text_color="#ffffff",
            anchor="w",
        ).pack(anchor="w", pady=(0, 8))

        ctk.CTkLabel(
            open_inner,
            text="Open an existing CSV file to resume annotation from your last saved state. Video fingerprint integrity will be verified automatically.",
            font=ctk.CTkFont(family=FONT_FAMILY, size=12),
            text_color="#cccccc",
            justify="left",
            wraplength=340,
        ).pack(anchor="w", pady=(0, 24))

        ctk.CTkButton(
            open_inner,
            text="Open Existing CSV",
            height=40,
            corner_radius=4,
            border_width=0,
            font=ctk.CTkFont(family=FONT_FAMILY, size=13, weight="bold"),
            fg_color="#3c3c3c",       # VS Code Button Dark Neutral
            hover_color="#4c4c4c",
            text_color="#ffffff",
            command=self._open_existing,
        ).pack(fill="x")

    def _validate_and_apply_config(self) -> bool:
        try:
            sz = int(self._ent_window_size.get().strip())
            ov = int(self._ent_window_overlap.get().strip())
            if sz < 1:
                messagebox.showerror("Invalid Setting", "Window size must be at least 1 frame.")
                return False
            if ov < 0 or ov >= sz:
                messagebox.showerror(
                    "Invalid Setting",
                    f"Overlap must be between 0 and window size - 1 (0 to {sz - 1})."
                )
                return False
            from annotator.constants import set_window_config
            w_size, w_overlap, w_step = set_window_config(sz, ov)
            self._lbl_config_status.configure(
                text=f"Active: {w_size}-frame window  ·  {w_overlap}-frame overlap  ·  step={w_step} frames"
            )
            return True
        except ValueError:
            messagebox.showerror("Invalid Input", "Window size and overlap must be integer numbers.")
            return False

    # ── Actions ───────────────────────────────────────────────────────────────

    def _create_new(self) -> None:
        logger.info("User clicked 'Create New Dataset'")
        if not self._validate_and_apply_config():
            return
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
        if not self._validate_and_apply_config():
            return
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
