"""
annotator/ui/annotation_screen.py

Main annotation interface styled with modern professional desktop HIG.
Includes 'Select Window...' picker button in top status bar and color-coded
finger touch toggles matching landmark colors.
"""
import logging
import os
from tkinter import messagebox

import customtkinter as ctk
from PIL import Image

from annotator.constants import (
    ANY_DIFF_PRESETS, CSV_HEADERS, FINGERS, FINGER_COLORS_HEX, JOINT_LABELS, POV_OPTIONS,
)
from annotator.csv_manager import CSVManager
from annotator.video_processor import (
    extract_window_record_data, frames_to_pil, get_window_frames,
    window_count, window_idx_from_start_frame,
)

logger = logging.getLogger("Annotator.AnnotationScreen")
_DISPLAY_W = 640   # target width for frame viewer


class AnnotationScreen(ctk.CTkFrame):
    def __init__(
        self, parent, app,
        frame_data, fps, total_frames, duration_ms,
        csv_path, video_path, video_hash,
        start_window_idx: int = 0,
        allow_override_last: bool = False,
    ) -> None:
        super().__init__(parent, fg_color="transparent")
        self.app = app
        self.frame_data = frame_data
        self.fps = fps
        self.total_frames = total_frames
        self.duration_ms = duration_ms
        self.csv_path = csv_path
        self.video_path = video_path
        self.video_hash = video_hash
        self.csv = CSVManager(csv_path)
        self.total_windows = window_count(total_frames)

        self._window_idx = start_window_idx
        self._allow_override_last = allow_override_last
        self._is_at_recovery_start = allow_override_last

        # Loop animation state
        self._loop_idx = 0
        self._loop_job = None
        self._loop_running = False
        self._individual_mode = False
        self._individual_idx = 0
        self._cached_pil: list[Image.Image] = []
        self._current_wf: list = []
        self._display_image_ref = None

        # Annotation variables
        self._touch = {f: ctk.BooleanVar(value=False) for f in FINGERS}
        self._hand_move = ctk.BooleanVar(value=False)
        self._hand_closer = ctk.BooleanVar(value=False)
        self._hovering = ctk.BooleanVar(value=False)
        self._daylight = ctk.BooleanVar(value=True)
        self._hand_visible = ctk.BooleanVar(value=True)
        self._pov = ctk.StringVar(value="front")
        self._any_diff = ctk.StringVar(value="")

        self._build()
        logger.info(f"AnnotationScreen initialized. Total windows={self.total_windows}, starting at index={start_window_idx}")
        self._load_window(self._window_idx)

    # ── UI construction ───────────────────────────────────────────────────────

    def _build(self) -> None:
        # Top Stats & Window Picker Bar
        top = ctk.CTkFrame(self, height=48, corner_radius=0,
                           fg_color=("gray90", "gray14"))
        top.pack(fill="x")
        top.pack_propagate(False)

        self._lbl_stats = ctk.CTkLabel(
            top, text="",
            font=ctk.CTkFont(size=12, weight="bold"),
        )
        self._lbl_stats.pack(side="left", padx=20)

        # Window Selection Button right on the top bar
        self._btn_jump = ctk.CTkButton(
            top, text="Select Window...", width=140, height=30,
            corner_radius=6, font=ctk.CTkFont(size=12, weight="bold"),
            fg_color="#2563eb", hover_color="#1d4ed8",
            command=self._pick_window,
        )
        self._btn_jump.pack(side="right", padx=16)

        self._lbl_recorded = ctk.CTkLabel(
            top, text="",
            font=ctk.CTkFont(size=12),
            text_color=("#d97706", "#f59e0b"),
        )
        self._lbl_recorded.pack(side="right", padx=12)

        # Main Split Frame
        main = ctk.CTkFrame(self, fg_color="transparent")
        main.pack(fill="both", expand=True, padx=12, pady=10)
        main.columnconfigure(0, weight=6)
        main.columnconfigure(1, weight=4)
        main.rowconfigure(0, weight=1)

        self._build_viewer(main)
        self._build_controls(main)

    def _build_viewer(self, parent) -> None:
        left = ctk.CTkFrame(
            parent, corner_radius=8,
            border_width=1, border_color=("gray80", "gray25"),
            fg_color=("gray95", "gray14"),
        )
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 6))
        left.rowconfigure(0, weight=1)
        left.columnconfigure(0, weight=1)

        # Video Frame Display Area
        self._frame_lbl = ctk.CTkLabel(
            left, text="", fg_color="black", corner_radius=6,
        )
        self._frame_lbl.grid(row=0, column=0, sticky="nsew", padx=8, pady=8)

        # Frame Info Bar
        info = ctk.CTkFrame(left, fg_color="transparent")
        info.grid(row=1, column=0, sticky="ew", padx=12, pady=(0, 4))
        self._lbl_fnum = ctk.CTkLabel(
            info, text="",
            font=ctk.CTkFont(size=12),
            text_color=("gray30", "gray70"),
        )
        self._lbl_fnum.pack(side="left")

        # Playback Controls Bar
        speed_row = ctk.CTkFrame(left, fg_color="transparent")
        speed_row.grid(row=2, column=0, sticky="ew", padx=12, pady=(0, 6))

        ctk.CTkLabel(
            speed_row, text="Loop Speed:",
            font=ctk.CTkFont(size=12),
        ).pack(side="left", padx=(0, 8))

        self._speed_slider = ctk.CTkSlider(
            speed_row, from_=1, to=10,
            number_of_steps=9, width=140,
            command=self._speed_changed,
        )
        self._speed_slider.set(3)
        self._speed_slider.pack(side="left")

        self._speed_lbl = ctk.CTkLabel(
            speed_row, text="3 FPS",
            font=ctk.CTkFont(size=12),
        )
        self._speed_lbl.pack(side="left", padx=8)

        # Step Buttons
        nav = ctk.CTkFrame(left, fg_color="transparent")
        nav.grid(row=3, column=0, pady=(0, 12))

        ctk.CTkButton(
            nav, text="Play Loop", width=95, height=30, corner_radius=6,
            font=ctk.CTkFont(size=12),
            fg_color="#2563eb", hover_color="#1d4ed8",
            command=self._start_loop,
        ).pack(side="left", padx=4)

        ctk.CTkButton(
            nav, text="Step Back", width=85, height=30, corner_radius=6,
            font=ctk.CTkFont(size=12),
            fg_color=("gray80", "gray28"), hover_color=("gray70", "gray36"),
            text_color=("gray10", "gray95"),
            command=self._prev_frame,
        ).pack(side="left", padx=4)

        ctk.CTkButton(
            nav, text="Step Forward", width=95, height=30, corner_radius=6,
            font=ctk.CTkFont(size=12),
            fg_color=("gray80", "gray28"), hover_color=("gray70", "gray36"),
            text_color=("gray10", "gray95"),
            command=self._next_frame,
        ).pack(side="left", padx=4)

    def _build_controls(self, parent) -> None:
        right = ctk.CTkScrollableFrame(
            parent, corner_radius=8, width=340,
            border_width=1, border_color=("gray80", "gray25"),
            fg_color=("gray95", "gray14"),
        )
        right.grid(row=0, column=1, sticky="nsew", padx=(6, 0))

        pad = {"padx": 16, "pady": 4}

        # ── Finger Touch Flags ───────────────────────────────────────────────
        self._section(right, "Finger Touch Flags")
        tg = ctk.CTkFrame(right, fg_color="transparent")
        tg.pack(fill="x", **pad)
        for col, finger in enumerate(FINGERS):
            hex_col = FINGER_COLORS_HEX.get(finger, "gray95")
            ctk.CTkCheckBox(
                tg, text=finger, variable=self._touch[finger],
                font=ctk.CTkFont(size=13, weight="bold"),
                text_color=hex_col,
                checkmark_color=hex_col,
            ).grid(
                row=col // 3, column=col % 3, padx=6, pady=4, sticky="w"
            )

        # ── Hand Motion & Environment ────────────────────────────────────────
        self._section(right, "Hand Motion & Environment")
        for text, var in [
            ("Hand Moving", self._hand_move),
            ("Hand Moving Closer", self._hand_closer),
            ("Hovering (Not Touching)", self._hovering),
            ("Daylight Lighting", self._daylight),
            ("Hand Visible", self._hand_visible),
        ]:
            ctk.CTkSwitch(
                right, text=text, variable=var,
                font=ctk.CTkFont(size=13),
            ).pack(anchor="w", **pad)

        # ── Point of View ────────────────────────────────────────────────────
        self._section(right, "Point of View (POV)")
        pov_row = ctk.CTkFrame(right, fg_color="transparent")
        pov_row.pack(fill="x", **pad)
        for opt in POV_OPTIONS:
            ctk.CTkRadioButton(
                pov_row, text=opt.capitalize(),
                variable=self._pov, value=opt,
                font=ctk.CTkFont(size=13),
            ).pack(side="left", padx=10)

        # ── Observation Notes ────────────────────────────────────────────────
        self._section(right, "Observation Notes (anyDifference)")
        self._diff_combo = ctk.CTkComboBox(
            right, variable=self._any_diff,
            values=ANY_DIFF_PRESETS, width=290,
            font=ctk.CTkFont(size=12),
        )
        self._diff_combo.pack(anchor="w", **pad)

        ctk.CTkFrame(right, height=1, fg_color=("gray80", "gray25")).pack(
            fill="x", padx=16, pady=12
        )

        # Override toggle switch (if resuming last record)
        self._override_frame = ctk.CTkFrame(right, fg_color="transparent")
        self._override_frame.pack(fill="x", **pad)
        self._override_var = ctk.BooleanVar(value=False)
        self._override_sw = ctk.CTkSwitch(
            self._override_frame, text="Override Record on Next",
            variable=self._override_var,
            font=ctk.CTkFont(size=12), text_color=("#d97706", "#f59e0b"),
        )

        ctk.CTkFrame(right, height=1, fg_color=("gray80", "gray25")).pack(
            fill="x", padx=16, pady=8
        )

        # Action Buttons Navigation
        nav_row = ctk.CTkFrame(right, fg_color="transparent")
        nav_row.pack(fill="x", **pad)

        self._btn_prev = ctk.CTkButton(
            nav_row, text="Previous Window", width=130, height=36,
            corner_radius=6, font=ctk.CTkFont(size=12),
            fg_color=("gray80", "gray28"), hover_color=("gray70", "gray36"),
            text_color=("gray10", "gray95"),
            command=self._go_prev,
        )
        self._btn_prev.pack(side="left", padx=4)

        self._btn_next = ctk.CTkButton(
            nav_row, text="Next Window", width=130, height=36,
            corner_radius=6, font=ctk.CTkFont(size=12, weight="bold"),
            fg_color="#2563eb", hover_color="#1d4ed8",
            command=self._go_next,
        )
        self._btn_next.pack(side="left", padx=4)

        ctk.CTkButton(
            right, text="Return to Setup",
            width=160, height=28, corner_radius=6,
            fg_color="transparent", hover_color=("gray85", "gray22"),
            border_width=1, border_color=("gray70", "gray35"),
            text_color=("gray40", "gray60"),
            font=ctk.CTkFont(size=11),
            command=self._exit,
        ).pack(pady=(12, 8))

    def _section(self, parent, title: str) -> None:
        ctk.CTkLabel(
            parent, text=title,
            font=ctk.CTkFont(size=13, weight="bold"),
            text_color=("gray20", "gray80"),
        ).pack(anchor="w", padx=16, pady=(12, 2))

    # ── Window loading & Navigation ───────────────────────────────────────────

    def _load_window(self, widx: int) -> None:
        logger.info(f"Loading window index {widx}/{self.total_windows}")
        self._window_idx = widx
        wf = get_window_frames(self.frame_data, widx)
        if wf is None:
            logger.info("Reached end of video (no more 5-frame windows).")
            self._video_end()
            return

        self._current_wf = wf
        self._cached_pil = frames_to_pil(wf)
        self._individual_idx = 0

        sf = wf[0]["frame_idx"]
        ef = wf[-1]["frame_idx"]
        sms = wf[0]["timestamp_ms"]
        ems = wf[-1]["timestamp_ms"]
        remaining = max(0, self.duration_ms - ems)

        self._lbl_stats.configure(
            text=(f"Window {widx + 1} / {self.total_windows}"
                  f"   ·   Frames {sf}–{ef}"
                  f"   ·   {sms}ms–{ems}ms"
                  f"   ·   Remaining: {remaining}ms")
        )

        rec_idx, existing = self.csv.find(sf, ef, sms, ems)
        if existing:
            logger.info(f"Found existing record in CSV at index #{rec_idx + 1} for frames {sf}–{ef}")
            self._lbl_recorded.configure(
                text=f"Already Recorded (Record #{rec_idx + 1})"
            )
            self._populate(existing)
        else:
            logger.info(f"No existing record found for frames {sf}–{ef}. Resetting controls to default.")
            self._lbl_recorded.configure(text="")
            self._reset_annotation()

        at_recovery_entry = self._is_at_recovery_start and widx == self._window_idx
        if self._allow_override_last and at_recovery_entry:
            self._override_sw.pack(anchor="w")
        else:
            self._override_sw.pack_forget()

        self._btn_prev.configure(state="normal" if widx > 0 else "disabled")
        self._start_loop()

    def _pick_window(self) -> None:
        logger.info(f"User clicked 'Select Window...' (current window index = {self._window_idx})")
        from annotator.ui.record_picker import RecordPickerDialog

        dialog = RecordPickerDialog(
            self,
            total_windows=self.total_windows,
            current_window_idx=self._window_idx,
            frame_data=self.frame_data,
            csv_manager=self.csv,
        )
        self.wait_window(dialog)

        if dialog.selected_window_idx is not None and dialog.selected_window_idx != self._window_idx:
            logger.info(f"User selected window index {dialog.selected_window_idx}. Jumping to window...")
            if not self._save_current():
                return
            self._stop_loop()
            self._is_at_recovery_start = False
            self._allow_override_last = False
            self._load_window(dialog.selected_window_idx)

    # ── Annotation helpers ────────────────────────────────────────────────────

    def _populate(self, r: dict) -> None:
        for fn in FINGERS:
            self._touch[fn].set(str(r.get(f"{fn.lower()}_touch", "False")).lower() == "true")
        self._hand_move.set(str(r.get("hand_move", "False")).lower() == "true")
        self._hand_closer.set(str(r.get("hand_closer", "False")).lower() == "true")
        self._hovering.set(str(r.get("hovering", "False")).lower() == "true")
        self._daylight.set(str(r.get("daylight", "True")).lower() == "true")
        self._hand_visible.set(str(r.get("hand_visible", "True")).lower() == "true")
        self._pov.set(r.get("hand_point_of_view", "front"))
        self._any_diff.set(r.get("any_difference", ""))

    def _reset_annotation(self) -> None:
        for fn in FINGERS:
            self._touch[fn].set(False)
        self._hand_move.set(False)
        self._hand_closer.set(False)
        self._hovering.set(False)
        self._daylight.set(True)
        self._hand_visible.set(True)
        self._pov.set("front")
        self._any_diff.set("")

    def _annotation_dict(self) -> dict:
        ann: dict = {f"{fn.lower()}_touch": self._touch[fn].get() for fn in FINGERS}
        ann.update({
            "hand_move": self._hand_move.get(),
            "hand_point_of_view": self._pov.get(),
            "hand_closer": self._hand_closer.get(),
            "hovering": self._hovering.get(),
            "daylight": self._daylight.get(),
            "hand_visible": self._hand_visible.get(),
            "any_difference": self._any_diff.get().strip(),
        })
        return ann

    def _records_same(self, existing: dict, new_rec: dict) -> bool:
        ann_fields = (
            [f"{fn.lower()}_touch" for fn in FINGERS]
            + ["hand_move", "hand_point_of_view", "hand_closer",
               "hovering", "daylight", "hand_visible", "any_difference"]
        )
        for f in ann_fields:
            if str(existing.get(f, "")).strip().lower() != str(new_rec.get(f, "")).strip().lower():
                return False
        return True

    # ── Loop animation ────────────────────────────────────────────────────────

    def _start_loop(self) -> None:
        self._individual_mode = False
        self._loop_idx = 0
        self._stop_loop()
        self._loop_running = True
        self._tick()

    def _stop_loop(self) -> None:
        self._loop_running = False
        if self._loop_job is not None:
            try:
                self.after_cancel(self._loop_job)
            except Exception:
                pass
            self._loop_job = None

    def _tick(self) -> None:
        if not self._loop_running or self._individual_mode or not self._cached_pil:
            return
        self._show(self._loop_idx)
        self._loop_idx = (self._loop_idx + 1) % len(self._cached_pil)
        delay = max(80, int(1000 / self._speed_slider.get()))
        self._loop_job = self.after(delay, self._tick)

    def _show(self, local_idx: int) -> None:
        if not self._cached_pil or local_idx >= len(self._cached_pil):
            return
        pil = self._cached_pil[local_idx]
        fd = self._current_wf[local_idx]

        try:
            avail_w = max(300, self.winfo_width() * 60 // 100 - 30)
            avail_h = max(240, self.winfo_height() - 200)
        except Exception:
            avail_w, avail_h = _DISPLAY_W, 460

        scaled = pil.copy()
        scaled.thumbnail((avail_w, avail_h), Image.LANCZOS)
        ctk_img = ctk.CTkImage(
            light_image=scaled, dark_image=scaled,
            size=(scaled.width, scaled.height),
        )
        self._display_image_ref = ctk_img
        self._frame_lbl.configure(image=ctk_img, text="")

        hand_state = "Hand Detected" if fd["hand_data"] else "No Hand Detected"
        self._lbl_fnum.configure(
            text=(f"Frame {fd['frame_idx']}"
                  f"   ·   {fd['timestamp_ms']} ms"
                  f"   ·   Position {local_idx + 1}/5"
                  f"   ·   {hand_state}")
        )

    def _speed_changed(self, val) -> None:
        fps = int(float(val))
        self._speed_lbl.configure(text=f"{fps} FPS")

    def _prev_frame(self) -> None:
        self._stop_loop()
        self._individual_mode = True
        self._individual_idx = max(0, self._individual_idx - 1)
        self._show(self._individual_idx)

    def _next_frame(self) -> None:
        self._stop_loop()
        self._individual_mode = True
        n = len(self._cached_pil) - 1
        self._individual_idx = min(n, self._individual_idx + 1)
        self._show(self._individual_idx)

    # ── Save & navigation ─────────────────────────────────────────────────────

    def _build_record(self) -> dict:
        wf = self._current_wf
        base = extract_window_record_data(
            wf,
            os.path.basename(self.video_path),
            self.video_hash,
            self.duration_ms,
        )
        ann = self._annotation_dict()
        rec = {**base, **ann}
        for h in CSV_HEADERS:
            rec.setdefault(h, "")
        return rec

    def _save_current(self) -> bool:
        wf = self._current_wf
        sf = wf[0]["frame_idx"]
        ef = wf[-1]["frame_idx"]
        sms = wf[0]["timestamp_ms"]
        ems = wf[-1]["timestamp_ms"]

        new_rec = self._build_record()
        rec_idx, existing = self.csv.find(sf, ef, sms, ems)

        if existing is None:
            logger.info(f"Saving new record for frames {sf}–{ef}")
            self.csv.append(new_rec)
            return True

        if self._records_same(existing, new_rec):
            logger.info(f"Record for frames {sf}–{ef} exists and is identical. Skipping rewrite.")
            return True

        logger.info(f"Record for frames {sf}–{ef} has modifications. Prompting user to override...")
        do_override = messagebox.askyesno(
            "Override Record?",
            f"A record for frames {sf}–{ef} already exists.\n\n"
            "Your annotation has changed. Override it?",
            parent=self,
        )
        if do_override:
            logger.info(f"User accepted override for frames {sf}–{ef}")
            self.csv.override(sf, ef, sms, ems, new_rec)
            return True
        logger.info(f"User rejected override for frames {sf}–{ef}")
        return False

    def _go_next(self) -> None:
        logger.info("User clicked 'Next Window'")
        if not self._save_current():
            return
        self._is_at_recovery_start = False
        self._allow_override_last = False
        next_widx = self._window_idx + 1
        if get_window_frames(self.frame_data, next_widx) is None:
            self._video_end()
        else:
            self._load_window(next_widx)

    def _go_prev(self) -> None:
        logger.info("User clicked 'Previous Window'")
        if self._window_idx > 0:
            self._stop_loop()
            self._load_window(self._window_idx - 1)

    def _video_end(self) -> None:
        self._stop_loop()
        logger.info("All video windows processed and annotated!")
        messagebox.showinfo(
            "Annotation Complete",
            f"All video windows annotated successfully.\n\n"
            f"Total records saved: {self.csv.total_records()}\n"
            f"CSV file path: {self.csv_path}",
            parent=self,
        )
        self.app.show_setup()

    def _exit(self) -> None:
        if messagebox.askyesno(
            "Exit Annotator",
            "Return to the setup screen?\n"
            "Unsaved changes on the current window will be lost.",
            parent=self,
        ):
            self._stop_loop()
            self.app.show_setup()

    def destroy(self) -> None:
        self._stop_loop()
        super().destroy()
