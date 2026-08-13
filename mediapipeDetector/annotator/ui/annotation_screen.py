"""
annotator/ui/annotation_screen.py

Main annotation interface — VS Code Dark+ theme, Helvetica fonts.
Includes: 'Select Window...' picker, visual HUD overlay toggle for per-frame joint
(x,y) coordinates and (vx,vy) velocities, color-coded finger touch toggles,
'Reset Window Changes', Yes/No/Cancel save modal when Auto-Save is OFF and edits
exist, automatic carry-forward of environmental/motion settings, 'Out of Sync'
toggle, and 'Auto-Save on Navigation' toggle.
Global scroll routing is handled by app.py — no per-frame binding needed.
"""
import logging
import os
from tkinter import messagebox

import customtkinter as ctk
from PIL import Image, ImageDraw

from annotator.constants import (
    ANY_DIFF_PRESETS, CSV_HEADERS, FINGERS, FINGER_COLORS_HEX, JOINT_LABELS, POV_OPTIONS,
)
from annotator.csv_manager import CSVManager
from annotator.video_processor import (
    extract_window_record_data, frames_to_pil, get_window_frames,
    window_count, window_idx_from_start_frame,
)
from annotator.ui.theme import (
    FF, BG, PANEL, BORDER, HDR_BG, TXT_PRI, TXT_SEC, TXT_MUT,
    BTN_PRI, BTN_HVP, BTN_SEC, BTN_HVS, BTN_GHO, BTN_GHH,
    BTN_PUR, BTN_PUR_H, AMBER, SW_TRACK, SECTION_C, make_switch, section_label,
)

logger = logging.getLogger("Annotator.AnnotationScreen")
_DISPLAY_W = 640


class AnnotationScreen(ctk.CTkFrame):
    def __init__(
        self, parent, app,
        frame_data, fps, total_frames, duration_ms,
        csv_path, video_path, video_hash,
        start_window_idx: int = 0,
    ) -> None:
        super().__init__(parent, fg_color=BG)
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
        self._loop_idx = 0
        self._loop_job = None
        self._loop_running = False
        self._individual_mode = False
        self._individual_idx = 0
        self._cached_pil: list[Image.Image] = []
        self._current_wf: list = []
        self._display_image_ref = None

        self._show_coords_vel = ctk.BooleanVar(value=False)
        self._auto_save = ctk.BooleanVar(value=False)

        self._touch = {f: ctk.BooleanVar(value=False) for f in FINGERS}
        self._right_hand = ctk.BooleanVar(value=False)
        self._hand_move = ctk.BooleanVar(value=False)
        self._hand_closer = ctk.BooleanVar(value=False)
        self._hovering = ctk.BooleanVar(value=False)
        self._daylight = ctk.BooleanVar(value=True)
        self._hand_visible = ctk.BooleanVar(value=True)
        self._out_of_sync = ctk.BooleanVar(value=False)
        self._pov = ctk.StringVar(value="front")
        self._any_diff = ctk.StringVar(value="")

        self._build()
        logger.info(f"AnnotationScreen initialized. Total windows={self.total_windows}, starting at index={start_window_idx}")
        self._load_window(self._window_idx)

    # ── UI construction ───────────────────────────────────────────────────────

    def _build(self) -> None:
        # ── Top status bar ────────────────────────────────────────────────────
        top = ctk.CTkFrame(self, height=48, corner_radius=0, fg_color=HDR_BG)
        top.pack(fill="x")
        top.pack_propagate(False)

        self._lbl_stats = ctk.CTkLabel(
            top, text="",
            font=ctk.CTkFont(family=FF, size=12, weight="bold"),
            text_color=TXT_SEC,
        )
        self._lbl_stats.pack(side="left", padx=20)

        self._btn_analyze = ctk.CTkButton(
            top, text="Analyze Window", width=140, height=30,
            corner_radius=4, border_width=0,
            font=ctk.CTkFont(family=FF, size=12, weight="bold"),
            fg_color=BTN_PUR, hover_color=BTN_PUR_H,
            command=self._open_analyzer,
        )
        self._btn_analyze.pack(side="right", padx=(8, 16))

        self._btn_jump = ctk.CTkButton(
            top, text="Select Window...", width=140, height=30,
            corner_radius=4, border_width=0,
            font=ctk.CTkFont(family=FF, size=12, weight="bold"),
            fg_color=BTN_PRI, hover_color=BTN_HVP,
            command=self._pick_window,
        )
        self._btn_jump.pack(side="right", padx=8)

        # HUD debug toggle — use make_switch with a specific amber/blue color
        self._sw_debug = ctk.CTkSwitch(
            top,
            text="Show (x,y) & (vx,vy)",
            variable=self._show_coords_vel,
            font=ctk.CTkFont(family=FF, size=12),
            text_color="#4ec9f0",
            fg_color=SW_TRACK,
            progress_color=BTN_PRI,
            button_color="#e0e0e0",
            button_hover_color="#ffffff",
            bg_color=HDR_BG,
            command=self._on_toggle_debug_overlay,
        )
        self._sw_debug.pack(side="right", padx=16)

        self._lbl_recorded = ctk.CTkLabel(
            top, text="",
            font=ctk.CTkFont(family=FF, size=12),
            text_color=AMBER,
        )
        self._lbl_recorded.pack(side="right", padx=12)

        # ── Main split layout ──────────────────────────────────────────────────
        main = ctk.CTkFrame(self, fg_color="transparent")
        main.pack(fill="both", expand=True, padx=12, pady=10)
        main.columnconfigure(0, weight=6)
        main.columnconfigure(1, weight=4)
        main.rowconfigure(0, weight=1)

        self._build_viewer(main)
        self._build_controls(main)

    def _build_viewer(self, parent) -> None:
        left = ctk.CTkFrame(
            parent, corner_radius=4,
            border_width=1, border_color=BORDER,
            fg_color=PANEL,
        )
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 6))
        left.rowconfigure(0, weight=1)
        left.columnconfigure(0, weight=1)

        self._frame_lbl = ctk.CTkLabel(left, text="", fg_color="black", corner_radius=4)
        self._frame_lbl.grid(row=0, column=0, sticky="nsew", padx=8, pady=8)

        info = ctk.CTkFrame(left, fg_color="transparent")
        info.grid(row=1, column=0, sticky="ew", padx=12, pady=(0, 4))
        self._lbl_fnum = ctk.CTkLabel(
            info, text="",
            font=ctk.CTkFont(family=FF, size=12), text_color=TXT_MUT,
        )
        self._lbl_fnum.pack(side="left")

        speed_row = ctk.CTkFrame(left, fg_color="transparent")
        speed_row.grid(row=2, column=0, sticky="ew", padx=12, pady=(0, 6))
        ctk.CTkLabel(
            speed_row, text="Loop Speed:",
            font=ctk.CTkFont(family=FF, size=12), text_color=TXT_SEC,
        ).pack(side="left", padx=(0, 8))
        self._speed_slider = ctk.CTkSlider(
            speed_row, from_=1, to=10, number_of_steps=9, width=140,
            button_color=BTN_PRI, button_hover_color=BTN_HVP,
            progress_color=BTN_PRI, fg_color=BTN_SEC,
            command=self._speed_changed,
        )
        self._speed_slider.set(3)
        self._speed_slider.pack(side="left")
        self._speed_lbl = ctk.CTkLabel(
            speed_row, text="3 FPS",
            font=ctk.CTkFont(family=FF, size=12), text_color=TXT_SEC,
        )
        self._speed_lbl.pack(side="left", padx=8)

        nav = ctk.CTkFrame(left, fg_color="transparent")
        nav.grid(row=3, column=0, pady=(0, 12))
        ctk.CTkButton(
            nav, text="Play Loop", width=95, height=30, corner_radius=4, border_width=0,
            font=ctk.CTkFont(family=FF, size=12),
            fg_color=BTN_PRI, hover_color=BTN_HVP,
            command=self._start_loop,
        ).pack(side="left", padx=4)
        ctk.CTkButton(
            nav, text="Step Back", width=85, height=30, corner_radius=4, border_width=0,
            font=ctk.CTkFont(family=FF, size=12),
            fg_color=BTN_SEC, hover_color=BTN_HVS, text_color=TXT_PRI,
            command=self._prev_frame,
        ).pack(side="left", padx=4)
        ctk.CTkButton(
            nav, text="Step Forward", width=95, height=30, corner_radius=4, border_width=0,
            font=ctk.CTkFont(family=FF, size=12),
            fg_color=BTN_SEC, hover_color=BTN_HVS, text_color=TXT_PRI,
            command=self._next_frame,
        ).pack(side="left", padx=4)

    def _build_controls(self, parent) -> None:
        right = ctk.CTkScrollableFrame(
            parent, corner_radius=4, width=340,
            border_width=1, border_color=BORDER,
            fg_color=PANEL,
            scrollbar_button_color=BTN_SEC,
            scrollbar_button_hover_color=BTN_HVS,
        )
        right.grid(row=0, column=1, sticky="nsew", padx=(6, 0))

        pad = {"padx": 16, "pady": 4}

        # ── Finger Touch Flags ──────────────────────────────────────────────
        section_label(right, "Finger Touch Flags").pack(anchor="w", padx=16, pady=(12, 2))
        tg = ctk.CTkFrame(right, fg_color="transparent")
        tg.pack(fill="x", **pad)
        for col, finger in enumerate(FINGERS):
            hex_col = FINGER_COLORS_HEX.get(finger, "#cccccc")
            ctk.CTkCheckBox(
                tg, text=finger, variable=self._touch[finger],
                font=ctk.CTkFont(family=FF, size=13, weight="bold"),
                text_color=hex_col,
                checkmark_color=hex_col,
                fg_color=BTN_PRI, hover_color=BTN_HVP,
                border_color=BORDER,
                bg_color=PANEL,
            ).grid(row=col // 3, column=col % 3, padx=6, pady=4, sticky="w")

        # ── Hand Motion & Environment ───────────────────────────────────────
        section_label(right, "Hand Motion & Environment").pack(anchor="w", padx=16, pady=(12, 2))
        for text, var in [
            ("Right Hand", self._right_hand),
            ("Hand Moving", self._hand_move),
            ("Hand Moving Closer", self._hand_closer),
            ("Hovering (Not Touching)", self._hovering),
            ("Daylight Lighting", self._daylight),
            ("Hand Visible", self._hand_visible),
            ("Out of Sync", self._out_of_sync),
        ]:
            make_switch(right, text, var, bg=PANEL).pack(anchor="w", **pad)

        # ── Point of View ───────────────────────────────────────────────────
        section_label(right, "Point of View (POV)").pack(anchor="w", padx=16, pady=(12, 2))
        pov_row = ctk.CTkFrame(right, fg_color="transparent")
        pov_row.pack(fill="x", **pad)
        for opt in POV_OPTIONS:
            ctk.CTkRadioButton(
                pov_row, text=opt.capitalize(),
                variable=self._pov, value=opt,
                font=ctk.CTkFont(family=FF, size=13),
                text_color=TXT_SEC,
                fg_color=BTN_PRI, hover_color=BTN_HVP,
                border_color=BORDER,
                bg_color=PANEL,
            ).pack(side="left", padx=10)

        # ── Observation Notes ───────────────────────────────────────────────
        section_label(right, "Observation Notes (anyDifference)").pack(anchor="w", padx=16, pady=(12, 2))
        self._diff_combo = ctk.CTkComboBox(
            right, variable=self._any_diff,
            values=self._get_diff_presets(), width=290,
            font=ctk.CTkFont(family=FF, size=12),
            fg_color=BTN_SEC, border_color=BORDER, text_color=TXT_SEC,
            button_color=BTN_SEC, button_hover_color=BTN_HVS,
            dropdown_fg_color=PANEL, dropdown_text_color=TXT_SEC,
            dropdown_hover_color=BTN_GHH,
        )
        self._diff_combo.pack(anchor="w", **pad)

        ctk.CTkFrame(right, height=1, fg_color=BORDER).pack(fill="x", padx=16, pady=10)

        ctk.CTkButton(
            right, text="Reset Window Changes", height=30,
            corner_radius=4, border_width=1, border_color=BORDER,
            fg_color=BTN_GHO, hover_color=BTN_GHH, text_color=TXT_MUT,
            font=ctk.CTkFont(family=FF, size=12),
            command=self._reset_current_window_edits,
        ).pack(fill="x", **pad)

        ctk.CTkFrame(right, height=1, fg_color=BORDER).pack(fill="x", padx=16, pady=8)

        # Auto-Save toggle — amber coloured track when ON
        ctk.CTkSwitch(
            right,
            text="Auto-Save on Navigation",
            variable=self._auto_save,
            onvalue=True,
            offvalue=False,
            font=ctk.CTkFont(family=FF, size=12),
            text_color=AMBER,
            fg_color=SW_TRACK,
            progress_color=AMBER,
            button_color="#e0e0e0",
            button_hover_color="#ffffff",
            bg_color=PANEL,
        ).pack(anchor="w", **pad)

        nav_row = ctk.CTkFrame(right, fg_color="transparent")
        nav_row.pack(fill="x", **pad)
        self._btn_prev = ctk.CTkButton(
            nav_row, text="← Previous", width=130, height=36,
            corner_radius=4, border_width=0,
            font=ctk.CTkFont(family=FF, size=12),
            fg_color=BTN_SEC, hover_color=BTN_HVS, text_color=TXT_PRI,
            command=self._go_prev,
        )
        self._btn_prev.pack(side="left", padx=4)
        self._btn_next = ctk.CTkButton(
            nav_row, text="Next →", width=130, height=36,
            corner_radius=4, border_width=0,
            font=ctk.CTkFont(family=FF, size=12, weight="bold"),
            fg_color=BTN_PRI, hover_color=BTN_HVP,
            command=self._go_next,
        )
        self._btn_next.pack(side="left", padx=4)

        ctk.CTkButton(
            right, text="Return to Setup",
            width=160, height=28, corner_radius=4,
            border_width=1, border_color=BORDER,
            fg_color=BTN_GHO, hover_color=BTN_GHH, text_color=TXT_MUT,
            font=ctk.CTkFont(family=FF, size=11),
            command=self._exit,
        ).pack(pady=(12, 8))

    # ── Window loading & navigation ───────────────────────────────────────────

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

        sf  = wf[0]["frame_idx"];  ef  = wf[-1]["frame_idx"]
        sms = wf[0]["timestamp_ms"]; ems = wf[-1]["timestamp_ms"]
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
            self._lbl_recorded.configure(text=f"Already Recorded (Record #{rec_idx + 1})")
            self._populate(existing)
        else:
            logger.info(
                f"No existing record found for frames {sf}–{ef}. "
                "Carrying forward motion/env parameters from previous window, clearing finger touches."
            )
            self._lbl_recorded.configure(text="")
            self._prepare_new_annotation(carry_context=True)

        self._btn_prev.configure(state="normal" if widx > 0 else "disabled")
        self._update_diff_combo()
        self._initial_rec = self._annotation_dict()
        self._start_loop()

    def _reset_current_window_edits(self) -> None:
        logger.info(f"User clicked 'Reset Window Changes' for window index {self._window_idx}")
        if not self._current_wf:
            return
        sf  = self._current_wf[0]["frame_idx"]; ef  = self._current_wf[-1]["frame_idx"]
        sms = self._current_wf[0]["timestamp_ms"]; ems = self._current_wf[-1]["timestamp_ms"]
        rec_idx, existing = self.csv.find(sf, ef, sms, ems)
        if existing:
            logger.info(f"Reverting controls for frames {sf}–{ef} to CSV record index #{rec_idx + 1}")
            self._populate(existing)
        else:
            logger.info(f"No CSV record for frames {sf}–{ef}. Resetting controls to defaults.")
            self._prepare_new_annotation(carry_context=False)

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
            self._load_window(dialog.selected_window_idx)

    def _open_analyzer(self) -> None:
        logger.info(f"User opened Window Analyzer for window index #{self._window_idx + 1}")
        if not self._current_wf:
            return
        from annotator.ui.window_analyzer import WindowAnalyzerDialog
        WindowAnalyzerDialog(self, window_idx=self._window_idx,
                             window_frames=self._current_wf, duration_ms=self.duration_ms)

    # ── Annotation helpers ────────────────────────────────────────────────────

    def _get_diff_presets(self) -> list[str]:
        """Combine default ANY_DIFF_PRESETS with all unique any_difference notes from CSV."""
        presets = list(ANY_DIFF_PRESETS)
        try:
            records = self.csv.read_all()
            for r in records:
                val = str(r.get("any_difference", "")).strip()
                if val and val not in presets:
                    presets.append(val)
        except Exception as exc:
            logger.warning(f"Could not load custom any_difference entries from CSV: {exc}")
        return presets

    def _update_diff_combo(self) -> None:
        """Update dropdown values of _diff_combo from CSV + default presets."""
        if hasattr(self, "_diff_combo"):
            self._diff_combo.configure(values=self._get_diff_presets())

    def _populate(self, r: dict) -> None:
        for fn in FINGERS:
            self._touch[fn].set(str(r.get(f"{fn.lower()}_touch", "False")).lower() == "true")
        self._right_hand.set(str(r.get("rightHand", "False")).lower() == "true")
        self._hand_move.set(str(r.get("hand_move", "False")).lower() == "true")
        self._hand_closer.set(str(r.get("hand_closer", "False")).lower() == "true")
        self._hovering.set(str(r.get("hovering", "False")).lower() == "true")
        self._daylight.set(str(r.get("daylight", "True")).lower() == "true")
        self._hand_visible.set(str(r.get("hand_visible", "True")).lower() == "true")
        self._out_of_sync.set(str(r.get("out_of_sync", "False")).lower() == "true")
        self._pov.set(r.get("hand_point_of_view", "front"))
        self._any_diff.set(r.get("any_difference", ""))

    def _prepare_new_annotation(self, carry_context: bool = True) -> None:
        for fn in FINGERS:
            self._touch[fn].set(False)
        self._any_diff.set("")
        if not carry_context:
            self._right_hand.set(False)
            self._hand_move.set(False); self._hand_closer.set(False)
            self._hovering.set(False); self._daylight.set(True)
            self._hand_visible.set(True); self._out_of_sync.set(False)
            self._pov.set("front")

    def _annotation_dict(self) -> dict:
        ann: dict = {f"{fn.lower()}_touch": self._touch[fn].get() for fn in FINGERS}
        ann.update({
            "rightHand": self._right_hand.get(),
            "hand_move": self._hand_move.get(),
            "hand_point_of_view": self._pov.get(),
            "hand_closer": self._hand_closer.get(),
            "hovering": self._hovering.get(),
            "daylight": self._daylight.get(),
            "hand_visible": self._hand_visible.get(),
            "out_of_sync": self._out_of_sync.get(),
            "any_difference": self._any_diff.get().strip(),
        })
        return ann

    def _records_same(self, existing: dict, new_rec: dict) -> bool:
        ann_fields = (
            [f"{fn.lower()}_touch" for fn in FINGERS]
            + ["rightHand", "hand_move", "hand_point_of_view", "hand_closer",
               "hovering", "daylight", "hand_visible", "out_of_sync", "any_difference"]
        )
        for f in ann_fields:
            if str(existing.get(f, "")).strip().lower() != str(new_rec.get(f, "")).strip().lower():
                return False
        return True

    # ── Visual Debug Overlay ──────────────────────────────────────────────────

    def _on_toggle_debug_overlay(self) -> None:
        curr_idx = self._individual_idx if self._individual_mode else (self._loop_idx - 1) % max(1, len(self._cached_pil))
        self._show(max(0, curr_idx))

    def _draw_debug_overlay(self, pil: Image.Image, fd: dict, local_idx: int) -> Image.Image:
        img = pil.copy().convert("RGBA")
        hd = fd.get("hand_data")
        # Frame 0 of ANY window: its velocity_data was computed from the frame BEFORE this window
        # (cross-window boundary transition) and is intentionally excluded from the CSV.
        # Suppress it here so HUD always matches exactly what gets saved.
        vd = None if local_idx == 0 else fd.get("velocity_data")
        is_first_frame = (local_idx == 0)
        overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        if not hd:
            draw.rectangle([10, 10, 280, 42], fill=(0, 0, 0, 180), outline=(255, 60, 60, 255))
            draw.text((16, 16), "No Hand Landmark Data in Frame", fill=(255, 120, 120))
            return Image.alpha_composite(img, overlay).convert("RGB")

        vel_label = "N/A (frame 1 – no intra-window velocity)" if is_first_frame else ""
        lines = [f"=== Joint Landmarks & Velocity HUD === {vel_label}"]
        wx, wy = hd["wrist"]
        wv = (vd.get("wrist_velocity") or (0.0, 0.0)) if vd else (0.0, 0.0)
        vel_str = "N/A" if is_first_frame else f"({wv[0]:.2f}, {wv[1]:.2f})"
        lines.append(f"Wrist   : Pos({wx:.2f}, {wy:.2f}) | Vel {vel_str}")
        for fn in FINGERS:
            pts = hd["fingers"].get(fn, [])
            fvels = (vd.get("finger_velocities", {}).get(fn, []) if vd else []) or []
            for j, jlabel in enumerate(JOINT_LABELS):
                pt = pts[j] if j < len(pts) else (0.0, 0.0)
                if is_first_frame:
                    jv_str = "N/A"
                else:
                    jv = fvels[j] if (j < len(fvels) and fvels[j] is not None) else (0.0, 0.0)
                    jv_str = f"({jv[0]:.2f}, {jv[1]:.2f})"
                lines.append(f"{fn:<5} {jlabel}: Pos({pt[0]:.2f}, {pt[1]:.2f}) | Vel {jv_str}")

        box_w = 390
        box_h = 16 + len(lines) * 15
        # Highlight frame-0 warning with orange border
        border_col = (255, 165, 0, 255) if is_first_frame else (0, 122, 204, 255)
        draw.rectangle([8, 8, 8 + box_w, 8 + box_h], fill=(30, 30, 30, 220), outline=border_col, width=2)
        y_off = 12
        for i, line in enumerate(lines):
            col = (255, 165, 0) if (i == 0 and is_first_frame) else ((78, 201, 240) if i == 0 else (204, 204, 204))
            draw.text((16, y_off), line, fill=col)
            y_off += 15

        wp = hd.get("wrist_pixel")
        if wp:
            px, py = int(wp[0]), int(wp[1])
            v_inline = "v:N/A" if is_first_frame else f"v:({wv[0]:.2f},{wv[1]:.2f})"
            draw.text((px + 6, py + 4), f"W:({wx:.2f},{wy:.2f})\n{v_inline}", fill=(255, 255, 0, 240))
        for fn, coords in hd.get("fingers_pixel", {}).items():
            pts_norm = hd["fingers"].get(fn, [])
            fvels = (vd.get("finger_velocities", {}).get(fn, []) if vd else []) or []
            for j, pt in enumerate(coords):
                px, py = int(pt[0]), int(pt[1])
                nx, ny = pts_norm[j] if j < len(pts_norm) else (0.0, 0.0)
                jlabel = JOINT_LABELS[j] if j < len(JOINT_LABELS) else f"J{j}"
                if is_first_frame:
                    joint_str = f"{fn[:1]}{jlabel[:1]}:({nx:.2f},{ny:.2f})\nv:N/A"
                else:
                    jv = fvels[j] if (j < len(fvels) and fvels[j] is not None) else (0.0, 0.0)
                    joint_str = f"{fn[:1]}{jlabel[:1]}:({nx:.2f},{ny:.2f})\nv:({jv[0]:.2f},{jv[1]:.2f})"
                draw.text((px + 5, py - 6), joint_str, fill=(0, 230, 255, 240))
        return Image.alpha_composite(img, overlay).convert("RGB")

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
        fd  = self._current_wf[local_idx]

        if self._show_coords_vel.get():
            pil = self._draw_debug_overlay(pil, fd, local_idx)

        try:
            avail_w = max(300, self.winfo_width() * 60 // 100 - 30)
            avail_h = max(240, self.winfo_height() - 200)
        except Exception:
            avail_w, avail_h = _DISPLAY_W, 460

        scaled = pil.copy()
        scaled.thumbnail((avail_w, avail_h), Image.LANCZOS)
        ctk_img = ctk.CTkImage(light_image=scaled, dark_image=scaled, size=(scaled.width, scaled.height))
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
        self._speed_lbl.configure(text=f"{int(float(val))} FPS")

    def _prev_frame(self) -> None:
        self._stop_loop(); self._individual_mode = True
        self._individual_idx = max(0, self._individual_idx - 1)
        self._show(self._individual_idx)

    def _next_frame(self) -> None:
        self._stop_loop(); self._individual_mode = True
        n = len(self._cached_pil) - 1
        self._individual_idx = min(n, self._individual_idx + 1)
        self._show(self._individual_idx)

    # ── Save & navigation ─────────────────────────────────────────────────────

    def _build_record(self) -> dict:
        wf = self._current_wf
        base = extract_window_record_data(wf, os.path.basename(self.video_path),
                                          self.video_hash, self.duration_ms)
        ann = self._annotation_dict()
        rec = {**base, **ann}
        for h in CSV_HEADERS:
            rec.setdefault(h, "")
        return rec

    def _save_current(self) -> bool:
        wf = self._current_wf
        if not wf:
            return True
        sf  = wf[0]["frame_idx"]; ef  = wf[-1]["frame_idx"]
        sms = wf[0]["timestamp_ms"]; ems = wf[-1]["timestamp_ms"]
        new_rec = self._build_record()
        rec_idx, existing = self.csv.find(sf, ef, sms, ems)

        if self._auto_save.get():
            if existing is None:
                logger.info(f"Auto-Save ON: Appending new record for frames {sf}–{ef}")
                self.csv.append(new_rec)
            else:
                if not self._records_same(existing, new_rec):
                    logger.info(f"Auto-Save ON: Overriding existing record for frames {sf}–{ef}")
                    self.csv.override(sf, ef, sms, ems, new_rec)
                else:
                    logger.info(f"Auto-Save ON: Record for frames {sf}–{ef} is identical. Skipping rewrite.")
            self._update_diff_combo()
            return True

        if existing is None:
            # NEW WINDOW (unrecorded in CSV): Save record to CSV silently on navigation (no popup)
            logger.info(f"Auto-Save OFF: New window frames {sf}–{ef}. Appending record to CSV silently.")
            self.csv.append(new_rec)
            self._update_diff_combo()
            return True

        # PREVIOUSLY ANNOTATED WINDOW (already exists in CSV):
        if self._records_same(existing, new_rec):
            logger.info(f"Auto-Save OFF: Previously annotated record for frames {sf}–{ef} is unchanged. Navigating silently.")
            return True

        # Prompt Yes/No/Cancel ONLY when user modifies a previously annotated window
        logger.info(f"Auto-Save OFF: Previously annotated record for frames {sf}–{ef} was modified. Prompting Yes/No/Cancel...")
        choice = messagebox.askyesnocancel(
            "Unsaved Modifications on Recorded Window",
            f"You modified previously annotated window (frames {sf}–{ef}).\n\n"
            "• [Yes]  — SAVE / OVERRIDE changes & navigate\n"
            "• [No]   — DISCARD changes & navigate\n"
            "• [Cancel] — STAY on this window",
            parent=self,
        )
        if choice is True:
            self.csv.override(sf, ef, sms, ems, new_rec)
            self._update_diff_combo()
            return True
        elif choice is False:
            return True
        else:
            return False

    def _go_next(self) -> None:
        logger.info("User clicked 'Next Window'")
        if not self._save_current():
            return
        self._stop_loop()
        next_widx = self._window_idx + 1
        if get_window_frames(self.frame_data, next_widx) is None:
            self._video_end()
        else:
            self._load_window(next_widx)

    def _go_prev(self) -> None:
        logger.info("User clicked 'Previous Window'")
        if self._window_idx > 0:
            if not self._save_current():
                return
            self._stop_loop()
            self._load_window(self._window_idx - 1)

    def _video_end(self) -> None:
        messagebox.showinfo(
            "🎉 Annotation Complete",
            f"All {self.total_windows} video windows have been annotated successfully!\n\n"
            f"Total records saved in CSV: {self.csv.total_records()}\n"
            f"CSV file path: {self.csv_path}\n\n"
            "You can review or edit any window using 'Select Window...' or 'Previous Window'.\n"
            "Click 'Return to Setup' when you are ready to exit.",
            parent=self,
        )

    def _exit(self) -> None:
        if messagebox.askyesno(
            "Exit Annotator",
            "Return to the setup screen?\nUnsaved changes on the current window will be lost.",
            parent=self,
        ):
            self._stop_loop()
            self.app.show_setup()

    def destroy(self) -> None:
        self._stop_loop()
        super().destroy()
