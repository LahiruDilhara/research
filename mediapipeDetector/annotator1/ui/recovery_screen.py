"""
annotator/ui/recovery_screen.py

Shown when re-opening an existing CSV — VS Code Dark+ theme.
Displays a clean summary of saved records and lets the user choose
where to resume: next unsaved window, last record, or a specific record picker.
Mousewheel scroll is bound so cursor hovering scrolls the list.
"""
import customtkinter as ctk

from annotator.constants import WINDOW_STEP
from annotator.csv_manager import CSVManager
from annotator.video_processor import window_count, window_idx_from_start_frame
from annotator.ui.theme import (
    FF, BG, PANEL, BORDER, HDR_BG, TXT_PRI, TXT_SEC, TXT_MUT,
    ROW_BG, BTN_PRI, BTN_HVP, BTN_SEC, BTN_HVS, BTN_GHO, BTN_GHH, AMBER,
    make_switch, section_label,
)


class RecoveryScreen(ctk.CTkFrame):
    def __init__(
        self, parent, app,
        frame_data, fps, total_frames, duration_ms,
        csv_path, video_path, video_hash,
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
        self._build()

    def _build(self) -> None:
        records = self.csv.read_all()
        total_wins = window_count(self.total_frames)
        last = records[-1] if records else None

        # ── Header bar ───────────────────────────────────────────────────────
        hdr = ctk.CTkFrame(self, fg_color=HDR_BG, corner_radius=0, height=52)
        hdr.pack(fill="x")
        hdr.pack_propagate(False)

        ctk.CTkLabel(
            hdr, text="Resume Session",
            font=ctk.CTkFont(family=FF, size=17, weight="bold"),
            text_color=TXT_PRI,
        ).pack(side="left", padx=20, pady=12)

        ctk.CTkLabel(
            hdr, text=self.csv_path,
            font=ctk.CTkFont(family=FF, size=11),
            text_color=TXT_MUT,
        ).pack(side="left", padx=8)

        # ── Scrollable body ──────────────────────────────────────────────────
        scroll_outer = ctk.CTkScrollableFrame(
            self, fg_color=BG, scrollbar_button_color=BTN_SEC,
            scrollbar_button_hover_color=BTN_HVS,
        )
        scroll_outer.pack(fill="both", expand=True, padx=20, pady=12)

        # ── Summary Card ─────────────────────────────────────────────────────
        card = ctk.CTkFrame(
            scroll_outer,
            corner_radius=4,
            border_width=1,
            border_color=BORDER,
            fg_color=PANEL,
        )
        card.pack(fill="x", pady=(0, 14))

        ctk.CTkLabel(
            card,
            text=f"Dataset Summary: {len(records)} / {total_wins} Windows Annotated",
            font=ctk.CTkFont(family=FF, size=15, weight="bold"),
            text_color=TXT_PRI,
        ).pack(anchor="w", padx=20, pady=(16, 4))

        if last:
            sf = last.get("start_frame", "?")
            ef = last.get("end_frame", "?")
            sms = last.get("start_ms", "?")
            ems = last.get("end_ms", "?")
            ctk.CTkLabel(
                card,
                text=f"Last Recorded Window: Frames {sf} – {ef}  ({sms} ms – {ems} ms)",
                font=ctk.CTkFont(family=FF, size=13),
                text_color=TXT_SEC,
            ).pack(anchor="w", padx=20, pady=2)

            touches = [
                f.capitalize() for f in ["thumb", "index", "middle", "ring", "pinky"]
                if str(last.get(f"{f}_touch", "False")).lower() == "true"
            ]
            oos_str = "  ·  Out of Sync" if str(last.get("out_of_sync", "False")).lower() == "true" else ""
            ctk.CTkLabel(
                card,
                text=(
                    f"Touch Flags: {', '.join(touches) or 'None'}"
                    f"  ·  POV: {last.get('hand_point_of_view', '?')}"
                    f"  ·  Motion: {last.get('hand_move', '?')}"
                    f"  ·  Visible: {last.get('hand_visible', '?')}"
                    f"{oos_str}"
                ),
                font=ctk.CTkFont(family=FF, size=12),
                text_color=TXT_MUT,
            ).pack(anchor="w", padx=20, pady=(0, 16))
        else:
            ctk.CTkLabel(
                card,
                text="No records saved yet in this CSV file.",
                font=ctk.CTkFont(family=FF, size=13),
                text_color=TXT_MUT,
            ).pack(anchor="w", padx=20, pady=16)

        # ── Records List ─────────────────────────────────────────────────────
        ctk.CTkLabel(
            scroll_outer,
            text="Recorded Frames Log",
            font=ctk.CTkFont(family=FF, size=13, weight="bold"),
            text_color=TXT_PRI,
        ).pack(anchor="w", pady=(4, 6))

        for i, r in enumerate(records):
            touches = [
                f[:2].upper() for f in ["thumb", "index", "middle", "ring", "pinky"]
                if str(r.get(f"{f}_touch", "False")).lower() == "true"
            ]
            touch_str = "+".join(touches) or "None"
            oos_flag = "  Out of Sync" if str(r.get("out_of_sync", "False")).lower() == "true" else ""
            row = ctk.CTkFrame(scroll_outer, fg_color=ROW_BG, corner_radius=4)
            row.pack(fill="x", pady=2)
            ctk.CTkLabel(
                row,
                text=(
                    f"#{i+1:>3}   "
                    f"Frames {r.get('start_frame')}–{r.get('end_frame')}   "
                    f"({r.get('start_ms')}–{r.get('end_ms')} ms)   "
                    f"Touch: {touch_str}   "
                    f"POV: {r.get('hand_point_of_view','?')}   "
                    f"Motion: {r.get('hand_move','?')}"
                    f"{oos_flag}"
                ),
                font=ctk.CTkFont(family=FF, size=11),
                text_color=TXT_SEC,
            ).pack(anchor="w", padx=12, pady=6)

        # ── Action Footer ────────────────────────────────────────────────────
        btn_row = ctk.CTkFrame(self, fg_color="transparent")
        btn_row.pack(pady=16)

        if last:
            last_widx = window_idx_from_start_frame(int(last.get("start_frame", 0)))
            next_widx = last_widx + 1

            ctk.CTkButton(
                btn_row,
                text="Continue Next Window",
                height=38, corner_radius=4, border_width=0,
                font=ctk.CTkFont(family=FF, size=13, weight="bold"),
                fg_color=BTN_PRI, hover_color=BTN_HVP, text_color="#ffffff",
                command=lambda: self._go(next_widx),
            ).pack(side="left", padx=6)

            ctk.CTkButton(
                btn_row,
                text="Re-annotate Last Record",
                height=38, corner_radius=4, border_width=0,
                font=ctk.CTkFont(family=FF, size=13),
                fg_color=BTN_SEC, hover_color=BTN_HVS, text_color=TXT_PRI,
                command=lambda: self._go(last_widx),
            ).pack(side="left", padx=6)
        else:
            ctk.CTkButton(
                btn_row,
                text="Start from Beginning",
                height=38, corner_radius=4, border_width=0,
                font=ctk.CTkFont(family=FF, size=13, weight="bold"),
                fg_color=BTN_PRI, hover_color=BTN_HVP, text_color="#ffffff",
                command=lambda: self._go(0),
            ).pack(side="left", padx=6)

        ctk.CTkButton(
            btn_row,
            text="Select Specific Record...",
            height=38, corner_radius=4, border_width=0,
            font=ctk.CTkFont(family=FF, size=13),
            fg_color=BTN_SEC, hover_color=BTN_HVS, text_color=TXT_PRI,
            command=self._pick,
        ).pack(side="left", padx=6)

        ctk.CTkButton(
            btn_row,
            text="Cancel",
            height=38, corner_radius=4, border_width=1, border_color=BORDER,
            font=ctk.CTkFont(family=FF, size=13),
            fg_color=BTN_GHO, hover_color=BTN_GHH, text_color=TXT_SEC,
            command=self.app.show_setup,
        ).pack(side="left", padx=6)

    def _go(self, window_idx: int) -> None:
        self.app.show_annotation(
            self.frame_data, self.fps, self.total_frames, self.duration_ms,
            self.csv_path, self.video_path, self.video_hash,
            start_window_idx=window_idx,
        )

    def _pick(self) -> None:
        from annotator.ui.record_picker import RecordPickerDialog
        total_wins = window_count(self.total_frames)
        dialog = RecordPickerDialog(
            self,
            records=self.csv.read_all(),
            total_windows=total_wins,
            frame_data=self.frame_data,
            csv_manager=self.csv,
        )
        self.wait_window(dialog)
        if dialog.selected_window_idx is not None:
            self._go(dialog.selected_window_idx)
