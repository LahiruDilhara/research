"""
annotator/ui/recovery_screen.py

Shown when re-opening an existing CSV.
Displays a clean, modern summary of saved records and lets the user choose
where to resume: next unsaved window, last record, or a specific record picker.
"""
import customtkinter as ctk

from annotator.constants import WINDOW_STEP
from annotator.csv_manager import CSVManager
from annotator.video_processor import window_count, window_idx_from_start_frame


class RecoveryScreen(ctk.CTkFrame):
    def __init__(
        self, parent, app,
        frame_data, fps, total_frames, duration_ms,
        csv_path, video_path, video_hash,
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
        self._build()

    def _build(self) -> None:
        records = self.csv.read_all()
        total_wins = window_count(self.total_frames)
        last = records[-1] if records else None

        # ── Header ───────────────────────────────────────────────────────────
        hdr = ctk.CTkFrame(self, fg_color=("gray90", "gray16"), corner_radius=0, height=56)
        hdr.pack(fill="x")
        hdr.pack_propagate(False)

        ctk.CTkLabel(
            hdr, text="Resume Session",
            font=ctk.CTkFont(size=18, weight="bold"),
        ).pack(side="left", padx=20, pady=12)

        ctk.CTkLabel(
            hdr, text=self.csv_path,
            font=ctk.CTkFont(size=11), text_color=("gray50", "gray60"),
        ).pack(side="left", padx=8)

        scroll_outer = ctk.CTkScrollableFrame(self)
        scroll_outer.pack(fill="both", expand=True, padx=20, pady=12)

        # ── Summary Card ─────────────────────────────────────────────────────
        card = ctk.CTkFrame(
            scroll_outer,
            corner_radius=8,
            border_width=1,
            border_color=("gray80", "gray25"),
            fg_color=("gray95", "gray14"),
        )
        card.pack(fill="x", pady=(0, 12))

        ctk.CTkLabel(
            card,
            text=f"Dataset Summary: {len(records)} / {total_wins} Windows Annotated",
            font=ctk.CTkFont(size=15, weight="bold"),
        ).pack(anchor="w", padx=20, pady=(16, 4))

        if last:
            sf = last.get("start_frame", "?")
            ef = last.get("end_frame", "?")
            sms = last.get("start_ms", "?")
            ems = last.get("end_ms", "?")
            ctk.CTkLabel(
                card,
                text=f"Last Recorded Window: Frames {sf} – {ef}  ({sms} ms – {ems} ms)",
                font=ctk.CTkFont(size=13),
            ).pack(anchor="w", padx=20, pady=2)

            touches = [
                f.capitalize() for f in ["thumb", "index", "middle", "ring", "pinky"]
                if str(last.get(f"{f}_touch", "False")).lower() == "true"
            ]
            ctk.CTkLabel(
                card,
                text=(
                    f"Touch Flags: {', '.join(touches) or 'None'}"
                    f"  ·  POV: {last.get('hand_point_of_view', '?')}"
                    f"  ·  Motion: {last.get('hand_move', '?')}"
                    f"  ·  Visible: {last.get('hand_visible', '?')}"
                ),
                font=ctk.CTkFont(size=12), text_color=("gray40", "gray60"),
            ).pack(anchor="w", padx=20, pady=(0, 16))
        else:
            ctk.CTkLabel(
                card,
                text="No records saved yet in this CSV file.",
                font=ctk.CTkFont(size=13), text_color=("gray40", "gray60"),
            ).pack(anchor="w", padx=20, pady=16)

        # ── Records List ─────────────────────────────────────────────────────
        ctk.CTkLabel(
            scroll_outer,
            text="Recorded Frames Log",
            font=ctk.CTkFont(size=13, weight="bold"),
        ).pack(anchor="w", pady=(4, 6))

        for i, r in enumerate(records):
            touches = [
                f[:2].upper() for f in ["thumb", "index", "middle", "ring", "pinky"]
                if str(r.get(f"{f}_touch", "False")).lower() == "true"
            ]
            touch_str = "+".join(touches) or "None"
            row = ctk.CTkFrame(scroll_outer, fg_color=("gray90", "gray20"), corner_radius=4)
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
                ),
                font=ctk.CTkFont(size=11),
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
                height=38,
                corner_radius=6,
                font=ctk.CTkFont(size=13, weight="bold"),
                fg_color="#2563eb", hover_color="#1d4ed8",
                command=lambda: self._go(next_widx, allow_override=False),
            ).pack(side="left", padx=6)

            ctk.CTkButton(
                btn_row,
                text="Re-annotate Last Record",
                height=38,
                corner_radius=6,
                font=ctk.CTkFont(size=13),
                fg_color=("gray80", "gray28"), hover_color=("gray70", "gray36"),
                text_color=("gray10", "gray95"),
                command=lambda: self._go(last_widx, allow_override=True),
            ).pack(side="left", padx=6)
        else:
            ctk.CTkButton(
                btn_row,
                text="Start from Beginning",
                height=38,
                corner_radius=6,
                font=ctk.CTkFont(size=13, weight="bold"),
                fg_color="#2563eb", hover_color="#1d4ed8",
                command=lambda: self._go(0, allow_override=False),
            ).pack(side="left", padx=6)

        ctk.CTkButton(
            btn_row,
            text="Select Specific Record...",
            height=38,
            corner_radius=6,
            font=ctk.CTkFont(size=13),
            fg_color=("gray80", "gray28"), hover_color=("gray70", "gray36"),
            text_color=("gray10", "gray95"),
            command=self._pick,
        ).pack(side="left", padx=6)

        ctk.CTkButton(
            btn_row,
            text="Cancel",
            height=38,
            corner_radius=6,
            font=ctk.CTkFont(size=13),
            fg_color="transparent", hover_color=("gray85", "gray22"),
            border_width=1, border_color=("gray70", "gray35"),
            text_color=("gray20", "gray80"),
            command=self.app.show_setup,
        ).pack(side="left", padx=6)

    def _go(self, window_idx: int, allow_override: bool) -> None:
        self.app.show_annotation(
            self.frame_data, self.fps, self.total_frames, self.duration_ms,
            self.csv_path, self.video_path, self.video_hash,
            start_window_idx=window_idx,
            allow_override_last=allow_override,
        )

    def _pick(self) -> None:
        from annotator.ui.record_picker import RecordPickerDialog
        dialog = RecordPickerDialog(self, self.csv.read_all())
        self.wait_window(dialog)
        if dialog.selected_record:
            widx = window_idx_from_start_frame(
                int(dialog.selected_record.get("start_frame", 0))
            )
            self._go(widx, allow_override=False)
