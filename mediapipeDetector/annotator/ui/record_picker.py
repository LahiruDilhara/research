"""
annotator/ui/record_picker.py

Modal Toplevel dialog for picking any video window / record — VS Code Dark+ theme.
Highlights currently active window, shows recorded vs unrecorded status,
and allows jumping to any specific window.
Mousewheel scroll bound so cursor hovering scrolls the list.
"""
import customtkinter as ctk
from annotator.video_processor import get_window_frames
from annotator.ui.theme import (
    FF, BG, PANEL, BORDER, TXT_PRI, TXT_SEC, TXT_MUT,
    ROW_BG, ROW_REC, ROW_CUR, ROW_CUR_T,
    BTN_PRI, BTN_HVP, BTN_SEC, BTN_HVS, BTN_GHO, BTN_GHH, SEL_BG,
)


class RecordPickerDialog(ctk.CTkToplevel):
    """Grab-set modal window picker dialog — VS Code Dark+ styled."""

    def __init__(
        self,
        parent,
        records: list[dict] = None,
        total_windows: int = None,
        current_window_idx: int = None,
        frame_data: list = None,
        csv_manager=None,
    ) -> None:
        super().__init__(parent)
        self.title("Select Target Window / Record")
        self.geometry("860x600")
        self.resizable(True, True)
        self.configure(fg_color=BG)
        self.grab_set()

        self.selected_window_idx: int | None = None
        self.selected_record: dict | None = None

        self._records = records
        self._total_windows = total_windows
        self._current_window_idx = current_window_idx
        self._frame_data = frame_data
        self._csv = csv_manager

        self._selected_row_idx: int | None = None
        self._row_items: list[dict] = []
        self._row_btns: list[ctk.CTkButton] = []
        self._ok_btn: ctk.CTkButton | None = None

        self._build()

    def _build(self) -> None:
        ctk.CTkLabel(
            self,
            text="Jump to Specific Video Window",
            font=ctk.CTkFont(family=FF, size=18, weight="bold"),
            text_color=TXT_PRI,
        ).pack(anchor="w", padx=20, pady=(16, 2))

        ctk.CTkLabel(
            self,
            text="Select any frame window to view or annotate. The currently active window is highlighted.",
            font=ctk.CTkFont(family=FF, size=12),
            text_color=TXT_MUT,
        ).pack(anchor="w", padx=20, pady=(0, 12))

        scroll = ctk.CTkScrollableFrame(
            self,
            corner_radius=4,
            border_width=1,
            border_color=BORDER,
            fg_color=PANEL,
            scrollbar_button_color=BTN_SEC,
            scrollbar_button_hover_color=BTN_HVS,
        )
        scroll.pack(fill="both", expand=True, padx=16, pady=(0, 12))

        # Footer actions (built before items so _select_row can update _ok_btn)
        footer = ctk.CTkFrame(self, fg_color="transparent")
        footer.pack(pady=(0, 16))

        self._ok_btn = ctk.CTkButton(
            footer, text="Jump to Selected Window",
            width=200, height=36, corner_radius=4, border_width=0,
            font=ctk.CTkFont(family=FF, size=13, weight="bold"),
            fg_color=BTN_PRI, hover_color=BTN_HVP, text_color="#ffffff",
            state="disabled",
            command=self._ok,
        )
        self._ok_btn.pack(side="left", padx=6)

        ctk.CTkButton(
            footer, text="Cancel",
            width=100, height=36, corner_radius=4, border_width=1, border_color=BORDER,
            font=ctk.CTkFont(family=FF, size=13),
            fg_color=BTN_GHO, hover_color=BTN_GHH, text_color=TXT_SEC,
            command=self.destroy,
        ).pack(side="left", padx=6)

        # Build list items
        if self._total_windows is not None and self._frame_data is not None:
            self._build_from_windows(scroll)
        elif self._records is not None:
            self._build_from_records(scroll)

    def _build_from_windows(self, parent_scroll) -> None:
        """Render all N sliding windows for the video."""
        for widx in range(self._total_windows):
            wf = get_window_frames(self._frame_data, widx)
            if not wf:
                continue

            sf, ef = wf[0]["frame_idx"], wf[-1]["frame_idx"]
            sms, ems = wf[0]["timestamp_ms"], wf[-1]["timestamp_ms"]
            is_current = (widx == self._current_window_idx)

            rec_idx, existing = (None, None)
            if self._csv:
                rec_idx, existing = self._csv.find(sf, ef, sms, ems)

            tag = ""
            if is_current:
                tag += "[CURRENT]  "
            if existing:
                touches = [
                    f.capitalize()
                    for f in ["thumb", "index", "middle", "ring", "pinky"]
                    if str(existing.get(f"{f}_touch", "False")).lower() == "true"
                ]
                hand_str = "R-Hand" if str(existing.get("rightHand", "False")).lower() == "true" else "L-Hand"
                tag += f"Saved #{rec_idx+1}  ·  Hand: {hand_str}  ·  Touch: {t_str}  ·  POV: {existing.get('hand_point_of_view','front')}"
            else:
                tag += "Unrecorded"

            label_text = (
                f"  Window #{widx+1:>3}   "
                f"Frames {sf}–{ef}   "
                f"({sms}–{ems} ms)   "
                f"{tag}"
            )

            if is_current:
                bg_col  = ROW_CUR
                txt_col = ROW_CUR_T
                hov_col = "#1a4d72"
            elif existing:
                bg_col  = ROW_REC
                txt_col = TXT_SEC
                hov_col = "#37373d"
            else:
                bg_col  = ROW_BG
                txt_col = TXT_MUT
                hov_col = "#37373d"

            item_idx = len(self._row_items)
            btn = ctk.CTkButton(
                parent_scroll,
                text=label_text,
                anchor="w",
                height=34,
                corner_radius=4,
                border_width=0,
                font=ctk.CTkFont(family=FF, size=12, weight="bold" if is_current else "normal"),
                fg_color=bg_col,
                hover_color=hov_col,
                text_color=txt_col,
                command=lambda row_i=item_idx: self._select_row(row_i),
            )
            btn.pack(fill="x", pady=1, padx=4)

            self._row_items.append({
                "widx": widx,
                "record": existing,
                "is_current": is_current,
                "default_bg": bg_col,
                "default_text": txt_col,
                "default_hover": hov_col,
            })
            self._row_btns.append(btn)

        # Pre-select current active window row
        if self._current_window_idx is not None and 0 <= self._current_window_idx < len(self._row_items):
            self._select_row(self._current_window_idx)

    def _build_from_records(self, parent_scroll) -> None:
        """Render from CSV records list (used when opened from RecoveryScreen)."""
        from annotator.video_processor import window_idx_from_start_frame

        for i, r in enumerate(self._records):
            widx = window_idx_from_start_frame(int(r.get("start_frame", 0)))
            touches = [
                f.capitalize()
                for f in ["thumb", "index", "middle", "ring", "pinky"]
                if str(r.get(f"{f}_touch", "False")).lower() == "true"
            ]
            t_str = " + ".join(touches) or "None"
            label_text = (
                f"  Record #{i+1:>3}   "
                f"Frames {r.get('start_frame')}–{r.get('end_frame')}   "
                f"({r.get('start_ms')}–{r.get('end_ms')} ms)   "
                f"Touch: {t_str}   "
                f"POV: {r.get('hand_point_of_view','?')}"
            )
            item_idx = len(self._row_items)
            btn = ctk.CTkButton(
                parent_scroll,
                text=label_text,
                anchor="w",
                height=34,
                corner_radius=4,
                border_width=0,
                font=ctk.CTkFont(family=FF, size=12),
                fg_color=ROW_REC,
                hover_color="#37373d",
                text_color=TXT_SEC,
                command=lambda row_i=item_idx: self._select_row(row_i),
            )
            btn.pack(fill="x", pady=1, padx=4)

            self._row_items.append({
                "widx": widx,
                "record": r,
                "is_current": False,
                "default_bg": ROW_REC,
                "default_text": TXT_SEC,
                "default_hover": "#37373d",
            })
            self._row_btns.append(btn)

    def _select_row(self, idx: int) -> None:
        """Highlight selected row and enable OK button."""
        if not (0 <= idx < len(self._row_items)):
            return

        self._selected_row_idx = idx

        for i, (item, b) in enumerate(zip(self._row_items, self._row_btns)):
            if i == idx:
                b.configure(fg_color=SEL_BG, text_color="#ffffff")
            else:
                b.configure(fg_color=item["default_bg"], text_color=item["default_text"])

        item = self._row_items[idx]
        self.selected_window_idx = item["widx"]
        self.selected_record = item["record"]
        if self._ok_btn:
            self._ok_btn.configure(state="normal")

    def _ok(self) -> None:
        self.destroy()
