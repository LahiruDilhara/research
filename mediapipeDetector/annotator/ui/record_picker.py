"""
annotator/ui/record_picker.py

Modal Toplevel dialog that lets the user select a specific CSV record to jump to.
Clean, modern layout matching the application design system.
"""
import customtkinter as ctk


class RecordPickerDialog(ctk.CTkToplevel):
    """
    Grab-set modal dialog.
    After the dialog closes, check `dialog.selected_record` (dict | None).
    """

    def __init__(self, parent, records: list[dict]) -> None:
        super().__init__(parent)
        self.title("Select Target Record")
        self.geometry("740x520")
        self.resizable(True, True)
        self.grab_set()
        self.selected_record: dict | None = None
        self._records = records
        self._selected_idx: int | None = None
        self._build()

    def _build(self) -> None:
        ctk.CTkLabel(
            self,
            text="Select a Record to Jump To",
            font=ctk.CTkFont(size=16, weight="bold"),
        ).pack(anchor="w", padx=20, pady=(16, 4))

        ctk.CTkLabel(
            self,
            text="Choose any recorded frame window to view or re-annotate:",
            font=ctk.CTkFont(size=12),
            text_color=("gray40", "gray60"),
        ).pack(anchor="w", padx=20, pady=(0, 12))

        scroll = ctk.CTkScrollableFrame(
            self,
            corner_radius=8,
            border_width=1,
            border_color=("gray80", "gray25"),
            fg_color=("gray95", "gray14"),
        )
        scroll.pack(fill="both", expand=True, padx=16, pady=(0, 12))

        self._row_btns: list[ctk.CTkButton] = []

        for i, r in enumerate(self._records):
            touches = [
                f.capitalize()
                for f in ["thumb", "index", "middle", "ring", "pinky"]
                if str(r.get(f"{f}_touch", "False")).lower() == "true"
            ]
            t_str = " + ".join(touches) or "None"
            label = (
                f"#{i+1:>3}   Frames {r.get('start_frame')}–{r.get('end_frame')}"
                f"   ({r.get('start_ms')}–{r.get('end_ms')} ms)"
                f"   Touch: {t_str}"
                f"   POV: {r.get('hand_point_of_view','?')}"
            )
            btn = ctk.CTkButton(
                scroll,
                text=label,
                anchor="w",
                height=32,
                corner_radius=4,
                font=ctk.CTkFont(size=12),
                fg_color=("gray90", "gray20"),
                hover_color=("gray82", "gray28"),
                text_color=("gray10", "gray95"),
                command=lambda idx=i: self._select(idx),
            )
            btn.pack(fill="x", pady=2, padx=4)
            self._row_btns.append(btn)

        footer = ctk.CTkFrame(self, fg_color="transparent")
        footer.pack(pady=(0, 16))

        self._ok_btn = ctk.CTkButton(
            footer, text="Confirm Selection",
            width=160, height=36, corner_radius=6,
            font=ctk.CTkFont(size=13, weight="bold"),
            fg_color="#2563eb", hover_color="#1d4ed8",
            state="disabled",
            command=self._ok,
        )
        self._ok_btn.pack(side="left", padx=6)

        ctk.CTkButton(
            footer, text="Cancel",
            width=100, height=36, corner_radius=6,
            font=ctk.CTkFont(size=13),
            fg_color="transparent", hover_color=("gray85", "gray22"),
            border_width=1, border_color=("gray70", "gray35"),
            text_color=("gray20", "gray80"),
            command=self.destroy,
        ).pack(side="left", padx=6)

    def _select(self, idx: int) -> None:
        for i, b in enumerate(self._row_btns):
            b.configure(
                fg_color=("#2563eb", "#1d4ed8") if i == idx
                else ("gray90", "gray20"),
                text_color=("white", "white") if i == idx
                else ("gray10", "gray95"),
            )
        self._selected_idx = idx
        self.selected_record = self._records[idx]
        self._ok_btn.configure(state="normal")

    def _ok(self) -> None:
        self.destroy()
