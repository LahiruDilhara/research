"""
annotator/ui/shortcut_settings.py

VS Code Dark-themed keyboard shortcut configuration dialog.
Click [Set] on any row, then press the key you want to assign.
Escape cancels a capture; Backspace clears a binding.
Settings are persisted via ShortcutManager to shortcuts.json.
"""
import logging
import customtkinter as ctk

from annotator.shortcuts import (
    ShortcutManager, ACTION_LABELS, DEFAULT_SHORTCUTS, keysym_to_display,
)
from annotator.ui.theme import (
    FF, BG, PANEL, BORDER, HDR_BG,
    TXT_PRI, TXT_SEC, TXT_MUT,
    BTN_PRI, BTN_HVP, BTN_SEC, BTN_HVS, BTN_GHO, BTN_GHH,
)

logger = logging.getLogger("Annotator.ShortcutSettings")

# Keys that should never be accepted as a shortcut binding
_MODIFIER_KEYSYMS = {
    "Shift_L", "Shift_R", "Control_L", "Control_R",
    "Alt_L", "Alt_R", "Super_L", "Super_R",
    "Caps_Lock", "Num_Lock", "Scroll_Lock",
    "ISO_Level3_Shift",
}

# Action groups for visual layout
_FINGER_ACTIONS = ["thumb", "index", "middle", "ring", "pinky"]
_NAV_ACTIONS = ["play_window", "next_window", "prev_window"]
_SECTION_LABELS = {
    "fingers": "Finger Touch Toggles",
    "nav":     "Playback & Navigation",
}


class ShortcutSettingsDialog(ctk.CTkToplevel):
    """
    Modal dialog for editing keyboard shortcut bindings.

    Parameters
    ----------
    parent :
        Parent widget (AnnotationScreen frame).
    shortcut_mgr :
        Shared ShortcutManager instance — mutated in-place on save.
    on_updated :
        Optional callback invoked after saving so the caller can
        refresh its internal keysym lookup table.
    """

    def __init__(
        self,
        parent,
        shortcut_mgr: ShortcutManager,
        on_updated=None,
    ) -> None:
        super().__init__(parent)
        self._mgr = shortcut_mgr
        self._on_updated = on_updated
        self._capturing_action: str | None = None
        # Working copy — only committed to the manager on Save
        self._working: dict[str, str] = shortcut_mgr.all()
        # badge StringVars keyed by action name
        self._badges: dict[str, ctk.StringVar] = {}
        # Set buttons keyed by action name (so we can style the active one)
        self._set_btns: dict[str, ctk.CTkButton] = {}

        self.title("Keyboard Shortcuts")
        self.geometry("540x500")
        self.resizable(False, False)
        self.configure(fg_color=BG)
        self.transient(parent)
        self.lift()
        self.focus_force()

        self._save_btn: ctk.CTkButton | None = None  # assigned in _build
        self._saved_lbl: ctk.CTkLabel | None = None  # "✓ Saved" indicator

        self._build()
        # Capture key events at the Toplevel level
        self.bind("<KeyPress>", self._on_key_press)
        # Closing the window with X also saves (same as clicking Save)
        self.protocol("WM_DELETE_WINDOW", self._save)
        logger.info("ShortcutSettingsDialog opened.")

    # ── UI construction ───────────────────────────────────────────────────────

    def _build(self) -> None:
        # ── Header ────────────────────────────────────────────────────────────
        hdr = ctk.CTkFrame(self, height=50, corner_radius=0, fg_color=HDR_BG)
        hdr.pack(fill="x")
        hdr.pack_propagate(False)

        ctk.CTkLabel(
            hdr, text="⌨   Keyboard Shortcuts",
            font=ctk.CTkFont(family=FF, size=15, weight="bold"),
            text_color=TXT_PRI,
        ).pack(side="left", padx=18, pady=12)

        # ── Instruction banner ────────────────────────────────────────────────
        info = ctk.CTkFrame(self, fg_color=PANEL, corner_radius=0)
        info.pack(fill="x")
        ctk.CTkLabel(
            info,
            text="Click  [Set]  next to an action, then press the desired key.  "
                 "Esc cancels.  Backspace clears.",
            font=ctk.CTkFont(family=FF, size=11),
            text_color=TXT_MUT,
        ).pack(padx=18, pady=8, anchor="w")

        sep = ctk.CTkFrame(self, height=1, fg_color=BORDER)
        sep.pack(fill="x")

        # ── Scrollable rows area ───────────────────────────────────────────────
        scroll = ctk.CTkScrollableFrame(
            self,
            fg_color="transparent",
            scrollbar_button_color=BTN_SEC,
            scrollbar_button_hover_color=BTN_HVS,
        )
        scroll.pack(fill="both", expand=True, padx=18, pady=12)

        self._add_section(scroll, _SECTION_LABELS["fingers"], _FINGER_ACTIONS)
        self._add_section(scroll, _SECTION_LABELS["nav"],     _NAV_ACTIONS)

        # ── Bottom button row ─────────────────────────────────────────────────
        btn_row = ctk.CTkFrame(self, height=54, corner_radius=0, fg_color=HDR_BG)
        btn_row.pack(fill="x", side="bottom")
        btn_row.pack_propagate(False)

        ctk.CTkButton(
            btn_row, text="Reset Defaults", width=130, height=32,
            corner_radius=4, border_width=1, border_color=BORDER,
            fg_color=BTN_GHO, hover_color=BTN_GHH,
            text_color=TXT_SEC,
            font=ctk.CTkFont(family=FF, size=12),
            command=self._reset_defaults,
        ).pack(side="left", padx=16, pady=11)

        ctk.CTkButton(
            btn_row, text="Cancel", width=90, height=32,
            corner_radius=4, border_width=1, border_color=BORDER,
            fg_color=BTN_GHO, hover_color=BTN_GHH,
            text_color=TXT_SEC,
            font=ctk.CTkFont(family=FF, size=12),
            command=self.destroy,
        ).pack(side="right", padx=(0, 8), pady=11)

        self._saved_lbl = ctk.CTkLabel(
            btn_row, text="",
            font=ctk.CTkFont(family=FF, size=12, weight="bold"),
            text_color="#4ade80",
        )
        self._saved_lbl.pack(side="right", padx=(0, 10))

        self._save_btn = ctk.CTkButton(
            btn_row, text="Save", width=100, height=32,
            corner_radius=4,
            fg_color=BTN_PRI, hover_color=BTN_HVP,
            text_color="white",
            font=ctk.CTkFont(family=FF, size=13, weight="bold"),
            command=self._save,
        )
        self._save_btn.pack(side="right", padx=(0, 12), pady=11)

    def _add_section(self, parent, title: str, actions: list[str]) -> None:
        ctk.CTkLabel(
            parent, text=title,
            font=ctk.CTkFont(family=FF, size=12, weight="bold"),
            text_color="#4ec9f0",
            anchor="w",
        ).pack(fill="x", pady=(10, 4))

        section_box = ctk.CTkFrame(
            parent, fg_color=PANEL,
            corner_radius=6, border_width=1, border_color=BORDER,
        )
        section_box.pack(fill="x", pady=(0, 8))

        for idx, action in enumerate(actions):
            bg = "#1e1e1e" if idx % 2 == 0 else PANEL
            self._add_row(section_box, action, bg)

    def _add_row(self, parent, action: str, bg: str) -> None:
        label = ACTION_LABELS.get(action, action)
        row = ctk.CTkFrame(parent, fg_color=bg, corner_radius=0)
        row.pack(fill="x")

        ctk.CTkLabel(
            row, text=label, width=220, anchor="w",
            font=ctk.CTkFont(family=FF, size=13),
            text_color=TXT_PRI,
        ).pack(side="left", padx=(14, 0), pady=10)

        # Key badge label (shows current binding)
        badge_var = ctk.StringVar(value=keysym_to_display(self._working.get(action, "")))
        self._badges[action] = badge_var

        badge = ctk.CTkLabel(
            row,
            textvariable=badge_var,
            width=130, height=28,
            corner_radius=5,
            fg_color="#252526",
            text_color="#4ec9f0",
            font=ctk.CTkFont(family=FF, size=13, weight="bold"),
        )
        badge.pack(side="left", padx=14)

        # Set button
        btn = ctk.CTkButton(
            row, text="Set", width=58, height=28, corner_radius=4,
            fg_color=BTN_SEC, hover_color=BTN_HVS,
            text_color=TXT_SEC,
            font=ctk.CTkFont(family=FF, size=12),
            command=lambda a=action: self._start_capture(a),
        )
        btn.pack(side="left")
        self._set_btns[action] = btn

    # ── Key capture logic ─────────────────────────────────────────────────────

    def _start_capture(self, action: str) -> None:
        # Cancel any previous pending capture
        if self._capturing_action and self._capturing_action in self._badges:
            prev = self._capturing_action
            self._badges[prev].set(keysym_to_display(self._working.get(prev, "")))
            self._set_btns[prev].configure(
                fg_color=BTN_SEC, hover_color=BTN_HVS, text="Set",
            )
        self._capturing_action = action
        self._badges[action].set("▸  Press any key…")
        self._set_btns[action].configure(
            fg_color="#8b0000", hover_color="#6b0000", text="Cancel",
        )
        logger.debug(f"Shortcut capture started for action: {action}")

    def _on_key_press(self, event) -> None:
        if not self._capturing_action:
            return
        keysym = event.keysym

        # Ignore bare modifier key presses
        if keysym in _MODIFIER_KEYSYMS:
            return

        action = self._capturing_action
        self._capturing_action = None

        # Restore Set button appearance
        self._set_btns[action].configure(
            fg_color=BTN_SEC, hover_color=BTN_HVS, text="Set",
        )

        if keysym == "Escape":
            # Cancel — restore previous value
            self._badges[action].set(keysym_to_display(self._working.get(action, "")))
            logger.debug(f"Shortcut capture cancelled for action: {action}")
            return

        if keysym == "BackSpace":
            # Clear the binding
            self._working[action] = ""
            self._badges[action].set("—")
            logger.debug(f"Shortcut cleared for action: {action}")
            return

        self._working[action] = keysym
        self._badges[action].set(keysym_to_display(keysym))
        logger.info(f"Shortcut assigned: {action} → {keysym}")

    # ── Bottom button actions ─────────────────────────────────────────────────

    def _reset_defaults(self) -> None:
        # Cancel any active capture first
        if self._capturing_action:
            self._set_btns[self._capturing_action].configure(
                fg_color=BTN_SEC, hover_color=BTN_HVS, text="Set",
            )
            self._capturing_action = None
        self._working = dict(DEFAULT_SHORTCUTS)
        for action, badge_var in self._badges.items():
            badge_var.set(keysym_to_display(self._working.get(action, "")))
        logger.info("Shortcut defaults restored.")

    def _save(self) -> None:
        for action, keysym in self._working.items():
            self._mgr.set(action, keysym)
        self._mgr.save()
        if self._saved_lbl:
            self._saved_lbl.configure(text="✓ Saved")
        if self._on_updated:
            self._on_updated()
        logger.info("Shortcuts saved to JSON file and dialog closed.")
        self.destroy()
