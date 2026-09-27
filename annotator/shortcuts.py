"""
datacreator/annotator/shortcuts.py

Keyboard shortcut manager and editor dialog for the Lightweight Annotator GUI.
Persists keybindings to shortcuts.json so custom shortcuts persist across restarts.
"""

import json
import os
import customtkinter as ctk

SHORTCUT_FILE = os.path.join(os.path.dirname(__file__), "shortcuts.json")

DEFAULT_SHORTCUTS: dict[str, str] = {
    "thumb":         "1",
    "index":         "2",
    "middle":        "3",
    "ring":          "4",
    "pinky":         "5",
    "next_window":   "d",
    "prev_window":   "a",
    "step_forward":  "Right",
    "step_back":     "Left",
    "play_window":   "space",
    "save":          "s",
}

ACTION_LABELS: dict[str, str] = {
    "thumb":         "Toggle Thumb Touch",
    "index":         "Toggle Index Touch",
    "middle":        "Toggle Middle Touch",
    "ring":          "Toggle Ring Touch",
    "pinky":         "Toggle Pinky Touch",
    "next_window":   "Next Window (D)",
    "prev_window":   "Previous Window (A)",
    "step_forward":  "Step Forward inside Window (Right Arrow / .)",
    "step_back":     "Step Backward inside Window (Left Arrow / ,)",
    "play_window":   "Play / Pause Loop Window (Space)",
    "save":          "Save Annotations CSV (Ctrl+S)",
}


class ShortcutManager:
    def __init__(self):
        self.shortcuts: dict[str, str] = dict(DEFAULT_SHORTCUTS)
        self.load()

    def load(self):
        if not os.path.exists(SHORTCUT_FILE):
            self.save()
            return
        try:
            with open(SHORTCUT_FILE, "r", encoding="utf-8") as f:
                saved = json.load(f)
            if isinstance(saved, dict):
                for k, v in saved.items():
                    if k in self.shortcuts and isinstance(v, str):
                        self.shortcuts[k] = v
        except Exception as e:
            print(f"[Warning] Error loading {SHORTCUT_FILE}: {e}")
            self.shortcuts = dict(DEFAULT_SHORTCUTS)
            self.save()

    def save(self):
        try:
            os.makedirs(os.path.dirname(os.path.abspath(SHORTCUT_FILE)), exist_ok=True)
            with open(SHORTCUT_FILE, "w", encoding="utf-8") as f:
                json.dump(self.shortcuts, f, indent=2)
            print(f"[Info] Saved keyboard shortcuts to {SHORTCUT_FILE}")
        except Exception as e:
            print(f"[Error] Failed to save {SHORTCUT_FILE}: {e}")

    def get(self, action: str) -> str:
        return self.shortcuts.get(action, DEFAULT_SHORTCUTS.get(action, ""))

    def set(self, action: str, key: str):
        self.shortcuts[action] = key.lower()
        self.save()

    def reset_defaults(self):
        self.shortcuts = dict(DEFAULT_SHORTCUTS)
        self.save()


class ShortcutEditorDialog(ctk.CTkToplevel):
    def __init__(self, parent, mgr: ShortcutManager, on_save_callback=None):
        super().__init__(parent)
        self.title("⌨ Customize Keyboard Shortcuts")
        self.geometry("520x560")
        self.configure(fg_color="#1e1e1e")
        self.transient(parent)
        self.grab_set()

        self.mgr = mgr
        self.on_save_callback = on_save_callback
        self.listening_action = None
        self.buttons = {}

        self._build_ui()

    def _build_ui(self):
        ctk.CTkLabel(
            self, text="⌨ Keyboard Shortcuts Editor",
            font=ctk.CTkFont(family="Helvetica", size=18, weight="bold"),
            text_color="#ffffff"
        ).pack(pady=(16, 6))

        ctk.CTkLabel(
            self, text="Click any button and press a key to rebind that shortcut.",
            font=ctk.CTkFont(family="Helvetica", size=12), text_color="#858585"
        ).pack(pady=(0, 16))

        scroll = ctk.CTkScrollableFrame(self, fg_color="#252526", corner_radius=6, border_width=1, border_color="#3c3c3c")
        scroll.pack(fill="both", expand=True, padx=16, pady=(0, 16))

        for action, label in ACTION_LABELS.items():
            row = ctk.CTkFrame(scroll, fg_color="transparent")
            row.pack(fill="x", padx=12, pady=6)

            ctk.CTkLabel(
                row, text=label, font=ctk.CTkFont(family="Helvetica", size=12),
                text_color="#cccccc", anchor="w"
            ).pack(side="left", fill="x", expand=True)

            curr_key = self.mgr.get(action).upper()
            btn = ctk.CTkButton(
                row, text=curr_key, width=90, height=28,
                fg_color="#3a3d41", hover_color="#007acc",
                font=ctk.CTkFont(family="Helvetica", size=12, weight="bold"),
                command=lambda a=action: self._start_listening(a)
            )
            btn.pack(side="right")
            self.buttons[action] = btn

        # Bottom buttons
        btn_bar = ctk.CTkFrame(self, fg_color="transparent")
        btn_bar.pack(fill="x", padx=16, pady=(0, 16))

        ctk.CTkButton(
            btn_bar, text="Reset Defaults", width=120, fg_color="#3a3d41",
            command=self._reset_defaults
        ).pack(side="left")

        ctk.CTkButton(
            btn_bar, text="Done / Close", width=120, fg_color="#007acc",
            command=self._on_close
        ).pack(side="right")

        self.bind("<KeyPress>", self._on_key_press)

    def _start_listening(self, action: str):
        self.listening_action = action
        self.buttons[action].configure(text="PRESS KEY...", fg_color="#ce9178")

    def _on_key_press(self, event):
        if not self.listening_action:
            return

        key = event.keysym.lower()
        if event.char and event.char.isalnum():
            key = event.char.lower()

        self.mgr.set(self.listening_action, key)
        self.buttons[self.listening_action].configure(text=key.upper(), fg_color="#3a3d41")
        self.listening_action = None

        if self.on_save_callback:
            self.on_save_callback()

    def _reset_defaults(self):
        self.mgr.reset_defaults()
        for action, btn in self.buttons.items():
            btn.configure(text=self.mgr.get(action).upper(), fg_color="#3a3d41")
        if self.on_save_callback:
            self.on_save_callback()

    def _on_close(self):
        self.destroy()
