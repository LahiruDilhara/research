"""
annotator/shortcuts.py

Keyboard shortcut manager — loads and saves shortcut bindings to a JSON
file alongside this module so settings persist across sessions.
"""
import json
import logging
import os

logger = logging.getLogger("Annotator.Shortcuts")

# Stored next to this file so it travels with the project
SHORTCUT_FILE = os.path.join(os.path.dirname(__file__), "shortcuts.json")

# Sensible defaults: fingers on Q W E R T, navigation on arrow keys
DEFAULT_SHORTCUTS: dict[str, str] = {
    "thumb":       "q",
    "index":       "w",
    "middle":      "e",
    "ring":        "r",
    "pinky":       "t",
    "next_window": "Right",
    "prev_window": "Left",
}

# Human-readable labels shown in the settings dialog
ACTION_LABELS: dict[str, str] = {
    "thumb":       "Toggle Thumb Touch",
    "index":       "Toggle Index Touch",
    "middle":      "Toggle Middle Touch",
    "ring":        "Toggle Ring Touch",
    "pinky":       "Toggle Pinky Touch",
    "next_window": "Next Window",
    "prev_window": "Previous Window",
}

# Pretty-print map for special tkinter keysym names
KEYSYM_DISPLAY: dict[str, str] = {
    "Right":     "→  Right",
    "Left":      "←  Left",
    "Up":        "↑  Up",
    "Down":      "↓  Down",
    "space":     "Space",
    "Return":    "Enter",
    "BackSpace": "Backspace",
    "Tab":       "Tab",
    "Escape":    "Esc",
    "Delete":    "Delete",
    "Home":      "Home",
    "End":       "End",
}


def keysym_to_display(keysym: str) -> str:
    """Convert a raw tkinter keysym string to a human-readable label."""
    if not keysym:
        return "—"
    return KEYSYM_DISPLAY.get(keysym, keysym.upper())


class ShortcutManager:
    """Load, mutate, and persist keyboard shortcut assignments."""

    def __init__(self) -> None:
        self._shortcuts: dict[str, str] = dict(DEFAULT_SHORTCUTS)
        self._load()

    # ── Persistence ───────────────────────────────────────────────────────────

    def _load(self) -> None:
        if not os.path.exists(SHORTCUT_FILE):
            return
        try:
            with open(SHORTCUT_FILE, "r", encoding="utf-8") as f:
                saved = json.load(f)
            if not isinstance(saved, dict):
                raise ValueError(f"Expected dict root in JSON, got {type(saved).__name__}")
            for key in DEFAULT_SHORTCUTS:
                if key in saved and isinstance(saved[key], str):
                    self._shortcuts[key] = saved[key]
            logger.info(f"Loaded keyboard shortcuts from {SHORTCUT_FILE}")
        except Exception as exc:
            logger.warning(
                f"Corrupted or invalid shortcut file '{SHORTCUT_FILE}' ({exc}). "
                f"Deleting and recreating with defaults."
            )
            try:
                if os.path.exists(SHORTCUT_FILE):
                    os.remove(SHORTCUT_FILE)
            except Exception as del_exc:
                logger.error(f"Failed to remove corrupted shortcut file: {del_exc}")
            self.reset_defaults()
            self.save()

    def save(self) -> None:
        """Write current bindings to the JSON file."""
        try:
            with open(SHORTCUT_FILE, "w", encoding="utf-8") as f:
                json.dump(self._shortcuts, f, indent=2)
            logger.info(f"Saved keyboard shortcuts to {SHORTCUT_FILE}")
        except Exception as exc:
            logger.warning(f"Could not save shortcuts file: {exc}")

    # ── Accessors ─────────────────────────────────────────────────────────────

    def get(self, action: str) -> str:
        """Return the keysym string bound to *action*, or empty string."""
        return self._shortcuts.get(action, "")

    def set(self, action: str, keysym: str) -> None:
        self._shortcuts[action] = keysym

    def all(self) -> dict[str, str]:
        """Return a copy of the full action→keysym mapping."""
        return dict(self._shortcuts)

    def reset_defaults(self) -> None:
        self._shortcuts = dict(DEFAULT_SHORTCUTS)
