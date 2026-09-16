"""
core/action/action_executor.py

Executes configured key actions when a touch event fires.

Supported action types
──────────────────────
  keystroke  — single key (e.g. "a", "enter", "f5", "space")
  shortcut   — key combo (e.g. "ctrl+c", "ctrl+shift+t")
  shell      — shell command string passed to subprocess
  macro      — colon-separated steps: "ctrl+c:200:ctrl+v"  (integers = ms delay)
  none       — no-op placeholder
"""

from __future__ import annotations

import shlex
import subprocess
import time
from dataclasses import dataclass

from utils.logger import setup_logger

logger = setup_logger("ActionExecutor")

# ── Domain objects ─────────────────────────────────────────────────────────────

@dataclass
class ActionData:
    """Holds a single key's configured action."""
    type: str     # "keystroke" | "shortcut" | "shell" | "macro" | "none"
    value: str    # action payload (key name, combo string, shell cmd, macro steps)

    @property
    def is_active(self) -> bool:
        return self.type != "none" and bool(self.value.strip())


# ── Executor ───────────────────────────────────────────────────────────────────

class ActionExecutor:
    """
    Executes ActionData objects.
    Uses pynput for keyboard/mouse events; falls back to xdotool subprocess
    if pynput raises a permission error (common on Wayland without config).
    """

    def __init__(self) -> None:
        self._keyboard = None
        self._pynput_ok = True
        try:
            from pynput.keyboard import Controller
            self._keyboard = Controller()
        except Exception as exc:
            logger.warning("pynput unavailable (%s). Falling back to xdotool.", exc)
            self._pynput_ok = False

    # ── Public API ─────────────────────────────────────────────────────────────

    def execute(self, action: ActionData) -> None:
        if not action.is_active:
            return
        dispatch = {
            "keystroke": self._keystroke,
            "shortcut":  self._shortcut,
            "shell":     self._shell,
            "macro":     self._macro,
        }
        handler = dispatch.get(action.type)
        if handler:
            try:
                handler(action.value.strip())
            except Exception as exc:
                logger.error("Action execution failed (%s): %s", action.type, exc)

    # ── Private handlers ───────────────────────────────────────────────────────

    def _keystroke(self, value: str) -> None:
        """Press and release a single key."""
        if self._pynput_ok:
            key = self._resolve_key(value)
            self._keyboard.press(key)
            self._keyboard.release(key)
        else:
            subprocess.Popen(["xdotool", "key", value])

    def _shortcut(self, value: str) -> None:
        """Press a key combination like 'ctrl+c'."""
        if self._pynput_ok:
            from pynput.keyboard import Key
            parts = [p.strip().lower() for p in value.split("+")]
            modifiers, main_key = [], None
            for p in parts:
                mod = self._MODIFIER_MAP.get(p)
                if mod:
                    modifiers.append(mod)
                else:
                    main_key = self._resolve_key(p)

            # Press modifiers, tap main key, release modifiers
            for m in modifiers:
                self._keyboard.press(m)
            if main_key:
                self._keyboard.press(main_key)
                self._keyboard.release(main_key)
            for m in reversed(modifiers):
                self._keyboard.release(m)
        else:
            # xdotool uses '+' separator
            subprocess.Popen(["xdotool", "key", value.replace(" ", "")])

    def _shell(self, value: str) -> None:
        """Run a shell command string non-blocking."""
        subprocess.Popen(shlex.split(value))

    def _macro(self, value: str) -> None:
        """Execute colon-separated steps; integers are interpreted as ms delays."""
        for step in value.split(":"):
            step = step.strip()
            if not step:
                continue
            try:
                delay_ms = int(step)
                time.sleep(delay_ms / 1000.0)
            except ValueError:
                # Treat as a keystroke or shortcut
                if "+" in step:
                    self._shortcut(step)
                else:
                    self._keystroke(step)

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _resolve_key(self, name: str):
        """Returns a pynput Key enum member or a plain character string."""
        from pynput.keyboard import Key
        name_lower = name.lower()
        if hasattr(Key, name_lower):
            return getattr(Key, name_lower)
        return name  # single character

    _MODIFIER_MAP: dict[str, object] = {}  # populated lazily below


def _build_modifier_map() -> dict:
    try:
        from pynput.keyboard import Key
        return {
            "ctrl":  Key.ctrl,
            "alt":   Key.alt,
            "shift": Key.shift,
            "meta":  Key.cmd,
            "super": Key.cmd,
            "win":   Key.cmd,
            "cmd":   Key.cmd,
        }
    except Exception:
        return {}


ActionExecutor._MODIFIER_MAP = _build_modifier_map()
