"""
viewmodels/action_config_viewmodel.py

ViewModel for the key action configuration view.
Manages loading, editing, selecting, and saving per-key action mappings into the unified XML layout.
"""

from __future__ import annotations

from PySide6.QtCore import QObject, Signal

from core.action.action_executor import ActionData
from core.layout.layout_parser import ButtonData, LayoutData
from services.action_config_service import ActionConfigService
from utils.logger import setup_logger

logger = setup_logger("ActionConfigViewModel")


class ActionConfigViewModel(QObject):
    """Manages action config state for all buttons in the loaded layout."""

    config_saved       = Signal(str)            # xml path saved to
    action_changed     = Signal(str, str, str)  # (button_id, type, value)
    button_selected    = Signal(str)           # button_id
    duplicate_rejected = Signal(str, str, str, str, str)  # (button_id, type, value, conflict_label, conflict_id)
    error_occurred     = Signal(str)

    def __init__(
        self,
        layout: LayoutData,
        xml_path: str,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._layout   = layout
        self._xml_path = xml_path
        self._svc      = ActionConfigService()
        self._selected_id: str | None = None

        # Load existing config from XML (or build defaults)
        button_ids = [b.id for b in layout.buttons]
        self._config: dict[str, ActionData] = self._svc.load(xml_path, button_ids)

    # ── Public API ─────────────────────────────────────────────────────────────

    @property
    def layout(self) -> LayoutData:
        return self._layout

    @property
    def buttons(self) -> list[ButtonData]:
        return self._layout.buttons

    @property
    def config(self) -> dict[str, ActionData]:
        return dict(self._config)

    @property
    def selected_button_id(self) -> str | None:
        return self._selected_id

    def select_button(self, button_id: str) -> None:
        self._selected_id = button_id
        self.button_selected.emit(button_id)

    def get_button(self, button_id: str) -> ButtonData | None:
        return next((b for b in self._layout.buttons if b.id == button_id), None)

    def get_action(self, button_id: str) -> ActionData:
        return self._config.get(button_id, ActionData(type="none", value=""))

    def find_duplicate(self, button_id: str, action_type: str, value: str) -> tuple[str, str] | None:
        """
        Check if (action_type, value) is already assigned to another key.
        Returns (conflicting_button_id, conflicting_button_label) if found, else None.
        """
        if action_type == "none" or not value.strip():
            return None

        val_norm = value.strip().lower()
        for b in self._layout.buttons:
            if b.id == button_id:
                continue
            act = self._config.get(b.id)
            if not act or act.type == "none" or not act.value.strip():
                continue
            if act.type == action_type and act.value.strip().lower() == val_norm:
                label = b.label.strip() if b.label.strip() else b.id
                return (b.id, label)
        return None

    def validate_all_actions(self) -> tuple[bool, str]:
        """Verify there are no duplicate keystroke or shortcut bindings across all keys."""
        seen: dict[tuple[str, str], str] = {}
        for b in self._layout.buttons:
            act = self._config.get(b.id)
            if not act or act.type == "none" or not act.value.strip():
                continue
            key_tuple = (act.type, act.value.strip().lower())
            if key_tuple in seen:
                other_id = seen[key_tuple]
                other_b = self.get_button(other_id)
                other_lbl = other_b.label.strip() if other_b and other_b.label.strip() else other_id
                curr_lbl = b.label.strip() if b.label.strip() else b.id
                return False, f"Duplicate {act.type} '{act.value}' between '{other_lbl}' ({other_id}) and '{curr_lbl}' ({b.id})."
            seen[key_tuple] = b.id
        return True, ""

    def set_action(self, button_id: str, action_type: str, value: str) -> bool:
        if action_type != "none" and value.strip():
            duplicate = self.find_duplicate(button_id, action_type, value)
            if duplicate:
                other_id, other_label = duplicate
                btn = self.get_button(button_id)
                curr_label = btn.label.strip() if btn and btn.label.strip() else button_id
                logger.warning(
                    f"Duplicate {action_type} '{value}' rejected for '{curr_label}' ({button_id}) "
                    f"— already assigned to '{other_label}' ({other_id})"
                )
                self.duplicate_rejected.emit(button_id, action_type, value, other_label, other_id)
                # Re-emit existing valid state to revert UI widgets
                current_valid_act = self.get_action(button_id)
                self.action_changed.emit(button_id, current_valid_act.type, current_valid_act.value)
                return False

        self._config[button_id] = ActionData(type=action_type, value=value)
        self.action_changed.emit(button_id, action_type, value)
        return True

    def save(self) -> bool:
        valid, err = self.validate_all_actions()
        if not valid:
            self.error_occurred.emit(err)
            return False
        try:
            self._svc.save(self._xml_path, self._config)
            self.config_saved.emit(self._xml_path)
            return True
        except Exception as exc:
            self.error_occurred.emit(str(exc))
            return False

    @property
    def xml_path(self) -> str:
        return self._xml_path
