"""
services/action_config_service.py

Reads and writes per-key action mappings directly into the unified XML layout file
under the <DetectorActions> sub-section.

Format within the layout XML:
─────────────────────────────
<PaperLayout ...>
  <DesignerLayout>
    ...
  </DesignerLayout>
  <DetectorActions>
    <Actions count="2">
      <Action button_id="btn_1" label="A" type="keystroke" value="a" />
      <Action button_id="btn_2" label="Copy" type="shortcut" value="ctrl+c" />
    </Actions>
  </DetectorActions>
</PaperLayout>
"""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from xml.dom import minidom
from pathlib import Path

from core.action.action_executor import ActionData
from utils.logger import setup_logger

logger = setup_logger("ActionConfigService")


class ActionConfigService:
    """Loads and saves key→action mappings directly into the unified XML file."""

    VALID_TYPES = {"keystroke", "shortcut", "shell", "macro", "none"}

    def __init__(self) -> None:
        self._config: dict[str, ActionData] = {}
        self._xml_path: str = ""

    # ── Public API ─────────────────────────────────────────────────────────────

    def load(self, xml_path: str, button_ids: list[str]) -> dict[str, ActionData]:
        """
        Load actions from the <DetectorActions> section of xml_path.
        Falls back to legacy .actions.json if not yet present in the XML.
        Missing keys are populated with 'none' ActionData.
        """
        path = Path(xml_path)
        self._xml_path = str(path)
        raw: dict[str, dict] = {}

        if path.exists():
            try:
                tree = ET.parse(path)
                root = tree.getroot()
                det_el = root.find("DetectorActions")
                if det_el is not None:
                    actions_el = det_el.find("Actions") or det_el
                    for a_el in actions_el.findall("Action"):
                        b_id = a_el.attrib.get("button_id") or a_el.attrib.get("id")
                        if b_id:
                            raw[b_id] = {
                                "type": a_el.attrib.get("type", "none"),
                                "value": a_el.attrib.get("value", ""),
                            }
                    logger.info("Action config loaded from XML: %s (%d actions)", path.name, len(raw))
            except Exception as exc:
                logger.warning("Could not parse actions from XML (%s): %s", path.name, exc)

        # Fallback to legacy sidecar if XML had no actions
        if not raw and path.exists():
            sidecar = path.with_suffix(".actions.json")
            if sidecar.exists():
                try:
                    with open(sidecar, "r", encoding="utf-8") as f:
                        raw = json.load(f)
                    logger.info("Loaded actions from legacy sidecar: %s", sidecar.name)
                except Exception:
                    pass

        self._config = {}
        for btn_id in button_ids:
            entry = raw.get(btn_id, {})
            action_type = entry.get("type", "none")
            if action_type not in self.VALID_TYPES:
                action_type = "none"
            self._config[btn_id] = ActionData(
                type=action_type,
                value=str(entry.get("value", "")),
            )

        return dict(self._config)

    def save(self, xml_path: str, config: dict[str, ActionData]) -> None:
        """
        Write the action config dict into the <DetectorActions> section of the XML file,
        preserving the <DesignerLayout> section completely intact.
        """
        path = Path(xml_path)
        if not path.exists():
            raise FileNotFoundError(f"Layout XML file not found: {path}")

        try:
            tree = ET.parse(path)
            root = tree.getroot()

            # Find or create <DetectorActions>
            det_el = root.find("DetectorActions")
            if det_el is not None:
                root.remove(det_el)

            det_el = ET.SubElement(root, "DetectorActions")
            actions_el = ET.SubElement(det_el, "Actions", count=str(len(config)))

            # Read button labels from designer section for cleaner XML display
            label_map: dict[str, str] = {}
            designer_el = root.find("DesignerLayout") or root
            buttons_el = designer_el.find("Buttons")
            if buttons_el is not None:
                for b_el in buttons_el.findall("Button"):
                    bid = b_el.attrib.get("id", "")
                    txt_el = b_el.find("Text")
                    label_map[bid] = txt_el.text if (txt_el is not None and txt_el.text) else bid

            for btn_id, act in config.items():
                a_el = ET.SubElement(actions_el, "Action")
                a_el.set("button_id", str(btn_id))
                a_el.set("label", label_map.get(btn_id, str(btn_id)))
                a_el.set("type", act.type)
                a_el.set("value", act.value)

            rough_string = ET.tostring(root, "utf-8")
            pretty = minidom.parseString(rough_string).toprettyxml(indent="  ")

            # Remove empty lines produced by minidom pretty-printing
            lines = [l for l in pretty.splitlines() if l.strip()]
            clean_xml = "\n".join(lines) + "\n"

            with open(path, "w", encoding="utf-8") as f:
                f.write(clean_xml)

            self._config = dict(config)
            logger.info("Action config saved into XML: %s", path.name)
        except Exception as exc:
            logger.error("Failed to save action config into XML: %s", exc)
            raise

    def get_action(self, button_id: str) -> ActionData | None:
        return self._config.get(button_id)

    @property
    def config(self) -> dict[str, ActionData]:
        return dict(self._config)

    @staticmethod
    def sidecar_exists(xml_path: str) -> bool:
        path = Path(xml_path)
        if not path.exists():
            return False
        try:
            tree = ET.parse(path)
            root = tree.getroot()
            det_el = root.find("DetectorActions")
            if det_el is not None:
                actions_el = det_el.find("Actions") or det_el
                for a_el in actions_el.findall("Action"):
                    if a_el.attrib.get("type", "none") != "none":
                        return True
        except Exception:
            pass
        return path.with_suffix(".actions.json").exists()
