"""
services/settings_service.py

Service responsible for reading and persisting environment and layout XML configuration.
Adheres to Single Responsibility Principle (SRP) by isolating file I/O and format logic.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any
from xml.dom import minidom

from utils.logger import setup_logger

logger = setup_logger("SettingsService")


class SettingsService:
    """Handles reading and writing configuration key-value pairs to .env and layout XML."""

    def __init__(self, env_path: str | Path) -> None:
        self._env_path = Path(env_path)

    @property
    def env_path(self) -> Path:
        return self._env_path

    # ── .env Persistence ───────────────────────────────────────────────────────

    def load_raw_entries(self) -> dict[str, str]:
        """Reads raw key-value pairs from the .env file."""
        entries: dict[str, str] = {}
        if not self._env_path.exists():
            return entries

        try:
            for line in self._env_path.read_text(encoding="utf-8").splitlines():
                stripped = line.strip()
                if stripped and not stripped.startswith("#") and "=" in stripped:
                    key, _, val = stripped.partition("=")
                    entries[key.strip()] = val.strip()
        except Exception as exc:
            logger.error("Failed to read settings from %s: %s", self._env_path, exc)
            raise

        return entries

    def save_settings(
        self,
        updates: dict[str, Any],
        xml_path: str | Path | None = None,
    ) -> None:
        """
        Updates specific configuration keys in the .env file and optionally in layout XML.
        """
        existing = self.load_raw_entries()
        for k, v in updates.items():
            existing[k] = str(v)

        lines: list[str] = [f"{k}={v}" for k, v in existing.items()]
        try:
            self._env_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            logger.info("Saved %d settings entries to %s", len(lines), self._env_path)
        except Exception as exc:
            logger.error("Failed to write settings to %s: %s", self._env_path, exc)
            raise

        if xml_path:
            p = Path(xml_path)
            if p.exists():
                self.save_xml_settings(p, updates)

    # ── XML Layout Persistence ─────────────────────────────────────────────────

    def load_xml_settings(self, xml_path: str | Path) -> dict[str, str]:
        """Reads settings from the <DetectorSettings> section of a layout XML file."""
        path = Path(xml_path)
        settings: dict[str, str] = {}
        if not path.exists():
            return settings

        try:
            tree = ET.parse(path)
            root = tree.getroot()
            det_settings = root.find("DetectorSettings")
            if det_settings is None:
                designer = root.find("DesignerLayout")
                if designer is not None:
                    det_settings = designer.find("DetectorSettings")

            if det_settings is not None:
                # 1. Read child <Setting name="..." value="..." /> tags
                for s_el in det_settings.findall("Setting"):
                    name = s_el.attrib.get("name") or s_el.attrib.get("key")
                    val = s_el.attrib.get("value")
                    if name and val is not None:
                        settings[name.strip()] = val.strip()

                # 2. Read attributes directly from <DetectorSettings ... />
                for attr_k, attr_v in det_settings.attrib.items():
                    norm_k = attr_k.upper()
                    if norm_k not in settings and norm_k != "COUNT":
                        settings[norm_k] = str(attr_v).strip()

            logger.info("Loaded %d settings entries from XML: %s", len(settings), path.name)
        except Exception as exc:
            logger.warning("Could not read settings from XML (%s): %s", path.name, exc)

        return settings

    def save_xml_settings(self, xml_path: str | Path, settings: dict[str, Any]) -> None:
        """
        Writes settings into the <DetectorSettings> section of the layout XML file,
        preserving all existing sections (<DesignerLayout>, <DetectorActions>, etc.).
        """
        path = Path(xml_path)
        if not path.exists():
            return

        try:
            tree = ET.parse(path)
            root = tree.getroot()

            # Remove existing DetectorSettings element if present
            det_el = root.find("DetectorSettings")
            if det_el is not None:
                root.remove(det_el)

            # Create new DetectorSettings element
            det_el = ET.SubElement(root, "DetectorSettings")
            det_el.set("count", str(len(settings)))

            for k, v in settings.items():
                s_el = ET.SubElement(det_el, "Setting")
                s_el.set("name", str(k))
                s_el.set("value", str(v))

            rough_string = ET.tostring(root, "utf-8")
            pretty = minidom.parseString(rough_string).toprettyxml(indent="  ")
            lines = [l for l in pretty.splitlines() if l.strip()]
            clean_xml = "\n".join(lines) + "\n"

            path.write_text(clean_xml, encoding="utf-8")
            logger.info("Successfully persisted DetectorSettings to XML: %s", path.name)
        except Exception as exc:
            logger.error("Failed to write settings to XML (%s): %s", path.name, exc)
            raise
