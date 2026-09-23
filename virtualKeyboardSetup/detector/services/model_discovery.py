"""
services/model_discovery.py

Scans the AI model plugins directory, imports every Python file found inside it,
and relies on the @register_model decorator in each file to self-register
the model in ModelRegistry.

After discovery, sets the resolved weights_path on each ModelEntry.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from core.interfaces.touch_model import ModelRegistry
from utils.logger import setup_logger

logger = setup_logger("AIModelDiscovery")


class ModelDiscoveryService:
    """Scans an AI model plugins directory and populates ModelRegistry."""

    def __init__(self, plugins_dir: str | Path) -> None:
        self._plugins_dir = Path(plugins_dir)

    def discover(self) -> int:
        """
        Import all .py files under plugins_dir (recursively).
        Each file that contains a @register_model-decorated class will
        automatically register itself in ModelRegistry upon import.

        Returns the number of successfully discovered model entries.
        """
        if not self._plugins_dir.exists():
            logger.warning("Plugins directory not found: %s", self._plugins_dir)
            return 0

        py_files = [
            p for p in self._plugins_dir.rglob("*.py")
            if p.name != "__init__.py" and not p.name.startswith("_")
        ]

        logger.info("Scanning %d plugin files in %s ...", len(py_files), self._plugins_dir)

        for py_file in py_files:
            self._import_file(py_file)

        # Resolve weights paths for all registered entries
        for entry in ModelRegistry.all_entries():
            if not entry.weights_path:
                # Find any .pth file with the expected name anywhere under plugins/
                candidates = list(self._plugins_dir.rglob(entry.weights_file))
                if candidates:
                    entry.weights_path = str(candidates[0])
                    logger.info(
                        "Model '%s' weights resolved: %s",
                        entry.name, entry.weights_path,
                    )
                else:
                    logger.warning(
                        "Model '%s' weights file '%s' not found under %s",
                        entry.name, entry.weights_file, self._plugins_dir,
                    )

        count = len(ModelRegistry.all_entries())
        logger.info("Discovery complete: %d model(s) registered.", count)
        return count

    @staticmethod
    def _import_file(path: Path) -> None:
        """Dynamically imports a single .py file as a module."""
        module_name = f"_plugin_{path.stem}_{abs(hash(str(path)))}"
        try:
            spec = importlib.util.spec_from_file_location(module_name, path)
            if spec is None or spec.loader is None:
                return
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
            logger.debug("Imported plugin file: %s", path.name)
        except Exception as exc:
            logger.error("Failed to import plugin %s: %s", path.name, exc)


# Backwards-compatible alias for explicit naming
AIModelDiscoveryService = ModelDiscoveryService

