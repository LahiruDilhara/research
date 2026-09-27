"""
core/interfaces/touch_model.py

AI model plugin contract for touch detection models.

Usage (AI model plugin author side)
───────────────────────────────────
1. Subclass ITouchModel.
2. Decorate with @register_model(...).
3. Implement load() and predict().
4. Place the .py file and the .pth weights inside any sub-directory of ai_model_plugins/.

The ModelDiscoveryService imports every .py in ai_model_plugins/, which triggers the decorator
and self-registers the class in ModelRegistry without manual registration needed.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable


# ── Registry ──────────────────────────────────────────────────────────────────

@dataclass
class ModelEntry:
    name: str
    description: str
    weights_file: str
    weights_path: str          # resolved absolute path (set by ModelDiscoveryService)
    cls: type["ITouchModel"]
    instance: "ITouchModel | None" = field(default=None, repr=False)
    is_primary: bool = False


class ModelRegistry:
    """Singleton in-process registry of all discovered AI model plugins."""

    _entries: list[ModelEntry] = []

    @classmethod
    def register(cls, entry: ModelEntry) -> None:
        # Avoid duplicates by name
        if not any(e.name == entry.name for e in cls._entries):
            cls._entries.append(entry)

    @classmethod
    def all_entries(cls) -> list[ModelEntry]:
        return list(cls._entries)

    @classmethod
    def get_by_name(cls, name: str) -> ModelEntry | None:
        return next((e for e in cls._entries if e.name == name), None)

    @classmethod
    def get_primary(cls) -> ModelEntry | None:
        """Returns the first primary model found, or falls back to the first available model."""
        for e in cls._entries:
            if e.is_primary:
                return e
        return cls._entries[0] if cls._entries else None

    @classmethod
    def clear(cls) -> None:
        cls._entries.clear()


# ── Decorators ────────────────────────────────────────────────────────────────

def register_model(
    name: str,
    description: str,
    weights_file: str,
    is_primary: bool = False,
) -> Callable[[type], type]:
    """
    Class decorator that auto-registers an ITouchModel subclass in ModelRegistry.
    """

    def decorator(cls: type) -> type:
        if not issubclass(cls, ITouchModel):
            raise TypeError(f"@register_model can only decorate ITouchModel subclasses, got {cls}")

        cls._plugin_name = name
        cls._plugin_description = description
        cls._plugin_weights_file = weights_file
        cls._is_primary = getattr(cls, "_is_primary", is_primary)

        entry = ModelEntry(
            name=name,
            description=description,
            weights_file=weights_file,
            weights_path="",
            cls=cls,
            is_primary=cls._is_primary,
        )
        ModelRegistry.register(entry)
        return cls

    return decorator


def primaryModel(cls: type) -> type:
    """
    Decorator that marks an ITouchModel plugin class as the primary active model.
    When plugins are scanned, the primary model is automatically loaded and activated.
    """
    cls._is_primary = True
    for entry in ModelRegistry.all_entries():
        if entry.cls is cls:
            entry.is_primary = True
    return cls


# ── Abstract base class ────────────────────────────────────────────────────────

class ITouchModel(ABC):
    """
    Interface every touch-detection AI model plugin must implement.

    The detector calls predict() for every 5-frame window.
    The plugin is fully responsible for its own feature extraction.
    """

    # Set by @register_model
    _plugin_name: str = ""
    _plugin_description: str = ""
    _plugin_weights_file: str = ""

    @abstractmethod
    def load(self, weights_path: str) -> None:
        """
        Load trained weights from the given absolute path.
        Called once by ModelDiscoveryService after the plugin is instantiated.
        """

    @abstractmethod
    def predict(self, window_5_frames: list[dict[str, float]]) -> dict[str, float]:
        """
        Run touch detection on a 5-frame window.

        Parameters
        ----------
        window_5_frames : list of 5 dicts, each containing 63 keys:
                          "<landmark_name>_x", "<landmark_name>_y", "<landmark_name>_z"
                          for all 21 hand landmarks, scale-normalised and wrist-centred
                          (matches the output of HandScaleNormalizer.build_norm_dict()).

        Returns
        -------
        dict mapping finger name → touch probability (0.0 .. 1.0):
            {"Thumb": 0.92, "Index": 0.07, "Middle": 0.02, "Ring": 0.01, "Pinky": 0.01}
        All five fingers must be present in the returned dict.
        """
