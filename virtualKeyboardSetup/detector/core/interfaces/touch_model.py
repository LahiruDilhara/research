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
    def clear(cls) -> None:
        cls._entries.clear()


# ── Decorator ─────────────────────────────────────────────────────────────────

def register_model(
    name: str,
    description: str,
    weights_file: str,
) -> Callable[[type], type]:
    """
    Class decorator that auto-registers an ITouchModel subclass in ModelRegistry.

    Parameters
    ----------
    name        : Human-readable model name shown in the UI dropdown.
    description : Short description of architecture / training dataset.
    weights_file: Filename of the .pth weights file expected in the same directory
                  as the plugin .py file (e.g. "LSTM_All_Combined_cfg01.pth").

    Example
    -------
    @register_model(
        name="LSTM All-Combined",
        description="2-layer LSTM trained on all hand joints coords+velocities.",
        weights_file="LSTM_All_Combined_cfg01.pth",
    )
    class LSTMAllCombinedPlugin(ITouchModel):
        ...
    """

    def decorator(cls: type) -> type:
        if not issubclass(cls, ITouchModel):
            raise TypeError(f"@register_model can only decorate ITouchModel subclasses, got {cls}")

        # Attach metadata to the class for later use by ModelDiscoveryService
        cls._plugin_name = name
        cls._plugin_description = description
        cls._plugin_weights_file = weights_file

        # Register a placeholder entry; weights_path is filled in by discovery service
        entry = ModelEntry(
            name=name,
            description=description,
            weights_file=weights_file,
            weights_path="",   # filled by ModelDiscoveryService after path resolution
            cls=cls,
        )
        ModelRegistry.register(entry)
        return cls

    return decorator


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
