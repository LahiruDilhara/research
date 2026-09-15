"""
Path utilities module.
Resolves absolute paths for application resources and output directories.
"""

import sys
from pathlib import Path


def get_app_root() -> Path:
    """Returns absolute path to root application directory."""
    return Path(__file__).resolve().parent.parent


def get_resource_path(relative_path: str) -> Path:
    """Resolves resource relative path to absolute filesystem path."""
    if hasattr(sys, "_MEIPASS"):
        return Path(sys._MEIPASS) / relative_path
    return get_app_root() / relative_path
