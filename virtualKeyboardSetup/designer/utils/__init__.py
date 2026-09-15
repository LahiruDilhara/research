"""Utilities package initialization."""
from .logger import setup_logger
from .path_utils import get_resource_path

__all__ = ["setup_logger", "get_resource_path"]
