"""
utils/logger.py

Shared structured logger factory — mirrors the pattern used by the designer app.
"""

from __future__ import annotations

import logging
import sys


def setup_logger(name: str, level: int = logging.DEBUG) -> logging.Logger:
    """Creates and returns a named logger with a console handler."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(level)
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)
    fmt = logging.Formatter(
        "[%(asctime)s] %(levelname)-8s %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )
    handler.setFormatter(fmt)
    logger.addHandler(handler)
    logger.propagate = False
    return logger
