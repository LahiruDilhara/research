#!/usr/bin/env python3
"""
annotator.py — Entry point for the Touch Detection Data Annotator.

Run with:
    python3 annotator.py
"""
import logging
import os
import sys

# Configure stdout logging immediately
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

logger = logging.getLogger("Annotator.Main")
logger.info("Initializing Touch Detection Data Annotator application...")

# Ensure project root is first on the sys.path
root_dir = os.path.dirname(os.path.abspath(__file__))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)
logger.info(f"Set project root in sys.path: {root_dir}")

from annotator.ui.app import AnnotatorApp

if __name__ == "__main__":
    logger.info("Instantiating AnnotatorApp UI window...")
    app = AnnotatorApp()
    logger.info("Starting Tkinter main event loop (app.mainloop)...")
    try:
        app.mainloop()
    except Exception as e:
        logger.exception(f"Unhandled exception in main loop: {e}")
    logger.info("Application exited.")
