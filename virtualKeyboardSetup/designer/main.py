"""
Application Main Bootstrap Script.
Initializes configuration, database connection, PySide6 QApplication, high-DPI attributes,
and launches the FluentMainWindow shell.
"""

import sys
from pathlib import Path
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

from config.app_config import AppConfig
from db.connection import DatabaseManager
from ui.main_window import MainWindow
from utils.logger import setup_logger

logger = setup_logger("PaperLayoutDesigner")


def main() -> None:
    # 1. Load Configuration
    env_file = Path(__file__).resolve().parent / ".env"
    config = AppConfig(env_file if env_file.exists() else None)
    logger.info("Configuration loaded successfully.")

    # 2. Initialize Database Connection & Tables
    db_manager = DatabaseManager(config.db_path)
    logger.info(f"Database initialized at: {db_manager.db_path}")

    # 3. Setup PySide6 Application & High-DPI Attributes
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )

    app = QApplication(sys.argv)
    app.setApplicationName(config.app_title)

    # Apply compact global font size (9.5pt)
    font = app.font()
    font.setPointSizeF(9.5)
    app.setFont(font)

    # 4. Launch Main Window
    window = MainWindow(config)
    window.show()
    logger.info("Application main window launched.")

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
