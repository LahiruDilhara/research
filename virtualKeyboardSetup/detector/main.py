"""
main.py — Virtual Keyboard Detector

Entry point.  Bootstraps the Qt application, loads config, and launches the
main FluentWindow shell.

Usage
─────
  uv run python main.py
"""

from __future__ import annotations

import sys
import os
from pathlib import Path

# Suppress Qt "Could not parse stylesheet" debug messages from qfluentwidgets'
# internal dark-mode theming. These are cosmetic only and not our code's error.
os.environ.setdefault("QT_LOGGING_RULES", "qt.qpa.*.warning=false")

from PySide6.QtCore import Qt, qInstallMessageHandler, QtMsgType
from PySide6.QtWidgets import QApplication
from qfluentwidgets import Theme, setTheme

from config.app_config import AppConfig
from ui.main_window import MainWindow


def _qt_message_handler(msg_type, context, message: str) -> None:
    """Drop noisy 'Could not parse stylesheet' warnings from qfluentwidgets."""
    if "Could not parse stylesheet" in message:
        return
    if msg_type == QtMsgType.QtWarningMsg:
        print(f"[Qt WARNING] {message}", file=sys.stderr)
    elif msg_type == QtMsgType.QtCriticalMsg or msg_type == QtMsgType.QtFatalMsg:
        print(f"[Qt CRITICAL] {message}", file=sys.stderr)


def main() -> None:
    # High-DPI scaling
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )

    app = QApplication(sys.argv)
    # Suppress noisy qfluentwidgets internal CSS parse warnings
    qInstallMessageHandler(_qt_message_handler)
    app.setApplicationName("Virtual Keyboard Detector")
    app.setOrganizationName("VKB Research")

    # Load config from .env in same directory
    env_path = Path(__file__).resolve().parent / ".env"
    config = AppConfig(env_path)

    # Force dark theme — matches the designer
    setTheme(Theme.DARK)

    window = MainWindow(config)
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
