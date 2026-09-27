"""
datacreator/annotator/utils.py

Native OS System File Dialogs.
Executes system file picker dialogs (zenity / kdialog / tkinter filedialog) in non-blocking worker threads
to provide a native OS file browsing experience.
"""

import os
import shutil
import subprocess
import threading
import tkinter
from tkinter import filedialog


def _has_cmd(cmd: str) -> bool:
    return shutil.which(cmd) is not None


def _get_default_dir(provided_path: str | None = None) -> str:
    """Returns a valid existing directory path for file dialogs, defaulting to current working directory."""
    if provided_path and os.path.exists(provided_path):
        if os.path.isdir(provided_path):
            return os.path.abspath(provided_path)
        return os.path.dirname(os.path.abspath(provided_path))

    return os.getcwd()



def _run_native_cli_dialog(args: list[str]) -> str | None:
    """Executes a native Linux CLI dialog (zenity/kdialog) without freezing the Tkinter event loop."""
    result = [None]

    def _worker():
        try:
            res = subprocess.run(args, capture_output=True, text=True, timeout=300)
            if res.returncode == 0 and res.stdout.strip():
                result[0] = res.stdout.strip()
        except Exception:
            pass

    thread = threading.Thread(target=_worker, daemon=True)
    thread.start()

    while thread.is_alive():
        try:
            if tkinter._default_root and tkinter._default_root.winfo_exists():
                tkinter._default_root.update()
        except Exception:
            pass
        thread.join(timeout=0.02)

    return result[0]


def open_video_dialog(initial_path: str | None = None) -> str | None:
    """Opens system native OS file picker for video selection."""
    start_dir = _get_default_dir(initial_path)

    if _has_cmd("zenity"):
        path = _run_native_cli_dialog([
            "zenity", "--file-selection",
            f"--filename={os.path.join(start_dir, '')}",
            "--title=Select 12 FPS Video File",
            "--file-filter=Video Files (*.mp4, *.webm, *.avi, *.mov) | *.mp4 *.webm *.avi *.mov",
            "--file-filter=All Files | *",
        ])
        if path:
            return path

    if _has_cmd("kdialog"):
        path = _run_native_cli_dialog([
            "kdialog", "--getopenfilename", start_dir,
            "*.mp4 *.webm *.avi *.mov",
        ])
        if path:
            return path

    path = filedialog.askopenfilename(
        title="Select 12 FPS Video File",
        initialdir=start_dir,
        filetypes=[
            ("Video Files", "*.mp4 *.webm *.avi *.mov"),
            ("All Files", "*.*"),
        ],
    )
    return path if path else None


def open_csv_dialog(initial_path: str | None = None) -> str | None:
    """Opens system native OS file picker for raw landmarks CSV selection."""
    start_dir = _get_default_dir(initial_path)

    if _has_cmd("zenity"):
        path = _run_native_cli_dialog([
            "zenity", "--file-selection",
            f"--filename={os.path.join(start_dir, '')}",
            "--title=Select Raw Landmarks CSV",
            "--file-filter=CSV Files (*.csv) | *.csv",
            "--file-filter=All Files | *",
        ])
        if path:
            return path

    if _has_cmd("kdialog"):
        path = _run_native_cli_dialog([
            "kdialog", "--getopenfilename", start_dir, "*.csv",
        ])
        if path:
            return path

    path = filedialog.askopenfilename(
        title="Select Raw Landmarks CSV",
        initialdir=start_dir,
        filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")],
    )
    return path if path else None

