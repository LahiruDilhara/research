"""
annotator/utils.py

File-hash computation, CSV filename conventions, and native OS system file dialogs.
Runs native Linux system dialogs (zenity/kdialog) in a non-blocking worker thread
while maintaining Tkinter event pump responsiveness.
"""
import hashlib
import logging
import os
import shutil
import subprocess
import threading
import tkinter
from tkinter import filedialog

logger = logging.getLogger("Annotator.Utils")


def compute_file_hash(filepath: str, chunk_size: int = 65536) -> str:
    """SHA-256 of the file; returns first 16 hex chars for compact filenames."""
    logger.info(f"Starting SHA-256 hash calculation for file: {filepath}")
    sha = hashlib.sha256()
    bytes_processed = 0
    with open(filepath, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            sha.update(chunk)
            bytes_processed += len(chunk)
    digest = sha.hexdigest()[:16]
    logger.info(f"Completed SHA-256 hash for {filepath} ({bytes_processed} bytes): {digest}")
    return digest


def build_csv_path(directory: str, base_name: str, video_hash: str) -> str:
    filename = f"{base_name}.{video_hash}.csv"
    path = os.path.join(directory, filename)
    logger.info(f"Built CSV path: {path}")
    return path


def extract_hash_from_csv_filename(csv_path: str) -> str | None:
    """
    Extract expected SHA-256 video_hash.
    First checks the CSV content's 'video_hash' column, then falls back to the filename stem.
    """
    if os.path.isfile(csv_path):
        try:
            from annotator.csv_manager import CSVManager
            h = CSVManager(csv_path).get_video_hash()
            if h:
                logger.info(f"Extracted expected video_hash '{h}' from CSV content: {csv_path}")
                return h
        except Exception as e:
            logger.debug(f"Could not read hash from CSV records: {e}")

    stem = os.path.splitext(os.path.basename(csv_path))[0]
    parts = stem.rsplit(".", 1)
    extracted = parts[1] if len(parts) == 2 and len(parts[1]) >= 8 else None
    logger.info(f"Extracted expected hash '{extracted}' from CSV filename stem: {csv_path}")
    return extracted


# ── Native OS System File Dialogs ─────────────────────────────────────────────

def _has_cmd(cmd: str) -> bool:
    return shutil.which(cmd) is not None


def _run_native_cli_dialog(args: list[str]) -> str | None:
    """
    Executes a native Linux CLI dialog (e.g. zenity / kdialog) in a worker thread.
    Pumps Tkinter mainloop events via root.update() so the GUI never freezes.
    """
    result = [None]

    def _worker():
        try:
            res = subprocess.run(args, capture_output=True, text=True, timeout=300)
            if res.returncode == 0 and res.stdout.strip():
                result[0] = res.stdout.strip()
        except Exception as e:
            logger.debug(f"CLI dialog command {args[0]} failed: {e}")

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


def open_video_dialog() -> str | None:
    """Open system native OS file picker for video files."""
    logger.info("Opening system native OS file dialog for video selection...")
    
    if _has_cmd("zenity"):
        path = _run_native_cli_dialog([
            "zenity", "--file-selection",
            "--title=Select Video File",
            "--file-filter=Video Files (*.mp4, *.avi, *.mov, *.mkv) | *.mp4 *.avi *.mov *.mkv *.webm *.MP4 *.AVI *.MOV",
            "--file-filter=All Files | *",
        ])
        if path:
            logger.info(f"User selected video file via zenity: {path}")
            return path

    if _has_cmd("kdialog"):
        path = _run_native_cli_dialog([
            "kdialog", "--getopenfilename", os.path.expanduser("~"),
            "*.mp4 *.avi *.mov *.mkv *.webm",
        ])
        if path:
            logger.info(f"User selected video file via kdialog: {path}")
            return path

    path = filedialog.askopenfilename(
        title="Select Video File",
        filetypes=[
            ("Video files", "*.mp4 *.avi *.mov *.mkv *.webm *.MP4 *.AVI *.MOV"),
            ("All files", "*.*"),
        ],
    )
    if path:
        logger.info(f"User selected video file via tkinter: {path}")
        return path
    else:
        logger.info("User cancelled video file selection.")
        return None


def open_csv_dialog() -> str | None:
    """Open system native OS file picker for CSV files."""
    logger.info("Opening system native OS file dialog for CSV selection...")
    
    if _has_cmd("zenity"):
        path = _run_native_cli_dialog([
            "zenity", "--file-selection",
            "--title=Open Existing CSV",
            "--file-filter=CSV Files (*.csv) | *.csv",
            "--file-filter=All Files | *",
        ])
        if path:
            logger.info(f"User selected CSV file via zenity: {path}")
            return path

    if _has_cmd("kdialog"):
        path = _run_native_cli_dialog([
            "kdialog", "--getopenfilename", os.path.expanduser("~"), "*.csv",
        ])
        if path:
            logger.info(f"User selected CSV file via kdialog: {path}")
            return path

    path = filedialog.askopenfilename(
        title="Open Existing CSV",
        filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
    )
    if path:
        logger.info(f"User selected CSV file via tkinter: {path}")
        return path
    else:
        logger.info("User cancelled CSV file selection.")
        return None


def save_csv_dialog(initial_name: str = "annotation_data") -> str | None:
    """Open system native OS save dialog. Returns selected path or None."""
    logger.info("Opening system native OS save dialog for CSV location...")
    home = os.path.expanduser("~")
    start = os.path.join(home, initial_name)

    if _has_cmd("zenity"):
        path = _run_native_cli_dialog([
            "zenity", "--file-selection", "--save",
            "--title=Save CSV As (extension added automatically)",
            f"--filename={start}",
        ])
        if path:
            logger.info(f"User specified CSV save location via zenity: {path}")
            return path

    if _has_cmd("kdialog"):
        path = _run_native_cli_dialog([
            "kdialog", "--getsavefilename", start,
        ])
        if path:
            logger.info(f"User specified CSV save location via kdialog: {path}")
            return path

    path = filedialog.asksaveasfilename(
        title="Save CSV As (extension added automatically)",
        initialfile=initial_name,
        initialdir=home,
        filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
    )
    if path:
        logger.info(f"User specified CSV save location via tkinter: {path}")
        return path
    else:
        logger.info("User cancelled CSV save dialog.")
        return None
