"""
summary_utils.py
================
Shared helper utility for logging pipeline stage summaries to dataprocessing/summaries/ as JSON.
"""

import json
from pathlib import Path


def get_summary_dir() -> Path:
    d = Path("./dataprocessing/summaries")
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_step_summary(step_filename: str, summary_data: dict):
    """
    Saves a dictionary of metrics as a JSON file in ./dataprocessing/summaries/
    """
    try:
        summary_dir = get_summary_dir()
        out_path = summary_dir / step_filename
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(summary_data, f, indent=2)
        print(f"  [Summary Logger] Stage audit metrics recorded → {out_path}")
    except Exception as e:
        print(f"  [Warning] Failed to write summary JSON '{step_filename}': {e}")
