"""
annotator/csv_manager.py

Crash-safe CSV operations using a write-close pattern with logging.
The file is never held open between operations; every call opens, writes, closes.
"""
import csv
import logging
import os
from annotator.constants import CSV_HEADERS

logger = logging.getLogger("Annotator.CSVManager")


class CSVManager:
    """
    Manages a single CSV annotation file.
    """

    def __init__(self, csv_path: str) -> None:
        self.path = csv_path

    # ── Creation ─────────────────────────────────────────────────────────────

    def create(self) -> None:
        """Create a new CSV file with the standard header row."""
        logger.info(f"Creating new CSV file at: {self.path}")
        with open(self.path, "w", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=CSV_HEADERS).writeheader()

    # ── Read ─────────────────────────────────────────────────────────────────

    def read_all(self) -> list[dict]:
        """Return all records as a list of dicts (empty list if file missing)."""
        if not os.path.exists(self.path):
            logger.debug(f"CSV file does not exist: {self.path}")
            return []
        with open(self.path, "r", newline="", encoding="utf-8") as f:
            records = list(csv.DictReader(f))
            logger.debug(f"Read {len(records)} records from {self.path}")
            return records

    def get_video_hash(self) -> str | None:
        """Extract the SHA-256 video_hash column from the first record in the CSV."""
        recs = self.read_all()
        if recs and recs[0].get("video_hash"):
            h = recs[0]["video_hash"].strip()
            logger.info(f"Retrieved video_hash '{h}' from CSV records: {self.path}")
            return h
        return None

    def last_record(self) -> dict | None:
        recs = self.read_all()
        return recs[-1] if recs else None

    def total_records(self) -> int:
        return len(self.read_all())

    # ── Write ─────────────────────────────────────────────────────────────────

    def append(self, record: dict) -> None:
        """Append one record and immediately close the file."""
        sf = record.get("start_frame", "?")
        ef = record.get("end_frame", "?")
        logger.info(f"Appending record for frames {sf}–{ef} to CSV: {self.path}")
        full_rec = {h: record.get(h, "") for h in CSV_HEADERS}
        with open(self.path, "a", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=CSV_HEADERS).writerow(full_rec)

    def override(
        self,
        start_frame: int,
        end_frame: int,
        start_ms: int,
        end_ms: int,
        new_record: dict,
    ) -> bool:
        """
        Replace the matching record (identified by the four window-identity fields).
        """
        logger.info(f"Overriding existing record for frames {start_frame}–{end_frame} in CSV: {self.path}")
        recs = self.read_all()
        patched = False
        for i, r in enumerate(recs):
            if (
                _int(r, "start_frame") == start_frame
                and _int(r, "end_frame") == end_frame
                and _int(r, "start_ms") == start_ms
                and _int(r, "end_ms") == end_ms
            ):
                full = {h: new_record.get(h, "") for h in CSV_HEADERS}
                recs[i] = full
                patched = True
                break
        if patched:
            with open(self.path, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=CSV_HEADERS)
                w.writeheader()
                w.writerows(recs)
            logger.info(f"Successfully patched record in CSV: {self.path}")
        else:
            logger.warning(f"Could not find matching record to override for frames {start_frame}–{end_frame}")
        return patched

    # ── Lookup ────────────────────────────────────────────────────────────────

    def find(
        self,
        start_frame: int,
        end_frame: int,
        start_ms: int,
        end_ms: int,
    ) -> tuple[int | None, dict | None]:
        for i, r in enumerate(self.read_all()):
            if (
                _int(r, "start_frame") == start_frame
                and _int(r, "end_frame") == end_frame
                and _int(r, "start_ms") == start_ms
                and _int(r, "end_ms") == end_ms
            ):
                return i, r
        return None, None


def _int(record: dict, key: str, default: int = -1) -> int:
    try:
        return int(record.get(key, default))
    except (ValueError, TypeError):
        return default
