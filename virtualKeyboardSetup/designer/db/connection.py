"""
Core Database Manager class.
Manages thread-safe SQLite connection setup, schema initialization, context management, and query execution.
"""

import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Generator

from .schema import CREATE_PROJECTS_TABLE, CREATE_BUTTONS_TABLE, CREATE_MARKERS_TABLE, CREATE_SETTINGS_TABLE


class DatabaseManager:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, db_path: str | Path = "designer_data.db"):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self, db_path: str | Path = "designer_data.db"):
        if getattr(self, "_initialized", False):
            return
        self.db_path = Path(db_path).resolve()
        self._local = threading.local()
        self.init_db()
        self._initialized = True

    def _get_connection(self) -> sqlite3.Connection:
        if not hasattr(self._local, "connection") or self._local.connection is None:
            conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA foreign_keys = ON;")
            self._local.connection = conn
        return self._local.connection

    @contextmanager
    def get_cursor(self) -> Generator[sqlite3.Cursor, None, None]:
        conn = self._get_connection()
        cursor = conn.cursor()
        try:
            yield cursor
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            cursor.close()

    def init_db(self) -> None:
        """Create database tables if they do not exist."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self.get_cursor() as cursor:
            cursor.execute(CREATE_PROJECTS_TABLE)
            cursor.execute(CREATE_BUTTONS_TABLE)
            cursor.execute(CREATE_MARKERS_TABLE)
            cursor.execute(CREATE_SETTINGS_TABLE)

    def close(self) -> None:
        """Close current thread connection."""
        if hasattr(self._local, "connection") and self._local.connection is not None:
            self._local.connection.close()
            self._local.connection = None
