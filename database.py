import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path

from config import get_db_path

DB_PATH = get_db_path()


def _get_connection() -> sqlite3.Connection:
    """Create a new database connection."""
    conn = sqlite3.connect(DB_PATH, timeout=10.0)
    conn.row_factory = sqlite3.Row
    return conn


def init_database():
    """Initialize database schema. Call on app startup."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = _get_connection()
    try:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS file_hashes (
                hash TEXT PRIMARY KEY,
                file_id TEXT NOT NULL,
                file_size INTEGER NOT NULL,
                created_at REAL NOT NULL,
                filename TEXT
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_file_id ON file_hashes(file_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_created_at ON file_hashes(created_at)")
        conn.commit()
    finally:
        conn.close()


@contextmanager
def db_transaction():
    """Context manager for database transactions with IMMEDIATE locking."""
    conn = _get_connection()
    try:
        conn.execute("BEGIN IMMEDIATE")
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def find_by_hash(file_hash: str) -> dict | None:
    """Look up file_id by content hash. Returns dict with file info or None."""
    conn = _get_connection()
    try:
        cursor = conn.execute(
            "SELECT file_id, file_size, created_at, filename FROM file_hashes WHERE hash = ?",
            (file_hash,)
        )
        row = cursor.fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def insert_hash_record(conn: sqlite3.Connection, file_hash: str, file_id: str, file_size: int, filename: str) -> None:
    """Insert a new hash record within a transaction."""
    conn.execute(
        "INSERT INTO file_hashes (hash, file_id, file_size, created_at, filename) VALUES (?, ?, ?, ?, ?)",
        (file_hash, file_id, file_size, time.time(), filename)
    )