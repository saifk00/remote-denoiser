import os
import sqlite3
import tempfile
import threading
from pathlib import Path

import pytest

from database import init_database, find_by_hash, insert_hash_record, db_transaction, DB_PATH


@pytest.fixture(autouse=True)
def clean_test_db():
    """Remove test database before and after each test."""
    if DB_PATH.exists():
        DB_PATH.unlink()
    init_database()
    yield
    if DB_PATH.exists():
        DB_PATH.unlink()


def test_init_database():
    """Test that database schema is created correctly."""
    assert DB_PATH.exists()

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='file_hashes'")
    assert cursor.fetchone() is not None

    cursor = conn.execute("PRAGMA table_info(file_hashes)")
    columns = {row[1] for row in cursor.fetchall()}
    assert columns == {"hash", "file_id", "file_size", "created_at", "filename"}

    conn.close()


def test_insert_and_find():
    """Test basic insert and retrieval."""
    file_hash = "abc123" * 10 + "abcd"
    file_id = "test-file-id"
    file_size = 1024
    filename = "test.dng"

    with db_transaction() as conn:
        insert_hash_record(conn, file_hash, file_id, file_size, filename)

    result = find_by_hash(file_hash)
    assert result is not None
    assert result["file_id"] == file_id
    assert result["file_size"] == file_size
    assert result["filename"] == filename
    assert result["created_at"] > 0


def test_find_nonexistent_hash():
    """Test that finding a nonexistent hash returns None."""
    result = find_by_hash("nonexistent_hash")
    assert result is None


def test_duplicate_hash_rejection():
    """Test that duplicate hashes are rejected by PRIMARY KEY constraint."""
    file_hash = "duplicate_hash_test" + "x" * 44

    with db_transaction() as conn:
        insert_hash_record(conn, file_hash, "file-1", 100, "test1.dng")

    with pytest.raises(sqlite3.IntegrityError):
        with db_transaction() as conn:
            insert_hash_record(conn, file_hash, "file-2", 200, "test2.dng")

    result = find_by_hash(file_hash)
    assert result["file_id"] == "file-1"


def test_concurrent_inserts():
    """Test thread safety with concurrent identical hash insertions."""
    file_hash = "concurrent_test_hash" + "y" * 42
    results = []
    errors = []

    def insert_worker(worker_id: int):
        try:
            with db_transaction() as conn:
                insert_hash_record(conn, file_hash, f"worker-{worker_id}", 500, f"worker{worker_id}.dng")
            results.append(worker_id)
        except sqlite3.IntegrityError:
            errors.append(worker_id)

    threads = [threading.Thread(target=insert_worker, args=(i,)) for i in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(results) == 1
    assert len(errors) == 9

    final_result = find_by_hash(file_hash)
    assert final_result is not None
    assert final_result["file_id"] in [f"worker-{i}" for i in results]
