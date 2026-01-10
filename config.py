import os
from pathlib import Path

# Read from environment, default to "data" for production
DATA_DIR = Path(os.environ.get("DATA_DIR", "data"))


def get_data_dir() -> Path:
    """Get the configured data directory path."""
    return DATA_DIR


def get_db_path() -> Path:
    """Get the database file path within the data directory."""
    return DATA_DIR / "dedup.db"
