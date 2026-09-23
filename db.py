"""
SQLite storage layer for document chunks and their embeddings.

Schema (Week 2-3 of the plan):
    chunks(id, source, content, embedding_json)

Embeddings are stored as a JSON-encoded list of floats (simple & portable
for a small, single-machine project; a real vector DB would use a native
vector column instead).
"""

import json
import os
import sqlite3
from contextlib import contextmanager

import config


def _ensure_data_dir():
    os.makedirs(config.DATA_DIR, exist_ok=True)


@contextmanager
def get_connection():
    _ensure_data_dir()
    conn = sqlite3.connect(config.DB_PATH)
    try:
        yield conn
    finally:
        conn.close()


def init_db():
    """Create the chunks table if it doesn't exist yet."""
    with get_connection() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source TEXT NOT NULL,
                content TEXT NOT NULL,
                embedding_json TEXT NOT NULL
            )
            """
        )
        conn.execute("CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT)")
        conn.commit()


def set_meta(key: str, value: str):
    with get_connection() as conn:
        conn.execute("INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)", (key, value))
        conn.commit()


def get_meta(key: str):
    """Return a stored metadata value, or None (also for indexes built before meta existed)."""
    with get_connection() as conn:
        conn.execute("CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT)")
        row = conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
    return row[0] if row else None


def clear_chunks():
    """Wipe all chunks — used before a fresh ingestion run."""
    with get_connection() as conn:
        conn.execute("DELETE FROM chunks")
        conn.commit()


def insert_chunk(source: str, content: str, embedding: list[float]):
    with get_connection() as conn:
        conn.execute(
            "INSERT INTO chunks (source, content, embedding_json) VALUES (?, ?, ?)",
            (source, content, json.dumps(embedding)),
        )
        conn.commit()


def insert_chunks(rows: list[tuple[str, str, list[float]]]):
    """Bulk insert (source, content, embedding) tuples."""
    with get_connection() as conn:
        conn.executemany(
            "INSERT INTO chunks (source, content, embedding_json) VALUES (?, ?, ?)",
            [(source, content, json.dumps(embedding)) for source, content, embedding in rows],
        )
        conn.commit()


def get_all_chunks():
    """Return every chunk as a list of dicts: id, source, content, embedding."""
    with get_connection() as conn:
        cur = conn.execute("SELECT id, source, content, embedding_json FROM chunks")
        rows = cur.fetchall()
    return [
        {
            "id": r[0],
            "source": r[1],
            "content": r[2],
            "embedding": json.loads(r[3]),
        }
        for r in rows
    ]


def count_chunks() -> int:
    with get_connection() as conn:
        cur = conn.execute("SELECT COUNT(*) FROM chunks")
        return cur.fetchone()[0]
