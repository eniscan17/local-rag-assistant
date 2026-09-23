"""
Data ingestion pipeline (Week 3):
    1. Read every .txt / .md file in documents/
    2. Split each into paragraph-sized chunks
    3. Embed all chunks in one batch call
    4. Store (source, content, embedding) rows in SQLite

Run directly:
    python ingest.py
"""

import glob
import os

import config
import db
import llm
from chunking import chunk_text as _chunk_text


def _read_documents():
    """Return a list of (source_filename, full_text) for every doc file."""
    patterns = ["*.txt", "*.md"]
    paths = []
    for pattern in patterns:
        paths.extend(glob.glob(os.path.join(config.DOCUMENTS_DIR, pattern)))

    docs = []
    for path in sorted(paths):
        with open(path, "r", encoding="utf-8") as f:
            docs.append((os.path.basename(path), f.read()))
    return docs


def run_ingestion(progress_callback=None):
    """
    Full pipeline: read documents/, chunk, embed, store in SQLite.
    Returns the number of chunks indexed.
    """
    docs = _read_documents()
    if not docs:
        raise RuntimeError(
            f"No .txt or .md files found in {config.DOCUMENTS_DIR}. "
            "Add some documents there first."
        )

    all_chunks = []          # list of (source, content)
    for source, text in docs:
        for chunk_text in _chunk_text(text):
            all_chunks.append((source, chunk_text))

    if not all_chunks:
        raise RuntimeError("Documents were found but produced no text chunks.")

    llm.initialize(progress_callback=progress_callback)

    texts = [c[1] for c in all_chunks]
    embeddings = llm.embed_texts(texts)

    db.init_db()
    db.clear_chunks()
    rows = [(source, content, emb) for (source, content), emb in zip(all_chunks, embeddings)]
    db.insert_chunks(rows)

    return len(rows)


if __name__ == "__main__":
    db.init_db()
    n = run_ingestion()
    print(f"\nIndexed {n} chunks from {config.DOCUMENTS_DIR} into {config.DB_PATH}")
