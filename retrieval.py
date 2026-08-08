"""
Retrieval logic (Week 3): given a query embedding, find the most similar
document chunks stored in SQLite using cosine similarity.

For the small document collections this project targets, brute-force
comparison in Python is fast enough (per the plan's guidance — a dedicated
vector database is only needed at much larger scale).
"""

import math

import db
import config


def cosine_similarity(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    return dot / (norm_a * norm_b) if norm_a and norm_b else 0.0


def get_top_chunks(query_embedding, top_k: int = None):
    """
    Return the top_k most relevant chunks for a query embedding.
    Each result is a dict: {id, source, content, score}.
    """
    top_k = top_k or config.TOP_K
    chunks = db.get_all_chunks()

    scored = []
    for chunk in chunks:
        score = cosine_similarity(query_embedding, chunk["embedding"])
        scored.append({**chunk, "score": score})

    scored.sort(key=lambda c: c["score"], reverse=True)
    return scored[:top_k]
