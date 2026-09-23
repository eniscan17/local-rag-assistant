"""
Retrieval logic: given a question, find the most relevant document chunks.

Two modes (config.RETRIEVAL_MODE):
  - "dense":  cosine similarity between the query and chunk embeddings.
  - "hybrid": dense ranking fused with a Turkish-aware BM25 ranking via
              reciprocal rank fusion. On the XQuAD EN/TR benchmark
              (rag_eval/, see README) this is the best setup measured.

For the small document collections this project targets, brute-force
comparison in Python is fast enough — a dedicated vector database is only
needed at much larger scale.
"""

import math

import config
import db
from lexical import BM25Retriever, rrf

_bm25_cache = {"key": None, "retriever": None}


def cosine_similarity(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    return dot / (norm_a * norm_b) if norm_a and norm_b else 0.0


def _bm25_for(chunks):
    """Build the BM25 index once per knowledge-base state, not once per question."""
    key = tuple(c["id"] for c in chunks)
    if _bm25_cache["key"] != key:
        _bm25_cache["retriever"] = BM25Retriever(
            [c["id"] for c in chunks], [c["content"] for c in chunks],
            prefix_len=config.BM25_PREFIX_LEN,
        )
        _bm25_cache["key"] = key
    return _bm25_cache["retriever"]


def search(question: str, query_embedding, top_k: int = None):
    """
    Return (top_chunks, best_score).

    top_chunks: the top_k chunks as dicts {id, source, content, score}, in
    retrieval order; "score" is always the chunk's cosine similarity.
    best_score: the highest cosine similarity over the whole knowledge base —
    the guardrail signal, independent of how the chunks were ranked.
    """
    top_k = top_k or config.TOP_K
    chunks = db.get_all_chunks()
    if not chunks:
        return [], 0.0

    scored = [{**c, "score": cosine_similarity(query_embedding, c["embedding"])} for c in chunks]
    dense = sorted(scored, key=lambda c: c["score"], reverse=True)
    best_score = dense[0]["score"]

    if config.RETRIEVAL_MODE == "hybrid" and question:
        by_id = {c["id"]: c for c in scored}
        depth = len(chunks)
        fused = rrf([[c["id"] for c in dense], _bm25_for(chunks).rank(question, depth)], k=top_k)
        return [by_id[i] for i in fused], best_score

    return dense[:top_k], best_score


def get_top_chunks(query_embedding, top_k: int = None):
    """Dense-only retrieval (kept for backwards compatibility)."""
    top_k = top_k or config.TOP_K
    scored = [{**c, "score": cosine_similarity(query_embedding, c["embedding"])}
              for c in db.get_all_chunks()]
    scored.sort(key=lambda c: c["score"], reverse=True)
    return scored[:top_k]
