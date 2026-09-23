"""
Information-retrieval metrics.

Every function takes, per query, the ranked list of retrieved chunk ids and
the set of chunk ids that are relevant (contain the gold answer span).
"""

from statistics import mean


def hit_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    """1.0 if any relevant chunk is in the top-k, else 0.0.

    For RAG this is the metric that matters most: if the answer-bearing chunk
    is anywhere in the context window, the generator *can* answer.
    """
    return float(any(doc_id in relevant for doc_id in ranked[:k]))


def reciprocal_rank(ranked: list[str], relevant: set[str], k: int = 10) -> float:
    """1/rank of the first relevant chunk within the top-k (0 if none)."""
    for rank, doc_id in enumerate(ranked[:k], start=1):
        if doc_id in relevant:
            return 1.0 / rank
    return 0.0


def aggregate(runs: list[tuple[list[str], set[str]]], ks=(1, 3, 5, 10)) -> dict:
    """Average metrics over (ranked, relevant) pairs."""
    if not runs:
        raise ValueError("no queries to evaluate")
    out = {f"hit@{k}": mean(hit_at_k(r, rel, k) for r, rel in runs) for k in ks}
    out["mrr@10"] = mean(reciprocal_rank(r, rel, 10) for r, rel in runs)
    out["n_queries"] = len(runs)
    return out
