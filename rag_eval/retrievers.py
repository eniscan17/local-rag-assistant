"""Retrievers for the benchmark. Lexical parts live in lexical.py, shared with the app."""

import numpy as np

from lexical import BM25Retriever, rrf, tokenize  # noqa: F401  (re-exported)


class DenseRetriever:
    name = "dense"

    def __init__(self, chunk_ids, chunk_emb: np.ndarray):
        self.ids = list(chunk_ids)
        self.emb = chunk_emb

    def search_batch(self, query_emb: np.ndarray, k: int) -> list[list[str]]:
        scores = query_emb @ self.emb.T
        top = np.argsort(-scores, axis=1)[:, :k]
        return [[self.ids[j] for j in row] for row in top]
