"""Retrievers: dense (cosine), BM25 (lexical) and hybrid (reciprocal rank fusion)."""

import re

import numpy as np

_TOKEN = re.compile(r"\w+", re.UNICODE)


def tokenize(text: str) -> list[str]:
    # Turkish-aware lowercasing: "I" -> "ı" and "İ" -> "i" (str.lower gets these wrong).
    text = text.replace("I", "ı").replace("İ", "i").lower()
    return _TOKEN.findall(text)


class DenseRetriever:
    name = "dense"

    def __init__(self, chunk_ids, chunk_emb: np.ndarray):
        self.ids = list(chunk_ids)
        self.emb = chunk_emb

    def search_batch(self, query_emb: np.ndarray, k: int) -> list[list[str]]:
        scores = query_emb @ self.emb.T
        top = np.argsort(-scores, axis=1)[:, :k]
        return [[self.ids[j] for j in row] for row in top]


class BM25Retriever:
    """Okapi BM25.

    prefix_len truncates every token to its first N characters — "F5
    stemming", a standard, language-agnostic approximation for agglutinative
    languages like Turkish (Can et al., 2008), where "savunması" and
    "savunma" should match. None disables it.
    """

    def __init__(self, chunk_ids, chunk_texts, prefix_len: int | None = None):
        from rank_bm25 import BM25Okapi

        self.ids = list(chunk_ids)
        self.prefix_len = prefix_len
        self.name = "bm25" if prefix_len is None else f"bm25-p{prefix_len}"
        self.bm25 = BM25Okapi([self._tok(t) for t in chunk_texts])

    def _tok(self, text: str) -> list[str]:
        toks = tokenize(text)
        return toks if self.prefix_len is None else [t[: self.prefix_len] for t in toks]

    def search_batch(self, queries: list[str], k: int) -> list[list[str]]:
        out = []
        for q in queries:
            scores = self.bm25.get_scores(self._tok(q))
            top = np.argsort(-scores)[:k]
            out.append([self.ids[j] for j in top])
        return out


def rrf(rankings: list[list[str]], k: int, c: int = 60) -> list[str]:
    """Reciprocal rank fusion: score(d) = sum 1 / (c + rank_i(d))."""
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (c + rank)
    return sorted(scores, key=scores.get, reverse=True)[:k]
