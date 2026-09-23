"""
Lexical retrieval shared by the app (retrieval.py) and the benchmark
(rag_eval/), so the benchmark measures exactly the code the app runs.

BM25 with optional prefix truncation ("F5 stemming"): cutting every token
to its first 5 characters is a standard, language-agnostic approximation
for agglutinative languages like Turkish, where "savunması" and "savunma"
should match (Can et al., 2008). On XQuAD-TR it lifts BM25 hit@1 by
+7.6 points and leaves English unchanged (see README).
"""

import re

import numpy as np

_TOKEN = re.compile(r"\w+", re.UNICODE)


def tokenize(text: str) -> list[str]:
    # Turkish-aware lowercasing: "I" -> "ı" and "İ" -> "i" (str.lower gets these wrong).
    text = text.replace("I", "ı").replace("İ", "i").lower()
    return _TOKEN.findall(text)


class BM25Retriever:
    """Okapi BM25 over a fixed list of chunks. prefix_len=None disables stemming."""

    def __init__(self, chunk_ids, chunk_texts, prefix_len: int | None = None):
        from rank_bm25 import BM25Okapi

        self.ids = list(chunk_ids)
        self.prefix_len = prefix_len
        self.name = "bm25" if prefix_len is None else f"bm25-p{prefix_len}"
        self.bm25 = BM25Okapi([self._tok(t) for t in chunk_texts])

    def _tok(self, text: str) -> list[str]:
        toks = tokenize(text)
        return toks if self.prefix_len is None else [t[: self.prefix_len] for t in toks]

    def rank(self, query: str, k: int) -> list:
        scores = self.bm25.get_scores(self._tok(query))
        return [self.ids[j] for j in np.argsort(-scores)[:k]]

    def search_batch(self, queries: list[str], k: int) -> list[list]:
        return [self.rank(q, k) for q in queries]


def rrf(rankings: list[list], k: int, c: int = 60) -> list:
    """Reciprocal rank fusion: score(d) = sum over rankings of 1 / (c + rank(d))."""
    scores: dict = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (c + rank)
    return sorted(scores, key=scores.get, reverse=True)[:k]
