"""Unit tests for the evaluation harness (no models or network needed).

    python -m pytest tests/
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag_eval.datasets import chunk_with_offsets  # noqa: E402
from rag_eval.metrics import aggregate, hit_at_k, reciprocal_rank  # noqa: E402
from rag_eval.retrievers import BM25Retriever, DenseRetriever, rrf, tokenize  # noqa: E402


# --- metrics ----------------------------------------------------------------

def test_hit_at_k():
    ranked = ["a", "b", "c"]
    assert hit_at_k(ranked, {"b"}, 1) == 0.0
    assert hit_at_k(ranked, {"b"}, 2) == 1.0
    assert hit_at_k(ranked, {"z"}, 3) == 0.0


def test_reciprocal_rank():
    assert reciprocal_rank(["a", "b", "c"], {"c"}) == pytest.approx(1 / 3)
    assert reciprocal_rank(["a", "b", "c"], {"a", "c"}) == 1.0  # first relevant counts
    assert reciprocal_rank(["a", "b"], {"z"}) == 0.0
    assert reciprocal_rank(["a", "b", "c"], {"c"}, k=2) == 0.0  # outside cutoff


def test_aggregate_averages():
    out = aggregate([(["a", "b"], {"a"}), (["a", "b"], {"b"})], ks=(1,))
    assert out["hit@1"] == 0.5
    assert out["mrr@10"] == pytest.approx(0.75)
    assert out["n_queries"] == 2


def test_aggregate_rejects_empty():
    with pytest.raises(ValueError):
        aggregate([])


# --- chunk offsets (ground-truth mapping depends on these) --------------------

def test_chunk_offsets_match_source():
    text = "x" * 50 + " answer here " + "y" * 60
    for chunk, start, end in chunk_with_offsets(text, max_chars=40):
        assert text[start:end] == chunk


def test_chunk_offsets_with_paragraphs():
    text = "First para.\n\nSecond para.\n\n" + "z" * 30
    for chunk, start, end in chunk_with_offsets(text, max_chars=28):
        assert text[start:end] == chunk


# --- retrievers ---------------------------------------------------------------

def test_turkish_lowercasing():
    assert tokenize("IŞIK İstanbul") == ["ışık", "istanbul"]


def test_bm25_prefix_matches_turkish_suffixes():
    ids = ["def", "other"]
    texts = ["Panthers savunma hattı çok güçlüydü", "Bir başka konu hakkında metin"]
    plain = BM25Retriever(ids, texts)
    stemmed = BM25Retriever(ids, texts, prefix_len=5)
    q = ["savunması nasıldı"]
    # Plain BM25 has no exact token overlap; prefix stemming finds the match.
    assert stemmed.search_batch(q, 1) == [["def"]]
    assert plain.bm25.get_scores(plain._tok(q[0])).max() == 0.0


def test_dense_retriever_orders_by_cosine():
    emb = np.array([[1, 0], [0, 1], [0.7, 0.7]], dtype=np.float32)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    r = DenseRetriever(["x", "y", "xy"], emb)
    assert r.search_batch(np.array([[1.0, 0.0]]), 2) == [["x", "xy"]]


def test_rrf_rewards_agreement():
    fused = rrf([["a", "b", "c"], ["b", "a", "d"]], k=4)
    assert set(fused[:2]) == {"a", "b"}
    assert fused.index("c") > 1 and fused.index("d") > 1


# --- statistics ----------------------------------------------------------------

from rag_eval import stats  # noqa: E402
from rag_eval.run_retrieval import first_ranks  # noqa: E402


def test_first_ranks():
    runs = [["a", "b"], ["c", "d"], ["x", "y"]]
    gold = [{"b"}, {"c"}, {"z"}]
    assert first_ranks(runs, gold) == [2, 1, 0]
    assert list(stats.hits([2, 1, 0], 1)) == [0.0, 1.0, 0.0]
    assert list(stats.hits([2, 1, 0], 3)) == [1.0, 1.0, 0.0]


def test_ci_contains_mean_and_is_narrow_for_constant():
    lo, hi = stats.ci([1.0] * 50)
    assert lo == hi == 1.0
    lo, hi = stats.ci([0, 1] * 500)
    assert lo < 0.5 < hi and hi - lo < 0.1


def test_paired_diff_detects_real_gap_only():
    a = np.array([1.0] * 900 + [0.0] * 100)
    b = np.array([1.0] * 700 + [0.0] * 300)
    assert stats.paired_diff(a, b)["significant"]
    assert not stats.paired_diff(a, a)["significant"]
