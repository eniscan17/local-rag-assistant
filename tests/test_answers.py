"""Tests for the answer-quality benchmark (no models or network needed).

    python -m pytest tests/
"""

import argparse
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag_eval import run_answers  # noqa: E402
from rag_eval.answer_metrics import contains, exact_match, f1, is_refusal, normalize  # noqa: E402
from rag_eval.datasets import Chunk, Dataset, Query  # noqa: E402


# --- metrics ------------------------------------------------------------------

def test_normalize_en_drops_articles_and_punctuation():
    assert normalize("The Eiffel Tower!", "en") == "eiffel tower"


def test_normalize_tr_lowercases_dotted_and_dotless_i():
    assert normalize("İSTANBUL", "tr") == "istanbul"
    assert normalize("IRMAK", "tr") == "ırmak"
    assert normalize("IRMAK", "en") == "irmak"


def test_normalize_tr_drops_apostrophe_suffixes():
    assert normalize("Paris'te", "tr") == "paris"
    assert normalize("1891’de", "tr") == "1891"
    assert exact_match("Paris'te", ["Paris"], "tr") == 1.0
    assert exact_match("Paris'te", ["Paris"], "en") == 0.0  # English keeps the old behaviour


def test_exact_match_takes_max_over_golds():
    assert exact_match("Denver Broncos", ["Broncos", "Denver Broncos"]) == 1.0
    assert exact_match("Broncos!", ["the Broncos"]) == 1.0


def test_f1_partial_overlap():
    assert f1("Denver Broncos team", ["Denver Broncos"]) == pytest.approx(0.8)
    assert f1("nothing", ["Denver Broncos"]) == 0.0
    assert f1("", [""]) == 1.0


def test_contains_respects_token_boundaries():
    assert contains("They won in 2010.", ["2010"]) == 1.0
    assert contains("They won in 2010.", ["10"]) == 0.0
    assert contains("Super Bowl'u Denver Broncos kazandı.", ["Denver Broncos"], "tr") == 1.0


def test_refusal_detection():
    assert is_refusal("unanswerable")
    assert is_refusal("Bilmiyorum.")
    assert is_refusal("")
    assert not is_refusal("Denver Broncos")


# --- specs & sampling -----------------------------------------------------------

def test_parse_spec():
    assert run_answers.parse_spec("bm25-p5") == {"preset": None, "mode": "bm25", "lex": "bm25-p5"}
    assert run_answers.parse_spec("e5-small/dense")["mode"] == "dense"
    assert run_answers.parse_spec("e5-small/hybrid(bm25-p5)") == \
        {"preset": "e5-small", "mode": "hybrid", "lex": "bm25-p5"}
    with pytest.raises(ValueError):
        run_answers.parse_spec("e5-small/hybrid")


def test_sample_is_deterministic_and_order_independent():
    ids = [f"q{i}" for i in range(50)]
    a = run_answers.sample_ids(ids, 10, seed=0)
    assert a == run_answers.sample_ids(list(reversed(ids)), 10, seed=0)  # EN/TR get the same ids
    assert len(a) == 10 and a != run_answers.sample_ids(ids, 10, seed=1)
    assert run_answers.sample_ids(ids, None, 0) == sorted(ids)


# --- end to end with a fake model ----------------------------------------------------

def _toy_dataset(lang):
    chunks = [Chunk("p0-c0", "Denver Broncos won Super Bowl 50 in 2016."),
              Chunk("p1-c0", "The Eiffel Tower is in Paris and opened in 1889."),
              Chunk("p2-c0", "Mount Everest is the highest mountain on Earth.")]
    queries = [Query("q1", "Who won Super Bowl 50?", ["Denver Broncos"], {"p0-c0"}, "p0"),
               Query("q2", "When did the Eiffel Tower open?", ["1889"], {"p1-c0"}, "p1"),
               Query("q3", "What is the highest mountain on Earth?", ["Mount Everest"], {"p2-c0"}, "p2")]
    return Dataset(lang, chunks, queries, {})


class ReaderStub:
    """Answers correctly iff the answer is in its prompt, else refuses — a perfect reader."""
    name = "stub"
    answers = {"Who won Super Bowl 50?": "Denver Broncos",
               "When did the Eiffel Tower open?": "1889",
               "What is the highest mountain on Earth?": "Mount Everest"}

    def generate(self, system, user):
        a = self.answers[user]
        return a if a in system else "unanswerable"


def test_run_end_to_end(monkeypatch, tmp_path):
    monkeypatch.setattr(run_answers, "load_xquad", lambda lang, max_chars: _toy_dataset(lang))
    monkeypatch.setattr(run_answers, "CACHE_DIR", str(tmp_path))
    args = argparse.Namespace(generator="stub", langs=["en", "tr"],
                              conditions=["closed", "oracle", "bm25-p5"], prompt="short",
                              sample=None, seed=0, top_k=1, max_chars=800, max_tokens=None, name="t")
    out = run_answers.run(args, generator=ReaderStub())
    by = {(r["lang"], r["condition"]): r for r in out["summary"]}

    assert by[("en", "closed")]["em"] == 0.0 and by[("en", "closed")]["refused"] == 1.0
    assert by[("en", "oracle")]["em"] == 1.0
    # A perfect reader's answer accuracy equals retrieval hit@k.
    assert by[("en", "bm25-p5")]["em"] == by[("en", "bm25-p5")]["retrieval_hit"]
    attr = {a["lang"]: a for a in out["attribution"]}["en"]
    assert attr["retrieval_hit_wrong"] == 0.0  # the stub never fails when it has the chunk
    assert any(c["what"] == "TR − EN: oracle" for c in out["comparisons"])
    assert "| en | oracle |" in run_answers.to_markdown(out)

    # Second run is served entirely from the cache.
    class Boom:
        name = "stub"

        def generate(self, *a):
            raise AssertionError("should have been cached")
    run_answers.run(args, generator=Boom())
