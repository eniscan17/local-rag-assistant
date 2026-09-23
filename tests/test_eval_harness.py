"""
Unit tests for eval_harness.py's own logic (guardrail decisions, hit@1/hit@k,
keyword scoring, aggregation, report writing) — independent of Foundry
Local. These mock out llm.embed_query / llm.answer_stream, so they run
anywhere Python + pytest are available, with no models and no GPU/NPU
needed. They do NOT test whether the real embedding/chat models are any
good — only that the harness scores and reports correctly given whatever
they return. Run the real eval_harness.py against Foundry Local for that.

Run with:
    pytest tests/test_eval_harness.py -v
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import eval_harness  # noqa: E402

# Two orthogonal 3-D fake chunks so cosine similarity is easy to reason about:
# identical vector -> score 1.0, orthogonal vector -> score 0.0.
CHUNK_A = {"id": 1, "source": "doc_a.txt", "content": "Doc A content about widgets.", "embedding": [1.0, 0.0, 0.0]}
CHUNK_B = {"id": 2, "source": "doc_b.txt", "content": "Doc B content about gadgets.", "embedding": [0.0, 1.0, 0.0]}


@pytest.fixture(autouse=True)
def patch_db(monkeypatch):
    monkeypatch.setattr(eval_harness.db, "get_all_chunks", lambda: [CHUNK_A, CHUNK_B])


def test_in_scope_correct_when_retrieval_and_keywords_match(monkeypatch):
    monkeypatch.setattr(eval_harness.llm, "embed_query", lambda q: [1.0, 0.0, 0.0])
    monkeypatch.setattr(eval_harness.llm, "answer_stream", lambda sp, q: iter(["widgets are great"]))
    case = {
        "id": "t1", "category": "in_scope", "question": "What are widgets?",
        "expected_source": "doc_a.txt", "expected_keywords": ["widgets"],
    }
    result = eval_harness.run_case(case)
    assert result.hit_at_1 is True
    assert result.hit_at_k is True
    assert result.keyword_hit is True
    assert result.guardrail_fired is False
    assert result.correct is True


def test_out_of_scope_correctly_refused(monkeypatch):
    monkeypatch.setattr(eval_harness.llm, "embed_query", lambda q: [0.0, 0.0, 1.0])  # orthogonal to every chunk
    monkeypatch.setattr(eval_harness.llm, "answer_stream", lambda sp, q: iter(["should not be called"]))
    case = {"id": "t2", "category": "out_of_scope", "question": "Unrelated question?"}
    result = eval_harness.run_case(case)
    assert result.guardrail_fired is True
    assert result.correct is True


def test_false_refusal_on_in_scope_question_is_flagged(monkeypatch):
    monkeypatch.setattr(eval_harness.llm, "embed_query", lambda q: [0.0, 0.0, 1.0])  # orthogonal -> guardrail fires
    monkeypatch.setattr(eval_harness.llm, "answer_stream", lambda sp, q: iter(["irrelevant"]))
    case = {
        "id": "t3", "category": "in_scope", "question": "What are widgets?",
        "expected_source": "doc_a.txt", "expected_keywords": ["widgets"],
    }
    result = eval_harness.run_case(case)
    assert result.guardrail_fired is True
    assert result.correct is False
    assert "guardrail" in result.notes


def test_wrong_document_retrieved_is_flagged(monkeypatch):
    # Only doc_a in the "knowledge base" this time, so the expected source
    # (doc_a) is genuinely absent from what gets retrieved.
    monkeypatch.setattr(eval_harness.db, "get_all_chunks", lambda: [CHUNK_B])
    monkeypatch.setattr(eval_harness.llm, "embed_query", lambda q: [0.0, 1.0, 0.0])
    monkeypatch.setattr(eval_harness.llm, "answer_stream", lambda sp, q: iter(["gadgets are neat"]))
    case = {
        "id": "t4", "category": "in_scope", "question": "What are widgets?",
        "expected_source": "doc_a.txt", "expected_keywords": ["widgets"],
    }
    result = eval_harness.run_case(case)
    assert result.hit_at_1 is False
    assert result.hit_at_k is False
    assert result.correct is False
    assert "not retrieved" in result.notes


def test_summarize_and_report_writing(tmp_path, monkeypatch):
    monkeypatch.setattr(eval_harness.llm, "embed_query", lambda q: [1.0, 0.0, 0.0])
    monkeypatch.setattr(eval_harness.llm, "answer_stream", lambda sp, q: iter(["widgets"]))
    case = {
        "id": "t1", "category": "in_scope", "question": "What are widgets?",
        "expected_source": "doc_a.txt", "expected_keywords": ["widgets"],
    }
    results = [eval_harness.run_case(case)]
    summary = eval_harness.summarize(results)

    assert summary["total_cases"] == 1
    assert summary["retrieval_hit_at_1_pct"] == 100.0

    csv_path = tmp_path / "results.csv"
    md_path = tmp_path / "report.md"
    eval_harness.write_csv(results, csv_path)
    eval_harness.write_markdown_report(results, summary, md_path)

    assert csv_path.exists() and csv_path.stat().st_size > 0
    report_text = md_path.read_text(encoding="utf-8")
    assert "Headline metrics" in report_text
