"""
Automated evaluation harness for the Local RAG Assistant.

Replaces manually running each row of tests/test_queries.md by hand: it
drives the project's own retrieval + generation pipeline (retrieval.py,
db.py, llm.py, config.py — unmodified, imported directly) against a fixed
test set (tests/eval_set.json) and reports:

  - Retrieval accuracy: does the correct source document come back in the
    top-1 / top-K retrieved chunks for each in-scope question?
  - Guardrail behavior: for out-of-scope questions (no matching document),
    does MIN_RELEVANCE_SCORE correctly stop the app from calling the model
    at all? For in-scope questions, does the guardrail ever *wrongly*
    refuse a question it should have answered?
  - Answer quality proxy: for in-scope questions the model did answer, does
    the answer contain at least one of a few hand-picked keywords? This is
    a rough, no-LLM-judge proxy — not a substitute for reading answers, but
    catches regressions automatically.
  - Latency: embedding / retrieval / generation timings (mean, median, p95).

Usage (run from the project root, with the venv active and `python
ingest.py` already run at least once so the knowledge base has data):

    python eval_harness.py
    python eval_harness.py --set tests/eval_set.json --out tests/eval_results
    python eval_harness.py --limit 5          # quick smoke test
    python eval_harness.py --case fl-1 rag-2  # run only specific case ids

Needs no dependencies beyond the project's own requirements.txt — no LLM
judge, no extra packages. Writes a CSV (raw per-case rows) and a Markdown
report (summary tables) into --out, and prints the summary to the console.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

import config  # noqa: E402  (project module, must be importable from BASE_DIR)
import db  # noqa: E402
import llm  # noqa: E402
import retrieval  # noqa: E402

REFUSAL_MARKERS = ("don't have", "do not have", "don't know", "do not know", "not sure", "no information")


@dataclass
class CaseResult:
    id: str
    category: str
    question: str
    expected_source: str | None
    retrieved_sources: list[str]
    top_score: float
    guardrail_fired: bool
    answer: str
    embed_ms: float
    retrieve_ms: float
    generate_ms: float
    hit_at_1: bool | None = None
    hit_at_k: bool | None = None
    keyword_hit: bool | None = None
    correct: bool = False
    notes: str = ""


def load_eval_set(path: Path, case_ids: list[str] | None, limit: int | None):
    data = json.loads(path.read_text(encoding="utf-8"))
    cases = data["cases"]
    if case_ids:
        wanted = set(case_ids)
        cases = [c for c in cases if c["id"] in wanted]
    if limit:
        cases = cases[:limit]
    return cases


def looks_like_refusal(answer: str) -> bool:
    lower = answer.lower()
    return any(marker in lower for marker in REFUSAL_MARKERS)


def run_case(case: dict) -> CaseResult:
    question = case["question"]
    category = case["category"]
    expected_source = case.get("expected_source")
    expected_keywords = [k.lower() for k in case.get("expected_keywords", [])]

    t0 = time.perf_counter()
    query_embedding = llm.embed_query(question)
    t1 = time.perf_counter()

    top_chunks = retrieval.get_top_chunks(query_embedding)
    t2 = time.perf_counter()

    retrieved_sources = [c["source"] for c in top_chunks]
    top_score = top_chunks[0]["score"] if top_chunks else 0.0
    guardrail_fired = top_score < config.MIN_RELEVANCE_SCORE

    generate_ms = 0.0
    if guardrail_fired:
        answer = "I don't have that information."
    else:
        context = "\n\n".join(c["content"] for c in top_chunks)
        system_prompt = config.SYSTEM_PROMPT_TEMPLATE.format(context=context)
        t_gen_start = time.perf_counter()
        answer = "".join(llm.answer_stream(system_prompt, question))
        generate_ms = (time.perf_counter() - t_gen_start) * 1000

    result = CaseResult(
        id=case["id"],
        category=category,
        question=question,
        expected_source=expected_source,
        retrieved_sources=retrieved_sources,
        top_score=round(top_score, 4),
        guardrail_fired=guardrail_fired,
        answer=answer,
        embed_ms=round((t1 - t0) * 1000, 2),
        retrieve_ms=round((t2 - t1) * 1000, 2),
        generate_ms=round(generate_ms, 2),
    )

    if category == "in_scope":
        result.hit_at_1 = bool(retrieved_sources) and retrieved_sources[0] == expected_source
        result.hit_at_k = expected_source in retrieved_sources
        if guardrail_fired:
            result.keyword_hit = False
            result.correct = False
            result.notes = "guardrail wrongly refused an in-scope question"
        else:
            result.keyword_hit = any(k in answer.lower() for k in expected_keywords) if expected_keywords else None
            if not result.hit_at_k:
                result.correct = False
                result.notes = "expected document not retrieved in top-k"
            elif result.keyword_hit is False:
                result.correct = False
                result.notes = "answer did not contain any expected keyword"
            else:
                result.correct = True
    else:  # out_of_scope
        refused = guardrail_fired or looks_like_refusal(answer)
        result.correct = refused
        if not refused:
            result.notes = "model answered an out-of-scope question instead of refusing"

    return result


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    k = (len(s) - 1) * p
    f, c = int(k), min(int(k) + 1, len(s) - 1)
    if f == c:
        return s[f]
    return s[f] + (s[c] - s[f]) * (k - f)


def summarize(results: list[CaseResult]) -> dict:
    in_scope = [r for r in results if r.category == "in_scope"]
    out_scope = [r for r in results if r.category == "out_of_scope"]

    def rate(items, pred):
        return round(100 * sum(1 for i in items if pred(i)) / len(items), 1) if items else None

    per_doc: dict[str, list[CaseResult]] = {}
    for r in in_scope:
        per_doc.setdefault(r.expected_source, []).append(r)
    per_doc_hit1 = {doc: rate(rs, lambda r: r.hit_at_1) for doc, rs in per_doc.items()}

    all_embed = [r.embed_ms for r in results]
    all_retrieve = [r.retrieve_ms for r in results]
    all_generate = [r.generate_ms for r in results if r.generate_ms > 0]
    all_total = [r.embed_ms + r.retrieve_ms + r.generate_ms for r in results]

    def lat_stats(values):
        if not values:
            return {"mean": 0.0, "median": 0.0, "p95": 0.0}
        return {
            "mean": round(statistics.mean(values), 1),
            "median": round(statistics.median(values), 1),
            "p95": round(percentile(values, 0.95), 1),
        }

    return {
        "total_cases": len(results),
        "in_scope_count": len(in_scope),
        "out_of_scope_count": len(out_scope),
        "retrieval_hit_at_1_pct": rate(in_scope, lambda r: r.hit_at_1),
        "retrieval_hit_at_k_pct": rate(in_scope, lambda r: r.hit_at_k),
        "false_refusal_pct": rate(in_scope, lambda r: r.guardrail_fired),
        "keyword_match_pct": rate(
            [r for r in in_scope if not r.guardrail_fired], lambda r: r.keyword_hit
        ),
        "guardrail_refusal_rate_out_of_scope_pct": rate(out_scope, lambda r: r.correct),
        "per_doc_hit_at_1_pct": per_doc_hit1,
        "latency_ms": {
            "embed": lat_stats(all_embed),
            "retrieve": lat_stats(all_retrieve),
            "generate": lat_stats(all_generate),
            "total": lat_stats(all_total),
        },
    }


def write_csv(results: list[CaseResult], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "id", "category", "question", "expected_source", "retrieved_sources",
            "top_score", "guardrail_fired", "hit_at_1", "hit_at_k", "keyword_hit",
            "correct", "embed_ms", "retrieve_ms", "generate_ms", "answer", "notes",
        ])
        for r in results:
            writer.writerow([
                r.id, r.category, r.question, r.expected_source or "",
                "|".join(r.retrieved_sources), r.top_score, r.guardrail_fired,
                r.hit_at_1, r.hit_at_k, r.keyword_hit, r.correct,
                r.embed_ms, r.retrieve_ms, r.generate_ms, r.answer.replace("\n", " "), r.notes,
            ])


def fmt_pct(value) -> str:
    return "n/a" if value is None else f"{value}%"


def write_markdown_report(results: list[CaseResult], summary: dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    lines.append("# Automated Evaluation Report\n")
    lines.append(f"Generated by `eval_harness.py` — {summary['total_cases']} cases "
                 f"({summary['in_scope_count']} in-scope, {summary['out_of_scope_count']} out-of-scope).\n")

    lines.append("## Headline metrics\n")
    lines.append("| Metric | Value |")
    lines.append("|---|---|")
    lines.append(f"| Retrieval hit@1 (correct doc ranked first) | {fmt_pct(summary['retrieval_hit_at_1_pct'])} |")
    lines.append(f"| Retrieval hit@{config.TOP_K} (correct doc in top-{config.TOP_K}) | {fmt_pct(summary['retrieval_hit_at_k_pct'])} |")
    lines.append(f"| False refusals (in-scope Qs wrongly refused) | {fmt_pct(summary['false_refusal_pct'])} |")
    lines.append(f"| Keyword match on answered in-scope Qs | {fmt_pct(summary['keyword_match_pct'])} |")
    lines.append(f"| Guardrail correctly refused out-of-scope Qs | {fmt_pct(summary['guardrail_refusal_rate_out_of_scope_pct'])} |")
    lines.append("")

    lines.append("## Retrieval accuracy by document\n")
    lines.append("| Document | Hit@1 |")
    lines.append("|---|---|")
    for doc, pct in summary["per_doc_hit_at_1_pct"].items():
        lines.append(f"| {doc} | {fmt_pct(pct)} |")
    lines.append("")

    lines.append("## Latency (ms)\n")
    lines.append("| Stage | Mean | Median | P95 |")
    lines.append("|---|---|---|---|")
    for stage, stats in summary["latency_ms"].items():
        lines.append(f"| {stage} | {stats['mean']} | {stats['median']} | {stats['p95']} |")
    lines.append("")

    failures = [r for r in results if not r.correct]
    if failures:
        lines.append(f"## Failed cases ({len(failures)})\n")
        lines.append("| id | category | question | notes |")
        lines.append("|---|---|---|---|")
        for r in failures:
            lines.append(f"| {r.id} | {r.category} | {r.question} | {r.notes or '—'} |")
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def print_console_summary(summary: dict):
    print("\n=== Evaluation summary ===")
    print(f"Cases run: {summary['total_cases']} "
          f"({summary['in_scope_count']} in-scope, {summary['out_of_scope_count']} out-of-scope)")
    print(f"Retrieval hit@1:  {fmt_pct(summary['retrieval_hit_at_1_pct'])}")
    print(f"Retrieval hit@{config.TOP_K}:  {fmt_pct(summary['retrieval_hit_at_k_pct'])}")
    print(f"False refusals (in-scope): {fmt_pct(summary['false_refusal_pct'])}")
    print(f"Keyword match (answered in-scope): {fmt_pct(summary['keyword_match_pct'])}")
    print(f"Guardrail refusal rate (out-of-scope): {fmt_pct(summary['guardrail_refusal_rate_out_of_scope_pct'])}")
    lat = summary["latency_ms"]["total"]
    print(f"Total latency per question — mean {lat['mean']}ms, median {lat['median']}ms, p95 {lat['p95']}ms")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--set", default="tests/eval_set.json", help="Path to the eval set JSON file.")
    parser.add_argument("--out", default="tests/eval_results", help="Directory to write results into.")
    parser.add_argument("--limit", type=int, default=None, help="Only run the first N cases (quick smoke test).")
    parser.add_argument("--case", nargs="+", default=None, help="Only run these specific case ids.")
    args = parser.parse_args()

    set_path = BASE_DIR / args.set
    out_dir = BASE_DIR / args.out

    if db.count_chunks() == 0:
        print("No chunks indexed yet — run `python ingest.py` first.", file=sys.stderr)
        sys.exit(1)

    cases = load_eval_set(set_path, args.case, args.limit)
    print(f"Loaded {len(cases)} case(s) from {set_path}")

    print("Initializing Foundry Local models (first run downloads them — may take a while)...")
    llm.initialize()
    try:
        results = []
        for i, case in enumerate(cases, 1):
            print(f"[{i}/{len(cases)}] {case['id']}: {case['question'][:60]}")
            results.append(run_case(case))
    finally:
        llm.shutdown()

    summary = summarize(results)

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    csv_path = out_dir / f"results_{timestamp}.csv"
    report_path = out_dir / f"report_{timestamp}.md"
    write_csv(results, csv_path)
    write_markdown_report(results, summary, report_path)

    print_console_summary(summary)
    print(f"\nRaw results:  {csv_path.relative_to(BASE_DIR)}")
    print(f"Report:       {report_path.relative_to(BASE_DIR)}")


if __name__ == "__main__":
    main()
