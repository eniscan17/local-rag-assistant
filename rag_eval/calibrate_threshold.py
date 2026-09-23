"""
Calibrate the guardrail threshold (config.MIN_RELEVANCE_SCORE).

The guardrail refuses to call the chat model when the best cosine
similarity between the question and any chunk is below a threshold. The
right value depends on the embedding model (e5 similarities live in a much
higher, narrower range than Qwen3's), so it has to be re-derived whenever
the model changes.

This script embeds every question in tests/eval_set.json with the app's
current configuration, computes the guardrail score, and suggests the
threshold that best separates in-scope from out-of-scope questions.

    python ingest.py                          # index built with the current config
    python -m rag_eval.calibrate_threshold
"""

import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def suggest_threshold(in_scores: list[float], out_scores: list[float]) -> dict:
    """
    Pick the threshold with the fewest mistakes (in-scope refused + out-of-scope
    let through). Ties are broken toward the widest margin. If the two groups
    are perfectly separable the result is the midpoint of the gap.
    """
    if not in_scores or not out_scores:
        raise ValueError("need both in-scope and out-of-scope scores")
    points = sorted(set(in_scores) | set(out_scores))
    candidates = [points[0] - 1e-6] + [(a + b) / 2 for a, b in zip(points, points[1:])] + [points[-1] + 1e-6]

    def errors(t):
        false_refusals = sum(s < t for s in in_scores)
        false_accepts = sum(s >= t for s in out_scores)
        return false_refusals, false_accepts

    def margin(t):
        return min(abs(s - t) for s in in_scores + out_scores)

    best = min(candidates, key=lambda t: (sum(errors(t)), -margin(t)))
    fr, fa = errors(best)
    return {
        "threshold": round(best, 4),
        "false_refusals": fr,
        "false_accepts": fa,
        "separable": fr == 0 and fa == 0,
        "min_in_scope": min(in_scores),
        "max_out_of_scope": max(out_scores),
    }


def main():
    import config
    import db
    import llm
    import retrieval

    cases = json.load(open(os.path.join(ROOT, "tests", "eval_set.json"), encoding="utf-8"))["cases"]
    llm.initialize()
    indexed_with = db.get_meta("embedder")
    if indexed_with != llm.embedder_id():
        sys.exit(f"\nIndex built with {indexed_with or 'an older version'}, config uses "
                 f"{llm.embedder_id()}. Run `python ingest.py` first.")

    scores = {"in_scope": [], "out_of_scope": []}
    print(f"\n\nEmbedder: {llm.embedder_id()}\n")
    print(f"{'score':>6}  {'category':<13} question")
    rows = []
    for case in cases:
        _, best = retrieval.search(case["question"], llm.embed_query(case["question"]))
        scores[case["category"]].append(best)
        rows.append((best, case["category"], case["question"]))
    for best, cat, q in sorted(rows, reverse=True):
        print(f"{best:6.3f}  {cat:<13} {q[:70]}")

    s = suggest_threshold(scores["in_scope"], scores["out_of_scope"])
    print(f"\nLowest in-scope score:      {s['min_in_scope']:.3f}")
    print(f"Highest out-of-scope score: {s['max_out_of_scope']:.3f}")
    print(f"Suggested threshold:        {s['threshold']:.3f}  "
          f"({s['false_refusals']} false refusals, {s['false_accepts']} false accepts)")
    if not s["separable"]:
        print("Note: the groups overlap, so no threshold is perfect on this set. "
              "The chat model's own 'I don't know' is then the second line of defence.")
    print(f"\nCurrent config value for '{config.EMBEDDING_BACKEND}': {config.MIN_RELEVANCE_SCORE}")
    print(f"To apply: set MIN_RELEVANCE_SCORES['{config.EMBEDDING_BACKEND}'] = "
          f"{s['threshold']:.2f} in config.py")
    llm.shutdown()


if __name__ == "__main__":
    main()
