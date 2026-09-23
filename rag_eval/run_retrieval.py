"""
Retrieval evaluation CLI.

    python -m rag_eval.run_retrieval --langs en tr --embedders qwen3-0.6b e5-small

For each (language, embedder) it evaluates dense, BM25 and hybrid (RRF)
retrieval on XQuAD and writes results/<name>.json and results/<name>.md.
Embeddings are cached under rag_eval/data/cache/ so re-runs are fast.
"""

import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from rag_eval.datasets import CACHE_DIR, load_xquad  # noqa: E402
from rag_eval.embedders import PRESETS  # noqa: E402
from rag_eval.metrics import aggregate  # noqa: E402
from rag_eval import stats  # noqa: E402
from rag_eval.retrievers import BM25Retriever, DenseRetriever, rrf  # noqa: E402

CANDIDATES = 100  # depth of each ranking fed into hybrid fusion


def _cached_embed(preset, embedder_factory, kind, texts, lang, max_chars, state):
    key = hashlib.sha1("\n".join(texts).encode()).hexdigest()[:10]
    path = os.path.join(CACHE_DIR, "cache", f"{preset}_{lang}_{max_chars}_{kind}_{key}.npy")
    if os.path.exists(path):
        return np.load(path)
    if "model" not in state:
        state["model"] = embedder_factory()
    model = state["model"]
    emb = model.embed_documents(texts) if kind == "docs" else model.embed_queries(texts)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, emb)
    return emb


def first_ranks(runs, gold) -> list[int]:
    """Rank (1-based) of the first relevant chunk per question; 0 if not retrieved."""
    out = []
    for ranked, rel in zip(runs, gold):
        out.append(next((i for i, d in enumerate(ranked, 1) if d in rel), 0))
    return out


def make_row(lang, embedder, retriever, runs, gold, seconds):
    ranks = first_ranks(runs, gold)
    row = {"lang": lang, "embedder": embedder, "retriever": retriever,
           **aggregate(list(zip(runs, gold))), "seconds": seconds}
    for k in (1, 3):
        lo, hi = stats.ci(stats.hits(ranks, k))
        row[f"hit@{k}_ci"] = [round(lo, 4), round(hi, 4)]
    row["first_ranks"] = ranks
    return row


def dataset_stats(ds):
    straddling = sum(1 for q in ds.queries if len(q.relevant) > 1)
    return {
        "chunks": len(ds.chunks),
        "queries": len(ds.queries),
        "answers_split_across_chunks": straddling,
    }


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--langs", nargs="+", default=["en", "tr"])
    p.add_argument("--embedders", nargs="*", default=["qwen3-0.6b"], choices=sorted(PRESETS))
    p.add_argument("--max-chars", type=int, default=800, help="chunk size (app default: 800)")
    p.add_argument("--bm25-prefix", type=int, default=5,
                   help="token prefix length for the stemmed BM25 variant (F5 stemming)")
    p.add_argument("--baseline", default="qwen3-0.6b/dense",
                   help="embedder/retriever every other config is compared against "
                        "(default: the app's current setup); falls back to -/bm25")
    p.add_argument("--limit", type=int, default=None, help="evaluate only the first N questions")
    p.add_argument("--name", default="retrieval", help="output file stem in results/")
    args = p.parse_args(argv)

    rows, ds_stats = [], {}
    for lang in args.langs:
        ds = load_xquad(lang, max_chars=args.max_chars, limit_queries=args.limit)
        ds_stats[lang] = dataset_stats(ds)
        ids = [c.id for c in ds.chunks]
        questions = [q.question for q in ds.queries]
        gold = [q.relevant for q in ds.queries]

        lexical = {}
        for prefix in (None, args.bm25_prefix):
            t = time.time()
            retriever = BM25Retriever(ids, [c.text for c in ds.chunks], prefix_len=prefix)
            lexical[retriever.name] = retriever.search_batch(questions, CANDIDATES)
            rows.append(make_row(lang, "-", retriever.name, lexical[retriever.name], gold,
                                 round(time.time() - t, 1)))

        for preset in args.embedders:
            state = {}
            t = time.time()
            doc_emb = _cached_embed(preset, PRESETS[preset], "docs", [c.text for c in ds.chunks],
                                    lang, args.max_chars, state)
            q_emb = _cached_embed(preset, PRESETS[preset], "queries", questions,
                                  lang, args.max_chars, state)
            dense_runs = DenseRetriever(ids, doc_emb).search_batch(q_emb, CANDIDATES)
            secs = round(time.time() - t, 1)
            rows.append(make_row(lang, preset, "dense", dense_runs, gold, secs))
            for lex_name, lex_runs in lexical.items():
                hybrid_runs = [rrf([d, b], k=CANDIDATES) for d, b in zip(dense_runs, lex_runs)]
                rows.append(make_row(lang, preset, f"hybrid({lex_name})", hybrid_runs, gold, secs))
            print(f"[{lang}] {preset} done", file=sys.stderr)

    out = {"config": vars(args), "dataset": ds_stats, "results": rows,
           "comparisons": compare_to_baseline(rows, args.baseline)}
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    with open(os.path.join(ROOT, "results", f"{args.name}.json"), "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    md = to_markdown(out)
    with open(os.path.join(ROOT, "results", f"{args.name}.md"), "w") as f:
        f.write(md)
    print(md)


def compare_to_baseline(rows, baseline: str) -> list[dict]:
    """Paired-bootstrap hit@1 / hit@3 differences of every config vs the baseline, per language."""
    out = []
    for lang in dict.fromkeys(r["lang"] for r in rows):
        lang_rows = [r for r in rows if r["lang"] == lang]
        key = lambda r: f"{r['embedder']}/{r['retriever']}"  # noqa: E731
        base = next((r for r in lang_rows if key(r) == baseline), None) \
            or next(r for r in lang_rows if key(r) == "-/bm25")
        for r in lang_rows:
            if r is base:
                continue
            entry = {"lang": lang, "config": key(r), "baseline": key(base)}
            for k in (1, 3):
                entry[f"hit@{k}"] = stats.paired_diff(stats.hits(r["first_ranks"], k),
                                                      stats.hits(base["first_ranks"], k))
            out.append(entry)
    return out


def _pct(x):
    return f"{100 * x:.1f}"


def to_markdown(out) -> str:
    cfg = out["config"]
    lines = [f"# Retrieval results — XQuAD, chunk size {cfg['max_chars']} chars", ""]
    for lang, s in out["dataset"].items():
        lines.append(f"- **{lang}**: {s['queries']} questions, {s['chunks']} chunks, "
                     f"{s['answers_split_across_chunks']} answers split across a chunk boundary")
    lines += ["", "All numbers in %. Brackets: 95% bootstrap confidence interval.", "",
              "| lang | embedder | retriever | hit@1 | hit@3 | hit@10 | MRR@10 |",
              "|---|---|---|---|---|---|---|"]
    for r in out["results"]:
        c1, c3 = r["hit@1_ci"], r["hit@3_ci"]
        lines.append(f"| {r['lang']} | {r['embedder']} | {r['retriever']} | "
                     f"{_pct(r['hit@1'])} [{_pct(c1[0])}–{_pct(c1[1])}] | "
                     f"{_pct(r['hit@3'])} [{_pct(c3[0])}–{_pct(c3[1])}] | "
                     f"{_pct(r['hit@10'])} | {_pct(r['mrr@10'])} |")
    comps = out.get("comparisons") or []
    if comps:
        lines += ["", f"## Paired comparison vs baseline (`{comps[0]['baseline']}`)", "",
                  "Difference in points, 95% paired-bootstrap CI; ✓ = CI excludes 0.", "",
                  "| lang | config | Δ hit@1 | Δ hit@3 |", "|---|---|---|---|"]
        for c in comps:
            cells = []
            for k in (1, 3):
                d = c[f"hit@{k}"]
                mark = " ✓" if d["significant"] else ""
                cells.append(f"{100 * d['diff']:+.1f} [{100 * d['ci_low']:+.1f}, {100 * d['ci_high']:+.1f}]{mark}")
            lines.append(f"| {c['lang']} | {c['config']} | {cells[0]} | {cells[1]} |")
    lines.append("")
    lines.append("hit@3 is the app's setting (TOP_K = 3).")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
