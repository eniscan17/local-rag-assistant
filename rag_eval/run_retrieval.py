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
    p.add_argument("--limit", type=int, default=None, help="evaluate only the first N questions")
    p.add_argument("--name", default="retrieval", help="output file stem in results/")
    args = p.parse_args(argv)

    rows, stats = [], {}
    for lang in args.langs:
        ds = load_xquad(lang, max_chars=args.max_chars, limit_queries=args.limit)
        stats[lang] = dataset_stats(ds)
        ids = [c.id for c in ds.chunks]
        questions = [q.question for q in ds.queries]
        gold = [q.relevant for q in ds.queries]

        lexical = {}
        for prefix in (None, args.bm25_prefix):
            t = time.time()
            retriever = BM25Retriever(ids, [c.text for c in ds.chunks], prefix_len=prefix)
            lexical[retriever.name] = retriever.search_batch(questions, CANDIDATES)
            rows.append({"lang": lang, "embedder": "-", "retriever": retriever.name,
                         **aggregate(list(zip(lexical[retriever.name], gold))),
                         "seconds": round(time.time() - t, 1)})

        for preset in args.embedders:
            state = {}
            t = time.time()
            doc_emb = _cached_embed(preset, PRESETS[preset], "docs", [c.text for c in ds.chunks],
                                    lang, args.max_chars, state)
            q_emb = _cached_embed(preset, PRESETS[preset], "queries", questions,
                                  lang, args.max_chars, state)
            dense_runs = DenseRetriever(ids, doc_emb).search_batch(q_emb, CANDIDATES)
            secs = round(time.time() - t, 1)
            rows.append({"lang": lang, "embedder": preset, "retriever": "dense",
                         **aggregate(list(zip(dense_runs, gold))), "seconds": secs})
            for lex_name, lex_runs in lexical.items():
                hybrid_runs = [rrf([d, b], k=CANDIDATES) for d, b in zip(dense_runs, lex_runs)]
                rows.append({"lang": lang, "embedder": preset, "retriever": f"hybrid({lex_name})",
                             **aggregate(list(zip(hybrid_runs, gold))), "seconds": secs})
            print(f"[{lang}] {preset} done", file=sys.stderr)

    out = {"config": vars(args), "dataset": stats, "results": rows}
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    with open(os.path.join(ROOT, "results", f"{args.name}.json"), "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    md = to_markdown(out)
    with open(os.path.join(ROOT, "results", f"{args.name}.md"), "w") as f:
        f.write(md)
    print(md)


def to_markdown(out) -> str:
    cfg = out["config"]
    lines = [f"# Retrieval results — XQuAD, chunk size {cfg['max_chars']} chars", ""]
    for lang, s in out["dataset"].items():
        lines.append(f"- **{lang}**: {s['queries']} questions, {s['chunks']} chunks, "
                     f"{s['answers_split_across_chunks']} answers split across a chunk boundary")
    lines += ["", "| lang | embedder | retriever | hit@1 | hit@3 | hit@5 | hit@10 | MRR@10 |",
              "|---|---|---|---|---|---|---|---|"]
    for r in out["results"]:
        lines.append(f"| {r['lang']} | {r['embedder']} | {r['retriever']} | {r['hit@1']:.3f} | "
                     f"{r['hit@3']:.3f} | {r['hit@5']:.3f} | {r['hit@10']:.3f} | {r['mrr@10']:.3f} |")
    lines.append("")
    lines.append("hit@3 is the app's setting (TOP_K = 3).")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
