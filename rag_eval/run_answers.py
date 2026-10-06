"""
Answer-quality benchmark on XQuAD EN/TR: do the retrieval gains reach the answers?

    python -m rag_eval.run_answers --generator foundry:phi-3.5-mini --sample 300

Every sampled question is answered by the chat model under several conditions,
so the effect of each pipeline stage can be separated:

    closed                 no context: what the model already knows (floor)
    oracle                 only the gold chunk(s): perfect retrieval (ceiling)
    <retrieval spec>       top-k chunks from a real retriever, e.g.
                           e5-small/hybrid(bm25-p5)  (the app today)
                           qwen3-0.6b/dense          (the app before the benchmark)
                           bm25-p5                   (no embedding model at all)

The *same* question ids are used in English and Turkish (XQuAD is parallel),
so the EN/TR gap is measured with a paired bootstrap, like everything else.

Every answer is cached (rag_eval/data/answers/), so an interrupted run
resumes where it stopped and re-running only the report is instant.
Writes results/<name>.json and results/<name>.md.
"""

import argparse
import hashlib
import json
import os
import random
import re
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from rag_eval import stats  # noqa: E402
from rag_eval.answer_metrics import REFUSAL, score  # noqa: E402
from rag_eval.datasets import CACHE_DIR, load_xquad  # noqa: E402

APP_RETRIEVER = "e5-small/hybrid(bm25-p5)"
CANDIDATES = 100

_LANG_RULE = "Answer in the same language as the question."
PROMPTS = {
    # Extractive: makes EM/F1 meaningful (SQuAD-style scoring).
    "short": (
        "Answer the question using ONLY the context below. Reply with the shortest "
        "phrase from the context that answers it (a name, number, date or short "
        "phrase), not a full sentence. " + _LANG_RULE + " If the context does not "
        f"contain the answer, reply exactly: {REFUSAL}\n\nContext:\n{{context}}"
    ),
    "short-closed": (
        "Answer the question from your own knowledge. Reply with only the shortest "
        "phrase that answers it (a name, number, date or short phrase), not a full "
        f"sentence. {_LANG_RULE} If you do not know, reply exactly: {REFUSAL}"
    ),
}
PRIMARY = {"short": "f1", "app": "contains"}


def system_prompt(prompt: str, condition: str, context: str) -> str:
    if condition == "closed":
        return PROMPTS["short-closed"]
    if prompt == "app":
        import config
        return config.SYSTEM_PROMPT_TEMPLATE.format(context=context)
    return PROMPTS["short"].format(context=context)


# --- question sampling & retrieval ---------------------------------------------

def sample_ids(all_ids: list[str], n: int | None, seed: int) -> list[str]:
    """Deterministic sample of question ids, identical for every language."""
    ids = sorted(all_ids)
    if n is None or n >= len(ids):
        return ids
    return sorted(random.Random(seed).sample(ids, n))


_SPEC = re.compile(r"^(?:(?P<preset>[\w.-]+)/(?P<mode>dense|hybrid\((?P<lex>bm25(?:-p\d+)?)\))"
                   r"|(?P<bm25>bm25(?:-p\d+)?))$")


def parse_spec(spec: str) -> dict:
    m = _SPEC.match(spec)
    if not m:
        raise ValueError(f"bad retrieval spec {spec!r}; e.g. bm25-p5, e5-small/dense, "
                         f"e5-small/hybrid(bm25-p5)")
    if m["bm25"]:
        return {"preset": None, "mode": "bm25", "lex": m["bm25"]}
    mode = "dense" if m["mode"] == "dense" else "hybrid"
    return {"preset": m["preset"], "mode": mode, "lex": m["lex"]}


def _bm25(ds, lex: str):
    from rag_eval.retrievers import BM25Retriever

    prefix = int(lex.split("-p")[1]) if "-p" in lex else None
    return BM25Retriever([c.id for c in ds.chunks], [c.text for c in ds.chunks], prefix_len=prefix)


def retrieve(spec: str, ds, queries, max_chars: int, k: int) -> list[list[str]]:
    p = parse_spec(spec)
    questions = [q.question for q in queries]
    if p["mode"] == "bm25":
        return _bm25(ds, p["lex"]).search_batch(questions, k)

    from rag_eval.embedders import PRESETS
    from rag_eval.retrievers import DenseRetriever, rrf
    from rag_eval.run_retrieval import _cached_embed

    state = {}
    factory = PRESETS[p["preset"]]
    doc_emb = _cached_embed(p["preset"], factory, "docs", [c.text for c in ds.chunks],
                            ds.lang, max_chars, state)
    q_emb = _cached_embed(p["preset"], factory, "queries", questions, ds.lang, max_chars, state)
    dense = DenseRetriever([c.id for c in ds.chunks], doc_emb).search_batch(q_emb, CANDIDATES)
    if p["mode"] == "dense":
        return [r[:k] for r in dense]
    lexical = _bm25(ds, p["lex"]).search_batch(questions, CANDIDATES)
    return [rrf([d, b], k=k) for d, b in zip(dense, lexical)]


def contexts_for(condition: str, ds, queries, max_chars: int, k: int):
    """Per question: (context text, retrieval hit or None)."""
    text = {c.id: c.text for c in ds.chunks}
    order = {c.id: i for i, c in enumerate(ds.chunks)}
    if condition == "closed":
        return [("", None) for _ in queries]
    if condition == "oracle":
        return [("\n\n".join(text[c] for c in sorted(q.relevant, key=order.get)), None)
                for q in queries]
    runs = retrieve(condition, ds, queries, max_chars, k)
    return [("\n\n".join(text[c] for c in ranked), any(c in q.relevant for c in ranked))
            for q, ranked in zip(queries, runs)]


# --- answer cache ----------------------------------------------------------------

class AnswerCache:
    def __init__(self, generator_name: str):
        safe = re.sub(r"[^\w.-]+", "_", generator_name)
        self.path = os.path.join(CACHE_DIR, "answers", f"{safe}.jsonl")
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        self.data = {}
        if os.path.exists(self.path):
            with open(self.path, encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        rec = json.loads(line)
                        self.data[rec["key"]] = rec

    @staticmethod
    def key(system: str, user: str, max_tokens: int) -> str:
        return hashlib.sha1(f"{max_tokens}\x00{system}\x00{user}".encode()).hexdigest()

    def get(self, key):
        return self.data.get(key)

    def put(self, key, pred, seconds):
        rec = {"key": key, "pred": pred, "seconds": seconds}
        self.data[key] = rec
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


# --- aggregation -------------------------------------------------------------------

METRICS = ("em", "f1", "contains", "refused")


def summarize(records: list[dict]) -> list[dict]:
    rows = []
    for (lang, cond) in dict.fromkeys((r["lang"], r["condition"]) for r in records):
        rs = [r for r in records if r["lang"] == lang and r["condition"] == cond]
        row = {"lang": lang, "condition": cond, "n": len(rs)}
        for m in METRICS:
            vals = [r[m] for r in rs]
            row[m] = sum(vals) / len(vals)
            if m != "refused":
                row[f"{m}_ci"] = list(stats.ci(vals))
        hits = [r["retrieval_hit"] for r in rs if r["retrieval_hit"] is not None]
        row["retrieval_hit"] = sum(hits) / len(hits) if hits else None
        secs = sorted(r["seconds"] for r in rs if r["seconds"] is not None)
        row["median_seconds"] = secs[len(secs) // 2] if secs else None
        rows.append(row)
    return rows


def _paired(records, metric, sel_a, sel_b):
    a = {r["id"]: r[metric] for r in records if sel_a(r)}
    b = {r["id"]: r[metric] for r in records if sel_b(r)}
    ids = sorted(a.keys() & b.keys())
    if not ids:
        return None
    return stats.paired_diff([a[i] for i in ids], [b[i] for i in ids])


def comparisons(records, conditions, metric) -> list[dict]:
    out = []
    langs = list(dict.fromkeys(r["lang"] for r in records))
    retrieved = [c for c in conditions if c not in ("closed", "oracle")]
    for lang in langs:
        for cond in retrieved:
            for ref in ("oracle", "closed"):
                if ref in conditions:
                    d = _paired(records, metric,
                                lambda r, c=cond: r["lang"] == lang and r["condition"] == c,
                                lambda r, c=ref: r["lang"] == lang and r["condition"] == c)
                    out.append({"what": f"{lang}: {cond} − {ref}", **d})
        for i, c1 in enumerate(retrieved):
            for c2 in retrieved[i + 1:]:
                d = _paired(records, metric,
                            lambda r, c=c1: r["lang"] == lang and r["condition"] == c,
                            lambda r, c=c2: r["lang"] == lang and r["condition"] == c)
                out.append({"what": f"{lang}: {c1} − {c2}", **d})
    if "en" in langs and "tr" in langs:
        for cond in conditions:
            d = _paired(records, metric,
                        lambda r, c=cond: r["lang"] == "tr" and r["condition"] == c,
                        lambda r, c=cond: r["lang"] == "en" and r["condition"] == c)
            if d:
                out.append({"what": f"TR − EN: {cond}", **d})
    return out


def attribution(records) -> list[dict]:
    """Where do wrong answers come from? Correct = gold answer contained in the answer."""
    out = []
    for (lang, cond) in dict.fromkeys((r["lang"], r["condition"]) for r in records):
        rs = [r for r in records if r["lang"] == lang and r["condition"] == cond
              and r["retrieval_hit"] is not None]
        if not rs:
            continue
        n = len(rs)
        cell = lambda hit, ok: sum(1 for r in rs if r["retrieval_hit"] == hit and bool(r["contains"]) == ok) / n  # noqa: E731
        wrong = 1 - sum(r["contains"] for r in rs) / n
        out.append({"lang": lang, "condition": cond, "n": n, "wrong": wrong,
                    "retrieval_miss_wrong": cell(False, False),
                    "retrieval_hit_wrong": cell(True, False),
                    "retrieval_miss_right": cell(False, True)})
    return out


# --- report -------------------------------------------------------------------------

def _p(x):
    return "–" if x is None else f"{100 * x:.1f}"


def _ci(row, m):
    lo, hi = row[f"{m}_ci"]
    return f"{_p(row[m])} [{_p(lo)}–{_p(hi)}]"


def to_markdown(out) -> str:
    cfg, metric = out["config"], out["primary_metric"]
    lines = [f"# Answer quality — XQuAD, `{out['generator']}`, prompt `{cfg['prompt']}`", "",
             f"{out['n_questions']} questions per language (seed {cfg['seed']}, same ids in EN and TR), "
             f"top-{cfg['top_k']} chunks of {cfg['max_chars']} chars. All numbers in %; brackets: "
             f"95% bootstrap CI. Primary metric: **{metric}**.", "",
             "| lang | condition | EM | F1 | contains | refused | retrieval hit@k | median s |",
             "|---|---|---|---|---|---|---|---|"]
    for r in out["summary"]:
        secs = "–" if r["median_seconds"] is None else f"{r['median_seconds']:.2f}"
        lines.append(f"| {r['lang']} | {r['condition']} | {_ci(r, 'em')} | {_ci(r, 'f1')} | "
                     f"{_ci(r, 'contains')} | {_p(r['refused'])} | {_p(r['retrieval_hit'])} | {secs} |")
    lines += ["", f"## Paired differences ({metric}, points)", "",
              "✓ = 95% paired-bootstrap CI excludes 0.", "", "| comparison | Δ |", "|---|---|"]
    for c in out["comparisons"]:
        mark = " ✓" if c["significant"] else ""
        lines.append(f"| {c['what']} | {100 * c['diff']:+.1f} "
                     f"[{100 * c['ci_low']:+.1f}, {100 * c['ci_high']:+.1f}]{mark} |")
    if out["attribution"]:
        lines += ["", "## Where wrong answers come from", "",
                  "Share of all questions; correct = gold answer contained in the model's answer.", "",
                  "| lang | retriever | wrong | …retrieval missed | …retrieved, model still wrong | "
                  "missed but right anyway |", "|---|---|---|---|---|---|"]
        for a in out["attribution"]:
            lines.append(f"| {a['lang']} | {a['condition']} | {_p(a['wrong'])} | "
                         f"{_p(a['retrieval_miss_wrong'])} | {_p(a['retrieval_hit_wrong'])} | "
                         f"{_p(a['retrieval_miss_right'])} |")
    ex = out.get("examples") or []
    if ex:
        lines += ["", "## Examples: gold chunk was retrieved, answer still wrong", "",
                  "| lang | question | gold | model |", "|---|---|---|---|"]
        for e in ex:
            cell = lambda s: s.replace("|", "\\|").replace("\n", " ")[:120]  # noqa: E731
            lines.append(f"| {e['lang']} | {cell(e['question'])} | {cell(e['gold'])} | {cell(e['pred'])} |")
    return "\n".join(lines) + "\n"


def examples(records, n_per_lang=4):
    out = []
    for lang in dict.fromkeys(r["lang"] for r in records):
        bad = [r for r in records if r["lang"] == lang and r["retrieval_hit"] and not r["contains"]]
        out += [{"lang": lang, "question": r["question"], "gold": r["gold"][0], "pred": r["pred"]}
                for r in bad[:n_per_lang]]
    return out


# --- main ----------------------------------------------------------------------------

def run(args, generator=None) -> dict:
    max_tokens = args.max_tokens or (32 if args.prompt == "short" else 200)
    cache = None
    records, n_questions = [], None
    for lang in args.langs:
        ds = load_xquad(lang, max_chars=args.max_chars)
        keep = set(sample_ids([q.id for q in ds.queries], args.sample, args.seed))
        queries = [q for q in ds.queries if q.id in keep]
        n_questions = len(queries)
        for cond in args.conditions:
            ctxs = contexts_for(cond, ds, queries, args.max_chars, args.top_k)
            t0, done = time.time(), 0
            for q, (context, hit) in zip(queries, ctxs):
                system = system_prompt(args.prompt, cond, context)
                if generator is None:
                    from rag_eval.generators import make_generator
                    generator = make_generator(args.generator, max_tokens)
                if cache is None:
                    cache = AnswerCache(generator.name)
                key = AnswerCache.key(system, q.question, max_tokens)
                rec = cache.get(key)
                if rec is None:
                    t = time.time()
                    pred = generator.generate(system, q.question)
                    rec = {"pred": pred, "seconds": round(time.time() - t, 3)}
                    cache.put(key, pred, rec["seconds"])
                    done += 1
                    if done % 10 == 0:
                        rate = (time.time() - t0) / done
                        print(f"[{lang}/{cond}] {done} new answers, ~{rate:.1f}s each", file=sys.stderr)
                records.append({"id": q.id, "lang": lang, "condition": cond, "question": q.question,
                                "gold": q.answers, "pred": rec["pred"], "seconds": rec["seconds"],
                                "retrieval_hit": hit, **score(rec["pred"], q.answers, lang)})
            print(f"[{lang}] {cond} done", file=sys.stderr)

    metric = PRIMARY[args.prompt]
    return {"config": vars(args), "generator": generator.name if generator else args.generator,
            "n_questions": n_questions, "primary_metric": metric,
            "summary": summarize(records),
            "comparisons": comparisons(records, args.conditions, metric),
            "attribution": attribution(records), "examples": examples(records),
            "records": records}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--generator", default="foundry:phi-3.5-mini",
                   help="foundry:<alias> (Mac, the app's runtime) or hf:<model id>")
    p.add_argument("--langs", nargs="+", default=["en", "tr"])
    p.add_argument("--conditions", nargs="+",
                   default=["closed", "oracle", APP_RETRIEVER, "qwen3-0.6b/dense"],
                   help="closed, oracle and/or retrieval specs")
    p.add_argument("--prompt", choices=sorted(PRIMARY), default="short",
                   help="short = extractive span (EM/F1); app = the app's own 2-4 sentence prompt")
    p.add_argument("--sample", type=int, default=300, help="questions per language (0 = all 1190)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--top-k", type=int, default=3, help="chunks given to the model (app: 3)")
    p.add_argument("--max-chars", type=int, default=800)
    p.add_argument("--max-tokens", type=int, default=None, help="default: 32 (short) / 200 (app)")
    p.add_argument("--name", default="answers")
    args = p.parse_args(argv)
    args.sample = args.sample or None
    for c in args.conditions:
        if c not in ("closed", "oracle"):
            parse_spec(c)  # fail fast on typos, before any model loads

    out = run(args)
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    with open(os.path.join(ROOT, "results", f"{args.name}.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    md = to_markdown(out)
    with open(os.path.join(ROOT, "results", f"{args.name}.md"), "w", encoding="utf-8") as f:
        f.write(md)
    print(md)


if __name__ == "__main__":
    main()
