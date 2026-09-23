# Local RAG Assistant (Foundry Local + SQLite + Streamlit)

An offline Q&A chatbot that answers questions about your own documents,
built from the Summer School "Local RAG with Foundry Local" plan. It runs
entirely on your Mac — no cloud account, no API keys, no internet needed
after the one-time model download.

**Architecture:** `documents/*.txt` → chunked & embedded → stored in
`data/knowledge_base.sqlite3` → on a question, the top matching chunks are
retrieved and sent as context to a local chat model via Foundry Local →
answer streams back in the Streamlit UI.

---

## 0. What you need to do (permissions & installs)

Everything in this folder was generated for you. The remaining steps
require your Mac, so you need to run them yourself:

1. **Confirm your Mac is Apple Silicon** (M1/M2/M3/M4). Foundry Local's
   macOS support is Apple Silicon only — check via  → About This Mac.
2. **Install Homebrew** if you don't already have it (https://brew.sh) —
   used to install the Foundry Local runtime itself.
3. **Allow one-time internet access** for: (a) `brew install foundrylocal`
   itself, and (b) the first time you run ingestion or the app, which
   downloads two small models (roughly 400 MB–1.5 GB total). After that,
   everything works offline.
4. **Allow local network / firewall prompts if macOS shows one** the first
   time `foundry service start` runs — the runtime starts a local
   OpenAI-compatible server on `localhost` for the app to talk to. Click
   **Allow**.

Nothing here touches your Apple ID, sends data anywhere, or needs admin/
sudo access for normal use.

---

## 1. Install prerequisites (one time)

Open Terminal in this folder and run each step, checking the output before
moving to the next:

```bash
# Check Python version — need 3.11+
python3 --version

# If you don't have it, install via Homebrew:
brew install python@3.11
```

**Checkpoint:** `python3 --version` should print 3.11 or higher before continuing.

```bash
# Install the Foundry Local runtime itself (separate from the Python package below —
# this is the actual on-device AI engine; the Python SDK just talks to it)
brew tap microsoft/foundrylocal
brew install foundrylocal

# Verify:
foundry --version
```

**Checkpoint:** `foundry --version` should print a version number, not "command not found".

## 2. Create a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

**Checkpoint:** your terminal prompt should now start with `(.venv)`.

## 3. Install dependencies

```bash
pip install -r requirements.txt
```

**Checkpoint:** no red error lines at the end of the install. This pulls in
`foundry-local-sdk`, `openai`, and `streamlit`.

## 4. (Optional) Add your own documents

Three sample `.txt` files are already in `documents/` so you can test the
pipeline immediately. To use your own knowledge base, drop `.txt` or `.md`
files into `documents/` (replace or add to the samples).

## 5. Build the knowledge base

```bash
python ingest.py
```

This downloads the embedding model on first run (may take a few minutes —
status is printed to the terminal), chunks every file in `documents/`,
embeds each chunk, and stores it in `data/knowledge_base.sqlite3`.

**Checkpoint:** last line printed should be `Indexed N chunks from ...`
with N > 0. If N is 0, check that `documents/` actually has `.txt`/`.md`
files.

## 6. Run the app

```bash
streamlit run app.py
```

This opens a browser tab. The chat model downloads on first run (status
shown in the app and the terminal — first load can take a minute or two).
Ask a question — try the sample ones in `tests/test_queries.md`.

**Checkpoint:** you get a streamed answer, and the "Sources used" expander
shows a chunk from one of your documents.

---

## Everyday use after setup

Once installed, only two commands are needed:

```bash
source .venv/bin/activate
streamlit run app.py
```

Re-run `python ingest.py` any time you add/change files in `documents/`
(it wipes and rebuilds the index) or click **Rebuild knowledge base** in
the app's sidebar.

---

## Project structure

```
config.py        model names, paths, system prompt — edit here to tune behavior
db.py             SQLite schema + read/write helpers
ingest.py         chunk documents → embed → store (python ingest.py)
retrieval.py      cosine-similarity search over stored chunks
llm.py            Foundry Local SDK wrapper (model loading, embeddings, streaming chat)
app.py            Streamlit chat UI (streamlit run app.py)
eval_harness.py   automated evaluation runner (python eval_harness.py)
documents/        your source .txt / .md files (sample docs included)
data/             generated SQLite database (created automatically)
tests/test_queries.md      original manual QA log (Week 5 style)
tests/eval_set.json        automated eval question set (32 cases)
tests/test_eval_harness.py pytest suite for eval_harness.py's own logic
tests/eval_results/        generated CSV + Markdown reports (created automatically)
```

## Customizing

- **Change models:** edit `EMBEDDING_MODEL_ALIAS` / `CHAT_MODEL_ALIAS` in
  `config.py`. `phi-3.5-mini` is a good larger/slower chat model if you
  want better answers and don't mind a bit more latency.
- **Change how many chunks are retrieved:** `TOP_K` in `config.py`.
- **Change the system prompt / tone:** `SYSTEM_PROMPT_TEMPLATE` in `config.py`.

## Automated evaluation

`tests/test_queries.md` was the original manual Week 5 test log — useful,
but only as thorough as the 8 questions someone remembers to run by hand.
`eval_harness.py` automates and extends that idea: it drives the real
pipeline (`retrieval.py` + `llm.py`) through a larger, fixed
question set (`tests/eval_set.json` — 24 in-scope questions across all six
sample documents, plus 8 out-of-scope questions that should be refused) and
reports:

- **Retrieval accuracy** — does the correct source document come back in
  the top-1 / top-`TOP_K` retrieved chunks for each question?
- **Guardrail behavior** — for out-of-scope questions, does
  `MIN_RELEVANCE_SCORE` correctly stop the app from calling the model at
  all? For in-scope questions, does the guardrail ever *wrongly* refuse?
- **Answer quality proxy** — for questions the model did answer, does the
  answer contain at least one expected keyword? (A rough, no-LLM-judge
  check — not a substitute for reading answers, but it catches regressions
  automatically on every run.)
- **Latency** — embedding / retrieval / generation timings (mean, median, p95).

Run it (after `python ingest.py` has built the knowledge base):

```bash
source .venv/bin/activate
python eval_harness.py                    # full run, all 32 cases
python eval_harness.py --limit 5          # quick smoke test
python eval_harness.py --case fl-1 rag-2  # run only specific case ids
```

Each run writes a timestamped CSV (raw per-question rows) and a Markdown
report (summary tables + a list of failing cases with why) into
`tests/eval_results/`, and prints a summary to the console.

The harness's own logic — guardrail decisions, hit@1/hit@k scoring, report
generation — is covered by `tests/test_eval_harness.py`, which mocks the
Foundry Local calls so it runs instantly with no models or GPU/NPU needed:

```bash
pip install pytest   # if not already installed
pytest tests/test_eval_harness.py -v
```

## Model comparison (measured with eval_harness.py)

Ran the full 32-case eval set against both chat model options in `config.py`:

| Metric | qwen2.5-0.5b | phi-3.5-mini | phi-3.5-mini (optimized) |
|---|---|---|---|
| Retrieval hit@1 | 95.8% | 95.8% | 95.8% |
| Retrieval hit@3 | 100% | 100% | 100% |
| Guardrail accuracy (out-of-scope refused) | 100% | 100% | 100% |
| Answer keyword-match (in-scope, answered) | 70.8% | 100% | 100% |
| Mean latency | 603ms | 5387ms | 2873ms |
| P95 latency | 1470ms | 13787ms | 5280ms |

"Optimized" = same phi-3.5-mini model, with `CHAT_MAX_TOKENS` capping
output length and a system prompt that asks for a 2-4 sentence answer
(see "Speed follow-up" below). Capping generation length cut mean latency
by ~47% and p95 by ~62%, with zero loss in the keyword-match score.

Retrieval and guardrail scores are identical between runs, as expected —
only the chat model changed, and the harness confirms that isolation
actually holds in this codebase. The remaining gap between the two models
is entirely in answer generation: on the 7 cases qwen2.5-0.5b got wrong
(all retrieval was already correct), it either mis-stated a detail
(answered a "which file" question with a table name) or contradicted its
own retrieved context. phi-3.5-mini made no such errors on this set, at
roughly 9x the latency.

**Decision:** `phi-3.5-mini` is the default. This project's job is to be
shown to people (recruiters, interviewers), not to serve production
traffic — a wrong answer costs more credibility in that setting than a
few extra seconds does. Switch `CHAT_MODEL_ALIAS` back to `qwen2.5-0.5b`
in `config.py` if this is ever deployed somewhere response time is the
priority instead.

**Speed follow-up (confirmed):** phi-3.5-mini's original 5.4s average came
partly from generating longer answers than needed. `CHAT_MAX_TOKENS` in
`config.py` caps how much the model can generate, and the system prompt asks
for a 2-4 sentence answer. Re-running the full eval set after this change
(`tests/eval_results/report_20260921-235212.md`) confirmed mean latency
dropped from 5387ms to 2873ms (-47%) and p95 from 13787ms to 5280ms (-62%),
with keyword-match accuracy unchanged at 100%.

---

## Retrieval benchmark on XQuAD EN/TR (`rag_eval/`)

`eval_harness.py` above scores retrieval hit@3 at 100% — but the sample
knowledge base has only ~10 chunks, where almost any retriever succeeds. `rag_eval/` measures retrieval quantitatively on a
harder, labelled benchmark instead.

**Benchmark:** [XQuAD](https://github.com/google-deepmind/xquad) (CC BY-SA
4.0) — the same 240 Wikipedia paragraphs and 1,190 questions, professionally
translated into English and Turkish. Paragraphs are chunked with the app's
own chunker (`chunking.py`, 800 chars → ~340 chunks); a chunk counts as
relevant if it contains the gold answer span. Because content and questions
are identical across languages, any EN/TR gap is caused by language alone.

**Metrics:** hit@k (is an answer-bearing chunk in the top-k? — the app uses
k = 3) and MRR@10.

```bash
pip install -r requirements-eval.txt
python -m pytest tests/                                   # unit tests, no models needed
python -m rag_eval.run_retrieval --embedders              # BM25 only, ~1 min
python -m rag_eval.run_retrieval --embedders qwen3-0.6b e5-small --name dense_v1
python -m rag_eval.run_retrieval --embedders foundry --name foundry  # the app's own Foundry pipeline (Mac)
```

Results are written to `results/<name>.md` and `.json`.

### Results

Full tables with 95% bootstrap confidence intervals: [`results/retrieval_all.md`](results/retrieval_all.md)
(reproduce with `python -m rag_eval.run_retrieval --embedders qwen3-0.6b qwen3-0.6b-instruct e5-small --name retrieval_all`).

hit@1 / hit@3 in %, a selection:

| setup | EN hit@1 | EN hit@3 | TR hit@1 | TR hit@3 |
|---|---|---|---|---|
| **App today:** qwen3-embedding-0.6b, dense | 86.0 | 96.1 | 74.5 | 88.1 |
| BM25 | 89.1 | 96.4 | 78.6 | 89.0 |
| BM25 + F5 stemming | 88.6 | 96.2 | 86.2 | 94.4 |
| qwen3-0.6b + query instruction, dense | 90.0 | 97.8 | 79.5 | 91.7 |
| multilingual-e5-small, dense | 90.6 | 97.7 | 86.9 | 95.5 |
| **multilingual-e5-small + F5-BM25, hybrid (RRF)** | **92.4** | **98.5** | **89.5** | **97.1** |

**Findings** (all differences below are significant under a paired
bootstrap over questions — 95% CI excludes zero — unless noted):

1. **The app's current retriever is the weakest option on Turkish** —
   74.5% hit@1, below even plain BM25. On identical content, it trails its
   own English score by 11.5 points.
2. **Turkish morphology explains much of the gap.** F5 stemming lifts
   BM25 on Turkish by +7.6 hit@1 [+5.6, +9.7] and does nothing for English.
3. **The query instruction matters.** Qwen3-Embedding expects an
   instruction prefix on queries; the app omits it. Adding it gains +5.0
   hit@1 on Turkish and +4.0 on English with no model change.
4. **Hybrid only helps when the lexical side is language-aware.** On
   Turkish, fusing e5 with plain BM25 does not help (86.9 → 86.1); fusing
   it with F5-stemmed BM25 does (→ 89.5, +2.6 [+1.2, +3.9] over e5 alone).
5. **Best setup:** multilingual-e5-small + F5-BM25 hybrid. Versus the app
   today: Turkish hit@1 **+15.0 points [+12.6, +17.3]**, top-3 misses cut
   from 11.9% to 2.9% (≈4×); English hit@1 +6.5 [+4.5, +8.4]. The EN/TR
   gap shrinks from 11.5 to 2.9 points, with an embedding model 5× smaller
   (118M vs 600M parameters). On English it is statistically tied with
   instructed Qwen3 hybrid (+0.9 [−0.3, +2.1]).

Caveat: the cloud/CI runs use the Hugging Face release of
qwen3-embedding-0.6b; the Foundry Local build the app loads may be
quantised differently (`--embedders foundry` measures it directly).

### Applying it to the app

The app now uses the winning setup by default (`config.py`):
`EMBEDDING_BACKEND = "sentence-transformers"` (multilingual-e5-small) and
`RETRIEVAL_MODE = "hybrid"`. The lexical code lives in `lexical.py` and is
shared by the app and the benchmark, so the benchmark measures exactly what
the app runs. The Foundry embedding path is still available
(`EMBEDDING_BACKEND = "foundry"`) and now sends Qwen3's query instruction.
The index records which embedder built it; the app warns (and
`eval_harness.py` stops) if the configuration changed without re-running
`python ingest.py`.

**Guardrail recalibration.** Cosine scores are not comparable across
embedding models — e5 packs everything into roughly 0.72–0.92 — so the old
0.35 threshold would let every question through. `MIN_RELEVANCE_SCORE` is
now per backend and derived with `python -m rag_eval.calibrate_threshold`,
which scores the 32 questions of `tests/eval_set.json`: lowest in-scope
0.818, highest out-of-scope 0.792 → threshold **0.805**.

`eval_harness.py` before → after (same 32 questions; reports
`tests/eval_results/report_20260921-235212.md` → `report_20260923-185315.md`):

| | before (qwen3 dense) | after (e5 hybrid) |
|---|---|---|
| Retrieval hit@1 | 95.8% | **100%** |
| Retrieval hit@3 | 100% | 100% |
| False refusals (in-scope) | 0% | 0% |
| Out-of-scope correctly refused | 100% | 100% |
| Keyword match (answered in-scope) | 100% | 100% |

**Limitations — read before trusting these numbers:**

- The guardrail threshold was chosen on the same 32 questions it is then
  evaluated on, so the 100% refusal rate is optimistic by construction.
  The margin is also thin (0.026 between the two groups, 8 out-of-scope
  examples); an unseen off-topic question can land above it, in which case
  the chat model's own "I don't know" is the second line of defence. A
  larger held-out out-of-scope set is the obvious next step.
- The small eval set can't separate retrievers (both score 100% hit@3);
  that is what the XQuAD benchmark above is for.
- Latency is not compared: `CHAT_MAX_TOKENS` changed between the runs, so
  any difference can't be attributed to retrieval.

Next: evaluate answer quality on XQuAD (exact match / F1 against the gold
answers) to check that the retrieval gains carry through to the answers.

## Troubleshooting

- **`ModuleNotFoundError: foundry_local_sdk`** — you forgot to activate the
  venv (`source .venv/bin/activate`) or run `pip install -r requirements.txt`.
- **Model download hangs or fails** — check your internet connection; it's
  only needed for the first download of each model.
- **"No documents indexed yet"** — run `python ingest.py` before
  `streamlit run app.py`.
- **Answers seem to ignore your documents** — expand "Sources used" to see
  what was actually retrieved; if it's irrelevant, try increasing `TOP_K`
  in `config.py` or splitting documents into smaller, more focused files.

## Roadmap (matches the original plan's later phases)

- **Week 5 (testing):** use `tests/test_queries.md` as a starting checklist —
  add rows for your real documents, including a few questions the docs
  *don't* answer, to confirm the assistant says "I don't know" instead of
  guessing.
- **Week 6 (docs & demo):** this README + the sidebar's live chunk count
  can serve as your project write-up and live demo script.

## Related projects

- [**tool-calling-agent**](https://github.com/eniscan17/tool-calling-agent) —
  a companion project built on the same Foundry Local setup. Where this
  project always does one fixed step (retrieve → answer), tool-calling-agent
  generalizes that into a small AI agent that decides on its own which tool
  to use (calculator, Wikipedia, current time, or a local knowledge base) and
  can chain multiple tools together before answering.
- [**data-insights-agent**](https://github.com/eniscan17/data-insights-agent) —
  narrows that same tool-calling loop to one focused domain: upload a CSV
  and ask questions in plain language, and the agent calls real pandas/
  matplotlib tools (query, statistical summary, aggregation, charting)
  against your own data instead of guessing at numbers.
