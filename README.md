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
documents/        your source .txt / .md files (sample docs included)
data/             generated SQLite database (created automatically)
tests/test_queries.md   sample test questions for manual QA (Week 5 style)
```

## Customizing

- **Change models:** edit `EMBEDDING_MODEL_ALIAS` / `CHAT_MODEL_ALIAS` in
  `config.py`. `phi-3.5-mini` is a good larger/slower chat model if you
  want better answers and don't mind a bit more latency.
- **Change how many chunks are retrieved:** `TOP_K` in `config.py`.
- **Change the system prompt / tone:** `SYSTEM_PROMPT_TEMPLATE` in `config.py`.

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
