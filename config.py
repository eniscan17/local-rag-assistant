"""
Central configuration for the local RAG assistant.

Model aliases below are Foundry Local catalog names. If a model fails to
load, run this in a terminal to see everything available on your machine:

    python -c "from foundry_local_sdk import Configuration, FoundryLocalManager; \
FoundryLocalManager.initialize(Configuration(app_name='check')); \
print([m.alias for m in FoundryLocalManager.instance.catalog.list_models()])"
"""

import os

# --- Embeddings ---------------------------------------------------------------
# Chosen with the XQuAD EN/TR retrieval benchmark (rag_eval/, see README):
#   "sentence-transformers": multilingual-e5-small, best measured setup
#       (TR hit@1 89.5% in hybrid mode vs 74.5% for the original setup),
#       5x smaller than qwen3-embedding-0.6b. Downloads once from Hugging
#       Face, then runs offline like everything else.
#   "foundry": qwen3-embedding-0.6b through Foundry Local (no PyTorch
#       needed), with the query instruction the model expects.
# Changing the backend changes every stored vector: rebuild the knowledge
# base afterwards (python ingest.py). The app warns if you forget.
EMBEDDING_BACKEND = "sentence-transformers"

ST_EMBEDDING_MODEL = "intfloat/multilingual-e5-small"
ST_QUERY_PREFIX = "query: "      # e5 models are trained with these prefixes
ST_DOC_PREFIX = "passage: "

# Foundry Local embedding model (used when EMBEDDING_BACKEND = "foundry").
EMBEDDING_MODEL_ALIAS = "qwen3-embedding-0.6b"
# Qwen3-Embedding expects an instruction on queries (not on documents).
# Omitting it — as the app originally did — costs 5.0 hit@1 points on
# Turkish and 4.0 on English in the benchmark.
FOUNDRY_QUERY_INSTRUCTION = (
    "Instruct: Given a web search query, retrieve relevant passages that "
    "answer the query\nQuery:"
)

# --- Foundry Local chat model ------------------------------------------------

# Chat model used to generate grounded answers. Chosen with the XQuAD EN/TR
# answer-quality benchmark (rag_eval/run_answers.py, see README):
#   phi-3.5-mini: Turkish F1 43.6 with the app's retriever, English 75.4
#   qwen2.5-7b:   Turkish F1 68.7 (+25.1), English 78.3; ~1.3x slower
# On the app's own 32-question eval set qwen2.5-7b keeps 0% false refusals
# and 100% out-of-scope refusals (keyword match 95.8% vs 100%, one miss).
# Lighter options: "phi-3.5-mini" (2.2 GB) or "qwen2.5-0.5b" (fastest, weakest).
CHAT_MODEL_ALIAS = "qwen2.5-7b"

APP_NAME = "foundry_local_rag_summer_school"

# --- Chat generation ----------------------------------------------------
# Caps how long an answer can be. Smaller = faster to generate, and on this
# project's eval set (tests/eval_results/) also more reliable: the wrong
# answers small models gave were almost always rambling past the direct
# answer into an unsupported or drifted claim, not the first sentence.
CHAT_MAX_TOKENS = 200

# --- Retrieval --------------------------------------------------------------
TOP_K = 3                # how many chunks to retrieve per question

# "hybrid" = dense + Turkish-aware BM25, fused with reciprocal rank fusion
# (best in the benchmark); "dense" = embeddings only (original behaviour).
RETRIEVAL_MODE = "hybrid"
BM25_PREFIX_LEN = 5       # "F5 stemming" for BM25; None disables it
CHUNK_MAX_CHARS = 800     # rough max size of a chunk before it's split further

# Minimum cosine-similarity score the best-matching chunk must have before we
# even bother asking the chat model. Small models sometimes ignore the
# "say you don't know" instruction and answer from general knowledge anyway
# (Week 5 evaluation finding) — this threshold enforces it deterministically
# instead of relying on the model to behave. Tune by testing real queries:
# lower it if legitimate questions get rejected, raise it if off-topic
# questions still get answered.
#
# The right value depends on the embedding model — e5 similarities sit in a
# much higher, narrower range than Qwen3's — so there is one value per
# backend. Re-derive it with:  python -m rag_eval.calibrate_threshold
MIN_RELEVANCE_SCORES = {
    "foundry": 0.35,                 # hand-tuned in Week 5 (pre-instruction)
    "sentence-transformers": 0.805,  # calibrated with rag_eval.calibrate_threshold
}
MIN_RELEVANCE_SCORE = MIN_RELEVANCE_SCORES[EMBEDDING_BACKEND]

# --- Paths -------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DOCUMENTS_DIR = os.path.join(BASE_DIR, "documents")
DATA_DIR = os.path.join(BASE_DIR, "data")
DB_PATH = os.path.join(DATA_DIR, "knowledge_base.sqlite3")

# --- System prompt (Week 2 / Week 4 prompt-engineering guidance) ------------
SYSTEM_PROMPT_TEMPLATE = (
    "You are a helpful assistant that answers questions using ONLY the "
    "context provided below, which was retrieved from the user's own "
    "documents. If the context does not contain enough information to "
    "answer confidently, say you don't know instead of guessing. When you "
    "do answer, mention which source(s) you used. Keep the answer to 2-4 "
    "concise sentences — do not add extra explanation beyond what directly "
    "answers the question.\n\n"
    "Context:\n{context}"
)
