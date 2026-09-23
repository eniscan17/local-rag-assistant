"""
Central configuration for the local RAG assistant.

Model aliases below are Foundry Local catalog names. If a model fails to
load, run this in a terminal to see everything available on your machine:

    python -c "from foundry_local_sdk import Configuration, FoundryLocalManager; \
FoundryLocalManager.initialize(Configuration(app_name='check')); \
print([m.alias for m in FoundryLocalManager.instance.catalog.list_models()])"
"""

import os

# --- Foundry Local models -------------------------------------------------
# Small, fast embedding model (used to vectorize document chunks & queries).
EMBEDDING_MODEL_ALIAS = "qwen3-embedding-0.6b"

# Small, fast chat model used to generate grounded answers.
# Benchmarked with eval_harness.py (tests/eval_results/): qwen2.5-0.5b scores
# 603ms mean latency / 70.8% answer-keyword-match; phi-3.5-mini scores
# 5387ms mean latency / 100% answer-keyword-match (retrieval + guardrail
# accuracy are identical either way, since only the chat model changes).
# Default is phi-3.5-mini: for a demo/portfolio project, a wrong answer costs
# more credibility than a few extra seconds of latency. Switch to
# "qwen2.5-0.5b" if this is ever used somewhere response time is critical.
CHAT_MODEL_ALIAS = "phi-3.5-mini"

APP_NAME = "foundry_local_rag_summer_school"

# --- Chat generation ----------------------------------------------------
# Caps how long an answer can be. Smaller = faster to generate, and on this
# project's eval set (tests/eval_results/) also more reliable: the wrong
# answers small models gave were almost always rambling past the direct
# answer into an unsupported or drifted claim, not the first sentence.
CHAT_MAX_TOKENS = 200

# --- Retrieval --------------------------------------------------------------
TOP_K = 3                # how many chunks to retrieve per question
CHUNK_MAX_CHARS = 800     # rough max size of a chunk before it's split further

# Minimum cosine-similarity score the best-matching chunk must have before we
# even bother asking the chat model. Small models sometimes ignore the
# "say you don't know" instruction and answer from general knowledge anyway
# (Week 5 evaluation finding) — this threshold enforces it deterministically
# instead of relying on the model to behave. Tune by testing real queries:
# lower it if legitimate questions get rejected, raise it if off-topic
# questions still get answered.
MIN_RELEVANCE_SCORE = 0.35

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
