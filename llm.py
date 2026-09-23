"""
Thin wrapper around the Foundry Local SDK.

Handles: starting the local runtime, downloading/loading the embedding and
chat models (Week 1 "Hello Model" + Week 4 "LLM Integration"), and exposing
simple functions the rest of the app can call without knowing SDK details.

Uses the `foundry_local_sdk` package (foundry-local-sdk on PyPI, v1.x+),
which manages models in-process via a catalog API.

Embeddings come from either Foundry Local or sentence-transformers,
depending on config.EMBEDDING_BACKEND; the chat model is always Foundry.
"""

from foundry_local_sdk import Configuration, FoundryLocalManager

import config

_manager = None
_embedding_model = None
_chat_model = None
_embedding_client = None
_chat_client = None
_st_model = None


def embedder_id() -> str:
    """Identifies the embedding setup; stored with the index to detect stale vectors."""
    if config.EMBEDDING_BACKEND == "sentence-transformers":
        return f"st:{config.ST_EMBEDDING_MODEL}"
    return f"foundry:{config.EMBEDDING_MODEL_ALIAS}+instruct"


def initialize(progress_callback=None):
    """
    Start the Foundry Local runtime and load both models.
    Safe to call more than once (subsequent calls are no-ops).
    The first call will download models if they aren't cached yet —
    this needs internet access exactly once per model.
    """
    global _manager, _embedding_model, _chat_model, _embedding_client, _chat_client, _st_model

    if _manager is not None:
        return  # already initialized

    def report(stage, pct):
        if progress_callback:
            progress_callback(stage, pct)
        else:
            print(f"\r{stage}: {pct:.1f}%", end="", flush=True)

    cfg = Configuration(app_name=config.APP_NAME)
    FoundryLocalManager.initialize(cfg)
    _manager = FoundryLocalManager.instance

    if config.EMBEDDING_BACKEND == "sentence-transformers":
        from sentence_transformers import SentenceTransformer

        report("Loading embedding model", 0)
        _st_model = SentenceTransformer(config.ST_EMBEDDING_MODEL)
        report("Loading embedding model", 100)
    elif config.EMBEDDING_BACKEND == "foundry":
        _embedding_model = _manager.catalog.get_model(config.EMBEDDING_MODEL_ALIAS)
        _embedding_model.download(lambda p: report("Downloading embedding model", p))
        _embedding_model.load()
        _embedding_client = _embedding_model.get_embedding_client()
    else:
        raise ValueError(f"Unknown EMBEDDING_BACKEND: {config.EMBEDDING_BACKEND!r}")

    _chat_model = _manager.catalog.get_model(config.CHAT_MODEL_ALIAS)
    _chat_model.download(lambda p: report("Downloading chat model", p))
    _chat_model.load()
    _chat_client = _chat_model.get_chat_client()
    _chat_client.settings.max_tokens = config.CHAT_MAX_TOKENS

    if progress_callback:
        progress_callback("Models ready", 100)


def shutdown():
    global _manager, _embedding_model, _chat_model, _embedding_client, _chat_client, _st_model
    if _embedding_model:
        _embedding_model.unload()
    if _chat_model:
        _chat_model.unload()
    _manager = _embedding_model = _chat_model = None
    _embedding_client = _chat_client = _st_model = None


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a batch of document chunks (used during ingestion)."""
    if _st_model is not None:
        vecs = _st_model.encode([config.ST_DOC_PREFIX + t for t in texts], normalize_embeddings=True)
        return [v.tolist() for v in vecs]
    if _embedding_client is None:
        raise RuntimeError("Call llm.initialize() first.")
    response = _embedding_client.generate_embeddings(texts)
    return [item.embedding for item in response.data]


def embed_query(text: str) -> list[float]:
    """Embed a single question (with the query prefix/instruction the model expects)."""
    if _st_model is not None:
        return _st_model.encode(config.ST_QUERY_PREFIX + text, normalize_embeddings=True).tolist()
    if _embedding_client is None:
        raise RuntimeError("Call llm.initialize() first.")
    response = _embedding_client.generate_embedding(config.FOUNDRY_QUERY_INSTRUCTION + text)
    return response.data[0].embedding


def answer_stream(system_prompt: str, question: str):
    """Yield answer text chunks as they are generated (streaming)."""
    if _chat_client is None:
        raise RuntimeError("Call llm.initialize() first.")
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]
    for chunk in _chat_client.complete_streaming_chat(messages):
        if not chunk.choices:
            continue
        content = chunk.choices[0].delta.content
        if content:
            yield content
