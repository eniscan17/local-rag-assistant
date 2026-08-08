"""
Thin wrapper around the Foundry Local SDK.

Handles: starting the local runtime, downloading/loading the embedding and
chat models (Week 1 "Hello Model" + Week 4 "LLM Integration"), and exposing
simple functions the rest of the app can call without knowing SDK details.

Uses the `foundry_local_sdk` package (foundry-local-sdk on PyPI, v1.x+),
which manages models in-process via a catalog API.
"""

from foundry_local_sdk import Configuration, FoundryLocalManager

import config

_manager = None
_embedding_model = None
_chat_model = None
_embedding_client = None
_chat_client = None


def initialize(progress_callback=None):
    """
    Start the Foundry Local runtime and load both models.
    Safe to call more than once (subsequent calls are no-ops).
    The first call will download models if they aren't cached yet —
    this needs internet access exactly once per model.
    """
    global _manager, _embedding_model, _chat_model, _embedding_client, _chat_client

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

    _embedding_model = _manager.catalog.get_model(config.EMBEDDING_MODEL_ALIAS)
    _embedding_model.download(lambda p: report("Downloading embedding model", p))
    _embedding_model.load()
    _embedding_client = _embedding_model.get_embedding_client()

    _chat_model = _manager.catalog.get_model(config.CHAT_MODEL_ALIAS)
    _chat_model.download(lambda p: report("Downloading chat model", p))
    _chat_model.load()
    _chat_client = _chat_model.get_chat_client()

    if progress_callback:
        progress_callback("Models ready", 100)


def shutdown():
    global _manager, _embedding_model, _chat_model, _embedding_client, _chat_client
    if _embedding_model:
        _embedding_model.unload()
    if _chat_model:
        _chat_model.unload()
    _manager = _embedding_model = _chat_model = None
    _embedding_client = _chat_client = None


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a batch of strings (used during ingestion)."""
    if _embedding_client is None:
        raise RuntimeError("Call llm.initialize() first.")
    response = _embedding_client.generate_embeddings(texts)
    return [item.embedding for item in response.data]


def embed_query(text: str) -> list[float]:
    """Embed a single query string."""
    if _embedding_client is None:
        raise RuntimeError("Call llm.initialize() first.")
    response = _embedding_client.generate_embedding(text)
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
