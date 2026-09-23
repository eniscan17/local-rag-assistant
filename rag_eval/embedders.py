"""
Pluggable embedding backends.

The app embeds with Foundry Local, which only runs on Apple Silicon Macs.
To run the evaluation anywhere (cloud, CI) the harness also supports
sentence-transformers. `qwen3-embedding-0.6b` — the app's default model —
is available on both, so cloud numbers correspond to the app's setup.
"""

import numpy as np


class Embedder:
    name: str

    def embed_documents(self, texts: list[str]) -> np.ndarray: ...
    def embed_queries(self, texts: list[str]) -> np.ndarray: ...


def _normalize(x) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return x / np.clip(np.linalg.norm(x, axis=1, keepdims=True), 1e-12, None)


class SentenceTransformerEmbedder(Embedder):
    def __init__(self, model_name: str, query_prefix: str = "", doc_prefix: str = "",
                 query_prompt_name: str | None = None, batch_size: int = 16):
        from sentence_transformers import SentenceTransformer

        self.name = model_name
        self.model = SentenceTransformer(model_name, device="cpu")
        self.query_prefix, self.doc_prefix = query_prefix, doc_prefix
        self.query_prompt_name = query_prompt_name
        self.batch_size = batch_size

    def embed_documents(self, texts):
        return _normalize(self.model.encode([self.doc_prefix + t for t in texts],
                                            batch_size=self.batch_size, show_progress_bar=True))

    def embed_queries(self, texts):
        kw = {"prompt_name": self.query_prompt_name} if self.query_prompt_name else {}
        return _normalize(self.model.encode([self.query_prefix + t for t in texts],
                                            batch_size=self.batch_size, show_progress_bar=True, **kw))


class FoundryEmbedder(Embedder):
    """Uses the app's own Foundry Local pipeline (Mac only)."""

    def __init__(self):
        import config
        import llm

        llm.initialize()
        self._llm = llm
        self.name = f"foundry:{config.EMBEDDING_MODEL_ALIAS}"

    def embed_documents(self, texts):
        return _normalize(self._llm.embed_texts(texts))

    def embed_queries(self, texts):
        return _normalize([self._llm.embed_query(t) for t in texts])


# Named presets used by the CLI.
PRESETS = {
    # Same model the app uses, embedded the way the app does it (no instruction).
    "qwen3-0.6b": lambda: SentenceTransformerEmbedder("Qwen/Qwen3-Embedding-0.6B", batch_size=8),
    # Same model with its recommended query instruction.
    "qwen3-0.6b-instruct": lambda: SentenceTransformerEmbedder(
        "Qwen/Qwen3-Embedding-0.6B", query_prompt_name="query", batch_size=8),
    "e5-small": lambda: SentenceTransformerEmbedder(
        "intfloat/multilingual-e5-small", query_prefix="query: ", doc_prefix="passage: "),
    "minilm-multi": lambda: SentenceTransformerEmbedder(
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"),
    "foundry": FoundryEmbedder,
}
