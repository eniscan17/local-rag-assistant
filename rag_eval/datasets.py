"""
XQuAD loader (https://github.com/google-deepmind/xquad, CC BY-SA 4.0).

XQuAD contains the *same* 240 Wikipedia paragraphs and 1,190 questions
professionally translated into several languages. Loading the English and
Turkish versions gives a controlled experiment: identical content and
questions, only the language differs.

Paragraphs are chunked with the app's own chunker (chunking.py), so the
evaluation measures the retrieval setup the app actually uses. A chunk is
relevant to a question if it overlaps the gold answer's character span.
"""

import json
import os
import urllib.request
from dataclasses import dataclass, field

from chunking import chunk_text

XQUAD_URL = "https://raw.githubusercontent.com/google-deepmind/xquad/master/xquad.{lang}.json"
CACHE_DIR = os.path.join(os.path.dirname(__file__), "data")


@dataclass
class Chunk:
    id: str
    text: str


@dataclass
class Query:
    id: str
    question: str
    answers: list[str]
    relevant: set[str] = field(default_factory=set)
    paragraph_id: str = ""


@dataclass
class Dataset:
    lang: str
    chunks: list[Chunk]
    queries: list[Query]
    contexts: dict[str, str]  # paragraph_id -> full paragraph text


def _download(lang: str) -> str:
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = os.path.join(CACHE_DIR, f"xquad.{lang}.json")
    if not os.path.exists(path):
        urllib.request.urlretrieve(XQUAD_URL.format(lang=lang), path)
    return path


def chunk_with_offsets(text: str, max_chars: int) -> list[tuple[str, int, int]]:
    """Chunk text and recover each chunk's [start, end) offsets in the original."""
    out, cursor = [], 0
    for chunk in chunk_text(text, max_chars=max_chars):
        start = text.find(chunk, cursor)
        if start < 0:
            # Merged paragraphs can differ in whitespace; fall back to the
            # first paragraph of the chunk to anchor it.
            start = text.find(chunk.split("\n\n")[0], cursor)
        if start < 0:
            raise ValueError("could not locate chunk in source text")
        end = start + len(chunk)
        out.append((chunk, start, end))
        cursor = end
    return out


def load_xquad(lang: str, max_chars: int, limit_queries: int | None = None) -> Dataset:
    with open(_download(lang), encoding="utf-8") as f:
        articles = json.load(f)["data"]

    chunks, queries, contexts = [], [], {}
    for a_idx, article in enumerate(articles):
        for p_idx, para in enumerate(article["paragraphs"]):
            pid = f"a{a_idx}-p{p_idx}"
            context = para["context"]
            contexts[pid] = context
            spans = []
            for c_idx, (text, start, end) in enumerate(chunk_with_offsets(context, max_chars)):
                cid = f"{pid}-c{c_idx}"
                chunks.append(Chunk(cid, text))
                spans.append((cid, start, end))

            for qa in para["qas"]:
                ans = qa["answers"][0]
                a_start = ans["answer_start"]
                a_end = a_start + len(ans["text"])
                relevant = {cid for cid, s, e in spans if s < a_end and a_start < e}
                queries.append(Query(
                    id=qa["id"],
                    question=qa["question"],
                    answers=[x["text"] for x in qa["answers"]],
                    relevant=relevant,
                    paragraph_id=pid,
                ))

    if limit_queries:
        queries = queries[:limit_queries]
    return Dataset(lang=lang, chunks=chunks, queries=queries, contexts=contexts)
