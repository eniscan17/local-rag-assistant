"""
Answer-quality metrics for extractive QA (SQuAD-style), made language-aware.

- exact_match / f1: the official SQuAD v1.1 definitions (normalise, then
  compare strings / bag-of-tokens), max over the gold answers.
- contains: does the normalised gold answer appear inside the normalised
  prediction? The app's prompt asks for 2-4 sentences, not a span, so EM is
  meaningless there; "contains" is the fair metric for full-sentence answers.

Normalisation differs from the SQuAD script in two documented ways:
- Turkish-aware lowercasing ("I" -> "ı", "İ" -> "i"; str.lower gets these wrong).
- Turkish suffixes split off by an apostrophe are dropped ("Paris'te" ->
  "paris", "1891'de" -> "1891"). Turkish attaches case endings to proper
  nouns and numbers this way, so "Paris'te" vs gold "Paris" is the same
  answer, not a wrong one. Only applied for lang="tr".
"""

import re
import string
import unicodedata
from collections import Counter

_EN_ARTICLES = re.compile(r"\b(a|an|the)\b")
_TR_APOSTROPHE_SUFFIX = re.compile(r"['’]\w+")
_PUNCT = set(string.punctuation) | {"’", "‘", "“", "”", "«", "»", "…", "–", "—"}

REFUSAL = "unanswerable"
_REFUSAL_MARKERS = (
    REFUSAL, "cevaplanamaz", "yanıtlanamaz",
    "don't know", "do not know", "not mentioned", "does not contain", "doesn't contain",
    "no information", "bilmiyorum", "bilgi yok", "belirtilmemiş", "içermiyor",
)


def lower(text: str, lang: str) -> str:
    if lang == "tr":
        text = text.replace("I", "ı").replace("İ", "i")
    return unicodedata.normalize("NFC", text.lower())


def normalize(text: str, lang: str = "en") -> str:
    text = lower(text, lang)
    if lang == "tr":
        text = _TR_APOSTROPHE_SUFFIX.sub("", text)
    text = "".join(" " if ch in _PUNCT else ch for ch in text)
    if lang == "en":
        text = _EN_ARTICLES.sub(" ", text)
    return " ".join(text.split())


def exact_match(pred: str, golds: list[str], lang: str = "en") -> float:
    p = normalize(pred, lang)
    return float(any(p == normalize(g, lang) for g in golds))


def _f1(pred_toks: list[str], gold_toks: list[str]) -> float:
    if not pred_toks or not gold_toks:
        return float(pred_toks == gold_toks)
    common = Counter(pred_toks) & Counter(gold_toks)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision, recall = overlap / len(pred_toks), overlap / len(gold_toks)
    return 2 * precision * recall / (precision + recall)


def f1(pred: str, golds: list[str], lang: str = "en") -> float:
    p = normalize(pred, lang).split()
    return max(_f1(p, normalize(g, lang).split()) for g in golds)


def contains(pred: str, golds: list[str], lang: str = "en") -> float:
    """Gold answer appears in the prediction on token boundaries ("10" is not in "2010")."""
    p = f" {normalize(pred, lang)} "
    return float(any(f" {normalize(g, lang)} " in p for g in golds if normalize(g, lang)))


def is_refusal(pred: str) -> bool:
    p = lower(pred, "tr")  # Turkish-aware lowercasing is harmless for English markers
    return not p.strip() or any(m in p for m in _REFUSAL_MARKERS)


def score(pred: str, golds: list[str], lang: str) -> dict:
    return {
        "em": exact_match(pred, golds, lang),
        "f1": f1(pred, golds, lang),
        "contains": contains(pred, golds, lang),
        "refused": float(is_refusal(pred)),
    }
