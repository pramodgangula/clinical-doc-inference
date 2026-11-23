"""Text helpers shared by preprocessing, retrieval and validation."""

from __future__ import annotations

import re

_WORD = re.compile(r"[a-z0-9]+")
_SUFFIXES = ("ing", "ed", "es", "s")


def stem(word: str) -> str:
    for suffix in _SUFFIXES:
        if word.endswith(suffix) and len(word) - len(suffix) >= 4:
            return word[: -len(suffix)]
    return word


def tokenize(text: str) -> list[str]:
    return [stem(w) for w in _WORD.findall(text.lower())]


def normalize(text: str) -> str:
    """Lowercase, drop punctuation, collapse whitespace (for quote matching)."""
    return " ".join(_WORD.findall(text.lower()))


def keyword_hits(text: str, keywords: list[str]) -> list[str]:
    """Configured keywords (single or multi-word) that occur in the text."""
    tokens = tokenize(text)
    token_set, joined = set(tokens), " ".join(tokens)
    hits = []
    for kw in keywords:
        kw_tokens = tokenize(kw)
        if len(kw_tokens) == 1 and kw_tokens[0] in token_set:
            hits.append(kw)
        elif len(kw_tokens) > 1 and " ".join(kw_tokens) in joined:
            hits.append(kw)
    return hits


_SENTENCE_END = re.compile(r"(?<=[.!?;])\s+(?=[A-Z])")


def split_sentences(text: str) -> list[str]:
    sentences = []
    for line in text.splitlines():
        line = line.strip()
        if line:
            sentences.extend(s.strip() for s in _SENTENCE_END.split(line) if s.strip())
    return sentences
