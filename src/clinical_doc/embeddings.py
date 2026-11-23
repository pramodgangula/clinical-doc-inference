"""Embedding models behind one small interface.

OpenAIEmbedder (via LangChain) is what a production run uses. HashingEmbedder
is an offline, deterministic stand-in so tests and the replay evaluation run
without API keys. It only captures surface similarity.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np


class Embedder(Protocol):
    def embed(self, texts: list[str]) -> np.ndarray:
        """Return an (n, d) array of L2-normalised vectors."""


def _l2(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return x / norms


def _fnv1a(s: str) -> int:
    h = 2166136261
    for b in s.encode():
        h = ((h ^ b) * 16777619) & 0xFFFFFFFF
    return h


class HashingEmbedder:
    def __init__(self, dim: int = 512):
        self.dim = dim

    def embed(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            padded = f"  {text.lower()}  "
            for i in range(len(padded) - 2):
                out[row, _fnv1a(padded[i : i + 3]) % self.dim] += 1.0
        return _l2(out)


class OpenAIEmbedder:
    def __init__(self, model: str = "text-embedding-3-small"):
        from langchain_openai import OpenAIEmbeddings  # optional dependency

        self._emb = OpenAIEmbeddings(model=model)

    def embed(self, texts: list[str]) -> np.ndarray:
        return _l2(np.asarray(self._emb.embed_documents(texts), dtype=np.float32))
