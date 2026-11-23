"""Hybrid retrieval: metadata filter -> BM25 + vector search -> RRF fusion -> rerank.

* Metadata filtering narrows candidates to the sections and time window that
  can answer the field (e.g. medication fields only look at medication sections).
* BM25 over the field question plus its keywords catches exact clinical terms
  and lay language that semantic similarity alone can miss.
* Vector similarity catches paraphrases.
* Reciprocal Rank Fusion combines the two by rank, avoiding a fragile weight
  between scores on different scales.
* The reranker reorders the fused shortlist. The default HeuristicReranker
  adds topic-keyword overlap and recency; a cross-encoder can be plugged in
  behind the same interface.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Protocol

import numpy as np

from .config import FieldSpec
from .embeddings import Embedder, HashingEmbedder
from .schemas import Chunk
from .text import keyword_hits, tokenize


class BM25:
    def __init__(self, docs: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.tf = [Counter(d) for d in docs]
        self.len = [len(d) for d in docs]
        self.avgdl = sum(self.len) / len(docs) if docs else 0.0
        df: Counter[str] = Counter()
        for tf in self.tf:
            df.update(tf.keys())
        n = len(docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def scores(self, query: list[str], subset: list[int]) -> list[float]:
        out = []
        for i in subset:
            tf, dl, s = self.tf[i], self.len[i], 0.0
            for term in query:
                if term in tf:
                    f = tf[term]
                    s += self.idf[term] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * dl / (self.avgdl or 1)))
            out.append(s)
        return out


@dataclass
class Scored:
    chunk: Chunk
    score: float


class Reranker(Protocol):
    def rerank(self, field: FieldSpec, candidates: list[Scored], assessment_date: date) -> list[Scored]: ...


class HeuristicReranker:
    def __init__(self, keyword_weight: float = 0.01, recency_weight: float = 0.01):
        self.keyword_weight = keyword_weight
        self.recency_weight = recency_weight

    def rerank(self, field: FieldSpec, candidates: list[Scored], assessment_date: date) -> list[Scored]:
        out = []
        for c in candidates:
            age = (assessment_date - c.chunk.date).days
            recency = max(0.0, 1 - age / max(field.lookback_days, 1))
            hits = len(keyword_hits(c.chunk.text, field.keywords))
            out.append(Scored(c.chunk, c.score + self.keyword_weight * hits + self.recency_weight * recency))
        return sorted(out, key=lambda s: -s.score)


def _ranks(scores: list[float]) -> list[int]:
    order = sorted(range(len(scores)), key=lambda i: (-scores[i], i))
    ranks = [0] * len(scores)
    for r, i in enumerate(order):
        ranks[i] = r
    return ranks


class PatientIndex:
    """In-memory index over one patient's chunks (pgvector/FAISS in production)."""

    def __init__(
        self,
        chunks: list[Chunk],
        assessment_date: date,
        embedder: Embedder | None = None,
        reranker: Reranker | None = None,
        rrf_k: int = 60,
    ):
        self.chunks = chunks
        self.by_id = {c.chunk_id: c for c in chunks}
        self.assessment_date = assessment_date
        self.embedder = embedder or HashingEmbedder()
        self.reranker = reranker or HeuristicReranker()
        self.rrf_k = rrf_k
        self.bm25 = BM25([tokenize(c.text) for c in chunks])
        self.vectors = self.embedder.embed([c.text for c in chunks]) if chunks else np.zeros((0, 1))

    def candidates(self, field: FieldSpec) -> list[int]:
        """Metadata filter: allowed sections, within the lookback window, not after the assessment."""
        earliest = self.assessment_date - timedelta(days=field.lookback_days)
        return [
            i for i, c in enumerate(self.chunks)
            if c.section in field.sections and earliest <= c.date <= self.assessment_date
        ]

    def on_topic(self, field: FieldSpec, within_filter: bool = True) -> list[Chunk]:
        """Chunks that mention the field's topic; used by validators, not by the model."""
        pool = [self.chunks[i] for i in self.candidates(field)] if within_filter else self.chunks
        return [c for c in pool if keyword_hits(c.text, field.keywords)]

    def retrieve(self, field: FieldSpec, top_k: int = 4, shortlist: int = 8) -> list[Scored]:
        cand = self.candidates(field)
        if not cand:
            return []
        query_text = f"{field.question} {' '.join(field.keywords)}"
        lexical = self.bm25.scores(tokenize(query_text), cand)
        q = self.embedder.embed([query_text])[0]
        dense = (self.vectors[cand] @ q).tolist()

        lr, dr = _ranks(lexical), _ranks(dense)
        fused = [
            Scored(self.chunks[i], 1 / (self.rrf_k + lr[j]) + 1 / (self.rrf_k + dr[j]))
            for j, i in enumerate(cand)
            if lexical[j] > 0 or dense[j] > 0.2
        ]
        fused.sort(key=lambda s: -s.score)
        return self.reranker.rerank(field, fused[:shortlist], self.assessment_date)[:top_k]
