"""Turn raw clinical notes and structured records into retrieval chunks.

Steps
1. Clean: normalise whitespace, quotes and bullets; drop signature boilerplate.
2. Section: split on known headings ("FUNCTIONAL STATUS:", "DISCHARGE MEDICATIONS:", ...).
   Headings are the most reliable signal of what a span is about.
3. Group: for notes without headings, or sections that are too long, group
   consecutive sentences that describe the same clinical concept, using
   rule-based cues plus embedding similarity. Related sentences stay together
   because splitting them can change clinical meaning.
4. Attach metadata: note id, document type, section type, date, source.

Structured record data becomes chunks too, dated with the record's as-of date,
so it competes with notes on recency instead of being silently trusted.
"""

from __future__ import annotations

import re

import numpy as np

from .embeddings import Embedder, HashingEmbedder
from .schemas import Chunk, ClinicalNote, PatientRecord
from .text import keyword_hits, split_sentences

# Heading text (lowercase, without colon) -> section type.
HEADING_MAP: dict[str, str] = {
    "history": "HISTORY",
    "hpi": "HISTORY",
    "hospital course": "HISTORY",
    "functional status": "FUNCTIONAL_STATUS",
    "functional assessment": "FUNCTIONAL_STATUS",
    "adls": "FUNCTIONAL_STATUS",
    "medications": "MEDICATIONS",
    "current medications": "MEDICATIONS",
    "discharge medications": "MEDICATIONS",
    "assessment/plan": "ASSESSMENT_PLAN",
    "assessment and plan": "ASSESSMENT_PLAN",
    "a/p": "ASSESSMENT_PLAN",
    "plan": "ASSESSMENT_PLAN",
    "social": "SOCIAL",
    "social history": "SOCIAL",
    "vitals": "VITALS",
}

_HEADING = re.compile(r"^\s*([A-Za-z/ ]{2,40}):\s*$")
_SIGNATURE = re.compile(r"^\s*(electronically signed|signed by|e-signed)", re.IGNORECASE)

# Keyword topics used to label sentence groups in notes without headings.
TOPIC_KEYWORDS: dict[str, list[str]] = {
    "FUNCTIONAL_STATUS": ["walk", "walks", "walker", "cane", "ambulates", "gait", "shower", "bath",
                          "dress", "dresses", "transfer", "transfers"],
    "MEDICATIONS": ["medication", "medications", "dose", "doses", "pill", "pillbox", "mg"],
    "VITALS": ["vital", "vitals", "bp", "pulse", "afebrile"],
}

CONTINUATION_CUES = ("she ", "he ", "they ", "this ", "it ", "also ", "but ", "however", "which ")


def clean_text(text: str) -> str:
    text = text.replace("’", "'").replace("“", '"').replace("”", '"')
    lines = []
    for line in text.splitlines():
        if _SIGNATURE.match(line):
            continue
        line = re.sub(r"^\s*[-*•]\s+", "", line)   # bullets
        line = re.sub(r"[ \t]+", " ", line).strip()
        lines.append(line)
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


def split_sections(text: str) -> list[tuple[str, str]]:
    """Return (section_type, section_text) pairs. Text before any heading is GENERAL."""
    sections: list[tuple[str, list[str]]] = [("GENERAL", [])]
    for line in text.splitlines():
        m = _HEADING.match(line)
        if m and m.group(1).strip().lower() in HEADING_MAP:
            sections.append((HEADING_MAP[m.group(1).strip().lower()], []))
        elif line.strip():
            sections[-1][1].append(line)
    return [(sec, "\n".join(lines)) for sec, lines in sections if lines]


def topic_of(sentence: str) -> str:
    best, best_hits = "GENERAL", 0
    for topic, kws in TOPIC_KEYWORDS.items():
        n = len(keyword_hits(sentence, kws))
        if n > best_hits:
            best, best_hits = topic, n
    return best


def group_sentences(
    sentences: list[str],
    embedder: Embedder,
    similarity_threshold: float = 0.55,
    max_sentences: int = 4,
) -> list[tuple[str, list[str]]]:
    """Group consecutive sentences about the same clinical concept.

    A new group starts when the topic changes, unless the sentence is a
    continuation of the previous one (starts with a pronoun or connector) or
    is semantically very close to the current group.
    """
    if not sentences:
        return []
    vecs = embedder.embed(sentences)
    groups: list[tuple[str, list[int]]] = []
    for i, sent in enumerate(sentences):
        topic = topic_of(sent)
        if not groups:
            groups.append((topic, [i]))
            continue
        g_topic, members = groups[-1]
        topic_clash = topic != "GENERAL" and g_topic != "GENERAL" and topic != g_topic
        continuation = sent.lower().startswith(CONTINUATION_CUES)
        centroid = vecs[members].mean(axis=0)
        similar = float(vecs[i] @ centroid / (np.linalg.norm(centroid) or 1.0)) >= similarity_threshold
        if not topic_clash and len(members) < max_sentences and (continuation or similar or topic == g_topic != "GENERAL"):
            groups[-1] = (g_topic if g_topic != "GENERAL" else topic, members + [i])
        else:
            groups.append((topic, [i]))
    return [(topic, [sentences[i] for i in members]) for topic, members in groups]


def chunk_note(
    note: ClinicalNote,
    patient_id: str,
    embedder: Embedder,
    max_section_chars: int = 600,
) -> list[Chunk]:
    chunks: list[Chunk] = []

    def add(section: str, text: str) -> None:
        chunks.append(Chunk(
            chunk_id=f"{note.note_id}-c{len(chunks):02d}",
            patient_id=patient_id,
            note_id=note.note_id,
            source="NOTE",
            doc_type=note.doc_type,
            section=section,
            date=note.date,
            text=text,
        ))

    for section, text in split_sections(clean_text(note.text)):
        if section != "GENERAL" and len(text) <= max_section_chars:
            add(section, text)  # a short headed section is already one coherent unit
            continue
        for topic, sents in group_sentences(split_sentences(text), embedder):
            add(section if section != "GENERAL" else topic, " ".join(sents))
    return chunks


def chunk_structured(record: PatientRecord) -> list[Chunk]:
    s = record.structured
    note_id = f"{record.patient_id}-structured"

    def make(i: int, section: str, text: str, when) -> Chunk:
        return Chunk(chunk_id=f"{note_id}-c{i:02d}", patient_id=record.patient_id, note_id=note_id,
                     source="STRUCTURED", doc_type="STRUCTURED_RECORD", section=section, date=when, text=text)

    chunks = []
    if s.diagnoses:
        chunks.append(make(len(chunks), "HISTORY", "Active diagnoses: " + "; ".join(s.diagnoses) + ".", s.as_of))
    if s.medications:
        meds = "; ".join(f"{m.name} {m.dose}".strip() for m in s.medications)
        chunks.append(make(len(chunks), "MEDICATIONS", f"Medication list: {meds}.", s.as_of))
    for pa in s.prior_assessments:
        chunks.append(make(len(chunks), "PRIOR_ASSESSMENT",
                           f"Prior assessment of {pa.field}: {pa.value}.", pa.date))
    return chunks


def build_chunks(record: PatientRecord, embedder: Embedder | None = None) -> list[Chunk]:
    embedder = embedder or HashingEmbedder()
    chunks = chunk_structured(record)
    for note in record.notes:
        chunks.extend(chunk_note(note, record.patient_id, embedder))
    return chunks
