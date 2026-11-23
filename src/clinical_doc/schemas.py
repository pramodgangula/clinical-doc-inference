"""Typed data models shared across the pipeline."""

from __future__ import annotations

from datetime import date
from enum import Enum

from pydantic import BaseModel, Field, model_validator

# ---------------------------------------------------------------- inputs


class Medication(BaseModel):
    name: str
    dose: str = ""


class PriorAssessment(BaseModel):
    date: date
    field: str
    value: str


class StructuredRecord(BaseModel):
    as_of: date
    age: int | None = None
    diagnoses: list[str] = Field(default_factory=list)
    medications: list[Medication] = Field(default_factory=list)
    prior_assessments: list[PriorAssessment] = Field(default_factory=list)


class ClinicalNote(BaseModel):
    note_id: str
    doc_type: str
    author_role: str = ""
    date: date
    text: str


class PatientRecord(BaseModel):
    patient_id: str
    assessment_date: date
    structured: StructuredRecord
    notes: list[ClinicalNote]


# ------------------------------------------------------------- retrieval


class Chunk(BaseModel):
    """Retrieval unit: a clinically coherent span with the metadata needed to filter and date it."""

    chunk_id: str
    patient_id: str
    note_id: str
    source: str  # NOTE or STRUCTURED
    doc_type: str
    section: str
    date: date
    text: str

    def header(self) -> str:
        return f"[{self.chunk_id}] {self.doc_type} | {self.section} | {self.date.isoformat()}"


# ------------------------------------------------------------- inference


class Status(str, Enum):
    ANSWERED = "ANSWERED"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"  # not supported by the retrieved context
    CONFLICTING_EVIDENCE = "CONFLICTING_EVIDENCE"    # sources disagree and recency does not resolve it


Answer = str | list[str] | None


class EvidenceRef(BaseModel):
    chunk_id: str = Field(description="ID of the excerpt the quote comes from")
    quote: str = Field(description="Exact words copied from that excerpt")


class LLMFieldOutput(BaseModel):
    """Structured output contract for one documentation field."""

    status: Status
    answer: Answer = Field(default=None, description="Allowed value, list of items, or null when not ANSWERED")
    evidence: list[EvidenceRef] = Field(default_factory=list)
    rationale: str = ""

    @model_validator(mode="after")
    def _answer_matches_status(self) -> "LLMFieldOutput":
        if self.status == Status.ANSWERED and (self.answer is None or self.answer == [] or self.answer == ""):
            raise ValueError("status ANSWERED requires an answer")
        if self.status != Status.ANSWERED and self.answer not in (None, [], ""):
            raise ValueError(f"status {self.status.value} must have a null answer")
        if self.status != Status.ANSWERED:
            self.answer = None
        return self


class Flag(str, Enum):
    """Reasons a field is routed to clinician review instead of auto-filled."""

    INVALID_OUTPUT = "INVALID_OUTPUT"                    # failed the schema after retries
    INVALID_VALUE = "INVALID_VALUE"                      # value not allowed for this field
    MISSING_EVIDENCE = "MISSING_EVIDENCE"                # answered without citing evidence
    UNGROUNDED_EVIDENCE = "UNGROUNDED_EVIDENCE"          # quote not found in the cited chunk
    EVIDENCE_OUTSIDE_CONTEXT = "EVIDENCE_OUTSIDE_CONTEXT"  # cites a chunk the model was not given
    EVIDENCE_OFF_TOPIC = "EVIDENCE_OFF_TOPIC"            # evidence never mentions this field's topic
    STALE_EVIDENCE = "STALE_EVIDENCE"                    # newer documentation on the topic was ignored
    EVIDENCE_IN_CONTEXT_IGNORED = "EVIDENCE_IN_CONTEXT_IGNORED"  # abstained though topic is in context
    POSSIBLE_RETRIEVAL_MISS = "POSSIBLE_RETRIEVAL_MISS"  # abstained; topic exists outside the context
    CONSISTENCY_CONFLICT = "CONSISTENCY_CONFLICT"        # contradicts a related field


class FieldResult(BaseModel):
    patient_id: str
    field_id: str
    status: Status
    answer: Answer = None
    evidence: list[EvidenceRef] = Field(default_factory=list)
    rationale: str = ""
    flags: list[Flag] = Field(default_factory=list)
    retrieved_chunk_ids: list[str] = Field(default_factory=list)
    retrieved_note_ids: list[str] = Field(default_factory=list)
    evidence_dates: list[date] = Field(default_factory=list)
    attempts: int = 0

    @property
    def needs_review(self) -> bool:
        """Only a clean, grounded ANSWERED result is auto-filled."""
        return self.status != Status.ANSWERED or bool(self.flags)

    def to_dict(self) -> dict:
        d = self.model_dump(mode="json")
        d["needs_review"] = self.needs_review
        return d


class PatientResult(BaseModel):
    patient_id: str
    fields: list[FieldResult]


class GoldLabel(BaseModel):
    patient_id: str
    field_id: str
    status: Status
    answer: Answer = None
    evidence_note_ids: list[str] = Field(default_factory=list)
