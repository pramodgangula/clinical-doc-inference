"""Deterministic checks that decide auto-fill vs clinician review.

The model's own confidence is not used. Each check targets a failure mode
found in evaluation:

  output structure     INVALID_VALUE, MISSING_EVIDENCE
  evidence grounding   UNGROUNDED_EVIDENCE, EVIDENCE_OUTSIDE_CONTEXT
  over-inference       EVIDENCE_OFF_TOPIC
  temporal context     STALE_EVIDENCE
  abstention checks    EVIDENCE_IN_CONTEXT_IGNORED, POSSIBLE_RETRIEVAL_MISS
  related fields       CONSISTENCY_CONFLICT
"""

from __future__ import annotations

from datetime import timedelta

from .config import ConsistencyRule, FieldSpec
from .retrieval import PatientIndex
from .schemas import FieldResult, Flag, LLMFieldOutput, Status
from .text import keyword_hits, normalize


def quote_in(quote: str, text: str) -> bool:
    q = normalize(quote)
    return bool(q) and q in normalize(text)


def validate_field(
    output: LLMFieldOutput,
    field: FieldSpec,
    index: PatientIndex,
    retrieved_ids: set[str],
    stale_tolerance_days: int = 14,
) -> list[Flag]:
    flags: list[Flag] = []

    def add(flag: Flag) -> None:
        if flag not in flags:
            flags.append(flag)

    # --- output structure
    if output.status == Status.ANSWERED:
        if field.answer_type == "categorical":
            if not isinstance(output.answer, str) or output.answer not in field.allowed_values:
                add(Flag.INVALID_VALUE)
        elif not isinstance(output.answer, list) or not all(isinstance(a, str) and a for a in output.answer):
            add(Flag.INVALID_VALUE)
        if not output.evidence:
            add(Flag.MISSING_EVIDENCE)

    # --- evidence grounding
    cited = []
    for ev in output.evidence:
        chunk = index.by_id.get(ev.chunk_id)
        if chunk is None or not quote_in(ev.quote, chunk.text):
            add(Flag.UNGROUNDED_EVIDENCE)
            continue
        if ev.chunk_id not in retrieved_ids:
            add(Flag.EVIDENCE_OUTSIDE_CONTEXT)
        cited.append((ev, chunk))

    if output.status == Status.ANSWERED and cited:
        # --- over-inference: the evidence must actually be about this field
        if field.require_topic_evidence and not any(keyword_hits(ev.quote, field.keywords) for ev, _ in cited):
            add(Flag.EVIDENCE_OFF_TOPIC)

        # --- temporal: newer on-topic documentation exists than anything cited
        newest_cited = max(chunk.date for _, chunk in cited)
        newer = [c for c in index.on_topic(field) if c.date > newest_cited + timedelta(days=stale_tolerance_days)]
        if newer:
            add(Flag.STALE_EVIDENCE)

    # --- abstention: "not supported by the context" is not "not in the record"
    if output.status == Status.INSUFFICIENT_EVIDENCE:
        on_topic_ids = {c.chunk_id for c in index.on_topic(field, within_filter=False)}
        if on_topic_ids & retrieved_ids:
            add(Flag.EVIDENCE_IN_CONTEXT_IGNORED)
        elif on_topic_ids:
            add(Flag.POSSIBLE_RETRIEVAL_MISS)

    return flags


def apply_consistency_rules(results: list[FieldResult], rules: list[ConsistencyRule]) -> None:
    by_field = {r.field_id: r for r in results}
    for rule in rules:
        a, b = by_field.get(rule.when.field), by_field.get(rule.and_.field)
        if a and b and a.answer == rule.when.answer and b.answer == rule.and_.answer:
            for r in (a, b):
                if Flag.CONSISTENCY_CONFLICT not in r.flags:
                    r.flags.append(Flag.CONSISTENCY_CONFLICT)
