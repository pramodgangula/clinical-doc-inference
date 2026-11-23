"""Versioned prompts.

v1 is the baseline. v2 came out of error analysis on v1 (see README):
  - explicit abstention options, so the model has an alternative to guessing
  - evidence must be specific to the field being answered
  - current status: weigh documentation dates, do not use superseded information
  - conflicting sources that recency cannot resolve are reported, not resolved silently
"""

from __future__ import annotations

from datetime import date

from .config import FieldSpec
from .retrieval import Scored

SYSTEM_PROMPTS: dict[str, str] = {
    "v1": """\
You are a clinical documentation assistant. Answer the documentation question
for this patient using the patient record excerpts provided. Return status
ANSWERED, the answer, the evidence you used (excerpt id and an exact quote),
and a short rationale.""",

    "v2": """\
You are a clinical documentation assistant. You answer ONE documentation
question for a patient using ONLY the record excerpts provided.

Rules:
1. Evidence must be specific to THIS question. Do not infer one activity from
   another (help with bathing says nothing about dressing).
2. Answer the patient's CURRENT status as of the assessment date. Each excerpt
   shows its date. If newer documentation supersedes older documentation
   (e.g. after a hospitalization, or a medication was discontinued), use the
   newer information.
3. Cite evidence as the excerpt id plus a quote copied exactly from that excerpt.
4. If the excerpts do not support an answer, return INSUFFICIENT_EVIDENCE with
   answer null. This means "not supported by the provided excerpts"; do not
   guess.
5. If sources from the same period disagree (for example the patient and a
   caregiver give incompatible accounts) and recency does not resolve it,
   return CONFLICTING_EVIDENCE with answer null and cite both sides.""",
}

LATEST_PROMPT_VERSION = "v2"


def render_context(retrieved: list[Scored]) -> str:
    if not retrieved:
        return "(no excerpts matched this question)"
    ordered = sorted(retrieved, key=lambda s: s.chunk.date, reverse=True)
    return "\n\n".join(f"{s.chunk.header()}\n{s.chunk.text}" for s in ordered)


def build_user_prompt(field: FieldSpec, retrieved: list[Scored], assessment_date: date, feedback: str | None = None) -> str:
    if field.answer_type == "categorical":
        answer_spec = f"ANSWER: exactly one of {', '.join(field.allowed_values)}"
    else:
        answer_spec = "ANSWER: a list of items"
    prompt = f"""\
ASSESSMENT DATE: {assessment_date.isoformat()}

QUESTION ({field.field_id}): {field.question}
{answer_spec}
GUIDANCE:
{field.guidance.strip()}

RECORD EXCERPTS (newest first):
{render_context(retrieved)}"""
    if feedback:
        prompt += f"\n\nYOUR PREVIOUS RESPONSE WAS INVALID: {feedback}\nRespond again following the required structure."
    return prompt
