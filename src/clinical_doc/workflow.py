"""Per-field inference workflow.

    retrieve -> generate -> parse --(valid)--------> validate -> END
                   ^          |
                   +-(retry)--+--(retries used up)-> give_up -> END

Each step is a function that takes the state dict and returns a partial
update, which is the node contract LangGraph uses. graph.py wires these same
steps into a LangGraph StateGraph; run_local() executes them with the same
routing and no extra dependency (used in tests and offline runs).
"""

from __future__ import annotations

import json
from typing import Any, TypedDict

from pydantic import ValidationError

from .config import Config
from .llm import PredictRequest, Predictor
from .prompts import SYSTEM_PROMPTS, build_user_prompt
from .retrieval import PatientIndex, Scored
from .schemas import FieldResult, Flag, LLMFieldOutput, PatientRecord, Status
from .validation import validate_field


class FieldState(TypedDict, total=False):
    field_id: str
    retrieved: list[Scored]
    raw_output: Any
    parsed: LLMFieldOutput | None
    parse_error: str | None
    attempts: int
    result: FieldResult


class FieldWorkflow:
    def __init__(
        self,
        config: Config,
        record: PatientRecord,
        index: PatientIndex,
        predictor: Predictor,
        prompt_version: str,
        max_retries: int = 1,
    ):
        if prompt_version not in SYSTEM_PROMPTS:
            raise ValueError(f"unknown prompt version {prompt_version}")
        self.config = config
        self.record = record
        self.index = index
        self.predictor = predictor
        self.prompt_version = prompt_version
        self.max_retries = max_retries

    # ------------------------------------------------------------ nodes

    def retrieve(self, state: FieldState) -> FieldState:
        field = self.config.fields[state["field_id"]]
        return {"retrieved": self.index.retrieve(field, top_k=self.config.retrieval.top_k), "attempts": 0}

    def generate(self, state: FieldState) -> FieldState:
        field = self.config.fields[state["field_id"]]
        attempts = state.get("attempts", 0) + 1
        request = PredictRequest(
            system=SYSTEM_PROMPTS[self.prompt_version],
            user=build_user_prompt(field, state["retrieved"], self.record.assessment_date, state.get("parse_error")),
            metadata={
                "patient_id": self.record.patient_id,
                "field_id": field.field_id,
                "prompt_version": self.prompt_version,
                "attempt": attempts,
            },
        )
        try:
            raw = self.predictor.predict(request)
        except Exception as err:  # provider / structured-output failure counts as invalid output
            raw = {"_error": f"{type(err).__name__}: {err}"}
        return {"raw_output": raw, "attempts": attempts}

    def parse(self, state: FieldState) -> FieldState:
        raw = state["raw_output"]
        try:
            if isinstance(raw, dict) and "_error" in raw:
                raise ValueError(raw["_error"])
            data = json.loads(raw) if isinstance(raw, str) else raw
            return {"parsed": LLMFieldOutput.model_validate(data), "parse_error": None}
        except (ValueError, ValidationError) as err:
            return {"parsed": None, "parse_error": str(err).splitlines()[0]}

    def validate(self, state: FieldState) -> FieldState:
        field = self.config.fields[state["field_id"]]
        out = state["parsed"]
        retrieved_ids = {s.chunk.chunk_id for s in state["retrieved"]}
        flags = validate_field(out, field, self.index, retrieved_ids, self.config.retrieval.stale_evidence_tolerance_days)
        cited = [self.index.by_id[e.chunk_id] for e in out.evidence if e.chunk_id in self.index.by_id]
        return {"result": FieldResult(
            patient_id=self.record.patient_id,
            field_id=field.field_id,
            status=out.status,
            answer=out.answer,
            evidence=out.evidence,
            rationale=out.rationale,
            flags=flags,
            retrieved_chunk_ids=sorted(retrieved_ids),
            retrieved_note_ids=sorted({s.chunk.note_id for s in state["retrieved"]}),
            evidence_dates=sorted({c.date for c in cited}),
            attempts=state["attempts"],
        )}

    def give_up(self, state: FieldState) -> FieldState:
        return {"result": FieldResult(
            patient_id=self.record.patient_id,
            field_id=state["field_id"],
            status=Status.INSUFFICIENT_EVIDENCE,
            rationale=f"Model output invalid after {state['attempts']} attempts: {state.get('parse_error')}",
            flags=[Flag.INVALID_OUTPUT],
            retrieved_chunk_ids=sorted(s.chunk.chunk_id for s in state["retrieved"]),
            retrieved_note_ids=sorted({s.chunk.note_id for s in state["retrieved"]}),
            attempts=state["attempts"],
        )}

    # ---------------------------------------------------------- routing

    def route_after_parse(self, state: FieldState) -> str:
        if state.get("parsed") is not None:
            return "validate"
        if state["attempts"] <= self.max_retries:
            return "retry"
        return "give_up"

    # ------------------------------------------------- local execution

    def run_local(self, field_id: str) -> FieldResult:
        state: FieldState = {"field_id": field_id}
        state.update(self.retrieve(state))
        while True:
            state.update(self.generate(state))
            state.update(self.parse(state))
            route = self.route_after_parse(state)
            if route != "retry":
                break
        state.update(self.validate(state) if route == "validate" else self.give_up(state))
        return state["result"]
