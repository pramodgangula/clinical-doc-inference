"""Patient-level pipeline: preprocess -> index -> per-field workflow -> cross-field checks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from .config import Config
from .embeddings import Embedder
from .graph import build_field_graph, langgraph_available
from .llm import Predictor
from .preprocessing import build_chunks
from .prompts import LATEST_PROMPT_VERSION
from .retrieval import PatientIndex
from .schemas import FieldResult, PatientRecord, PatientResult
from .validation import apply_consistency_rules
from .workflow import FieldWorkflow

Engine = Literal["auto", "langgraph", "local"]


def load_patient(path: Path) -> PatientRecord:
    return PatientRecord.model_validate(json.loads(Path(path).read_text()))


class ClinicalDocPipeline:
    def __init__(
        self,
        config: Config,
        predictor: Predictor,
        prompt_version: str = LATEST_PROMPT_VERSION,
        embedder: Embedder | None = None,
        engine: Engine = "auto",
    ):
        self.config = config
        self.predictor = predictor
        self.prompt_version = prompt_version
        self.embedder = embedder
        if engine == "auto":
            engine = "langgraph" if langgraph_available() else "local"
        self.engine = engine

    def build_index(self, record: PatientRecord) -> PatientIndex:
        return PatientIndex(build_chunks(record, self.embedder), record.assessment_date, embedder=self.embedder)

    def run(self, record: PatientRecord, field_ids: list[str] | None = None) -> PatientResult:
        index = self.build_index(record)
        wf = FieldWorkflow(self.config, record, index, self.predictor, self.prompt_version)
        graph = build_field_graph(wf) if self.engine == "langgraph" else None

        results: list[FieldResult] = []
        for field_id in field_ids or list(self.config.fields):
            if graph is not None:
                results.append(graph.invoke({"field_id": field_id})["result"])
            else:
                results.append(wf.run_local(field_id))

        apply_consistency_rules(results, self.config.consistency_rules)
        return PatientResult(patient_id=record.patient_id, fields=results)
