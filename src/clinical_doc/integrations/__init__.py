"""Optional integrations with LangSmith, Ragas and DeepEval.

The core metrics in clinical_doc.evaluation have no external dependencies.
These modules push the same runs into the tools used for experiment tracking
and LLM-judged metrics (faithfulness, context recall).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..config import load_config
from ..pipeline import ClinicalDocPipeline, load_patient
from ..retrieval import PatientIndex


@dataclass
class RagSample:
    """One field prediction with its retrieved context, for LLM-judged metrics."""

    patient_id: str
    field_id: str
    question: str
    contexts: list[str]
    response: str
    reference: str


def collect_samples(pipeline: ClinicalDocPipeline, patients_dir: Path, gold: list) -> list[RagSample]:
    config = pipeline.config
    gold_by_key = {(g.patient_id, g.field_id): g for g in gold}
    samples = []
    for path in sorted(Path(patients_dir).glob("*.json")):
        record = load_patient(path)
        index: PatientIndex = pipeline.build_index(record)
        for res in pipeline.run(record).fields:
            g = gold_by_key[(res.patient_id, res.field_id)]
            answer = res.answer if res.status.value == "ANSWERED" else res.status.value
            reference = g.answer if g.status.value == "ANSWERED" else g.status.value
            samples.append(RagSample(
                patient_id=res.patient_id,
                field_id=res.field_id,
                question=config.fields[res.field_id].question,
                contexts=[index.by_id[c].text for c in res.retrieved_chunk_ids],
                response=f"{answer}. {res.rationale}",
                reference=str(reference),
            ))
    return samples


__all__ = ["RagSample", "collect_samples", "load_config"]
