"""Run the gold set as a LangSmith experiment.

    export LANGSMITH_API_KEY=... OPENAI_API_KEY=...
    python -m clinical_doc.integrations.langsmith_eval --prompt-version v2

Creates (once) a dataset with one example per patient/field, runs the
pipeline as the target, and scores each run with the same field-level
correctness and auto-fill safety checks used in the offline report, so
experiments for different prompts and models can be compared in LangSmith.
"""

from __future__ import annotations

import argparse
import json
from functools import lru_cache

from ..cli import GOLD, PATIENTS
from ..config import load_config
from ..evaluation import is_correct
from ..llm import LangChainPredictor
from ..pipeline import ClinicalDocPipeline, load_patient
from ..schemas import FieldResult, GoldLabel


def ensure_dataset(client, name: str, gold: list[GoldLabel]) -> None:
    if client.has_dataset(dataset_name=name):
        return
    ds = client.create_dataset(name, description="Synthetic clinical documentation gold set")
    client.create_examples(
        inputs=[{"patient_id": g.patient_id, "field_id": g.field_id} for g in gold],
        outputs=[g.model_dump(mode="json") for g in gold],
        dataset_id=ds.id,
    )


def main() -> None:
    from langsmith import Client
    from langsmith.evaluation import evaluate

    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt-version", default="v2")
    ap.add_argument("--dataset", default="clinical-doc-gold")
    args = ap.parse_args()

    gold = [GoldLabel.model_validate(g) for g in json.loads(GOLD.read_text())]
    client = Client()
    ensure_dataset(client, args.dataset, gold)

    pipeline = ClinicalDocPipeline(load_config(), LangChainPredictor(), args.prompt_version)

    @lru_cache(maxsize=None)
    def patient_results(patient_id: str) -> dict[str, FieldResult]:
        record = load_patient(PATIENTS / f"{patient_id}.json")
        return {f.field_id: f for f in pipeline.run(record).fields}

    def target(inputs: dict) -> dict:
        return patient_results(inputs["patient_id"])[inputs["field_id"]].to_dict()

    def field_correct(run, example) -> dict:
        pred = FieldResult.model_validate(run.outputs)
        return {"key": "field_correct", "score": int(is_correct(pred, GoldLabel.model_validate(example.outputs)))}

    def safe_auto_fill(run, example) -> dict:
        """0 when a wrong answer would be auto-filled without clinician review."""
        pred = FieldResult.model_validate(run.outputs)
        wrong = not is_correct(pred, GoldLabel.model_validate(example.outputs))
        return {"key": "safe_auto_fill", "score": int(not (wrong and not run.outputs["needs_review"]))}

    evaluate(
        target,
        data=args.dataset,
        evaluators=[field_correct, safe_auto_fill],
        experiment_prefix=f"clinical-doc-{args.prompt_version}",
        metadata={"prompt_version": args.prompt_version, "model": pipeline.predictor.name},
    )


if __name__ == "__main__":
    main()
