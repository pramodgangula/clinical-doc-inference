"""DeepEval faithfulness and contextual recall over pipeline outputs.

    export OPENAI_API_KEY=...
    python -m clinical_doc.integrations.deepeval_eval --predictor replay --prompt-version v2
"""

from __future__ import annotations

import argparse
import json

from ..cli import FIXTURES, GOLD, PATIENTS
from ..config import load_config
from ..llm import LangChainPredictor, ReplayPredictor
from ..pipeline import ClinicalDocPipeline
from ..schemas import GoldLabel
from . import collect_samples


def main() -> None:
    from deepeval import evaluate
    from deepeval.metrics import ContextualRecallMetric, FaithfulnessMetric
    from deepeval.test_case import LLMTestCase

    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt-version", default="v2")
    ap.add_argument("--predictor", choices=["replay", "openai"], default="replay")
    ap.add_argument("--threshold", type=float, default=0.8)
    args = ap.parse_args()

    predictor = (ReplayPredictor(FIXTURES / f"{args.prompt_version}.json") if args.predictor == "replay"
                 else LangChainPredictor())
    pipeline = ClinicalDocPipeline(load_config(), predictor, args.prompt_version)
    gold = [GoldLabel.model_validate(g) for g in json.loads(GOLD.read_text())]

    test_cases = [
        LLMTestCase(
            input=f"[{s.patient_id}/{s.field_id}] {s.question}",
            actual_output=s.response,
            expected_output=s.reference,
            retrieval_context=s.contexts,
        )
        for s in collect_samples(pipeline, PATIENTS, gold)
    ]
    evaluate(
        test_cases=test_cases,
        metrics=[FaithfulnessMetric(threshold=args.threshold), ContextualRecallMetric(threshold=args.threshold)],
    )


if __name__ == "__main__":
    main()
