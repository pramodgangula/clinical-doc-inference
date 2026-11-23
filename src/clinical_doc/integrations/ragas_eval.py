"""Ragas faithfulness and context recall over pipeline outputs.

    export OPENAI_API_KEY=...
    python -m clinical_doc.integrations.ragas_eval --predictor replay --prompt-version v2

Faithfulness checks that the answer and rationale are supported by the
retrieved context. Context recall checks that the context contains what the
reference answer needs. These complement the deterministic field-level metrics.
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
    from langchain_openai import ChatOpenAI
    from ragas import EvaluationDataset, evaluate
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import Faithfulness, LLMContextRecall

    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt-version", default="v2")
    ap.add_argument("--predictor", choices=["replay", "openai"], default="replay")
    ap.add_argument("--judge-model", default="gpt-4o")
    args = ap.parse_args()

    predictor = (ReplayPredictor(FIXTURES / f"{args.prompt_version}.json") if args.predictor == "replay"
                 else LangChainPredictor())
    pipeline = ClinicalDocPipeline(load_config(), predictor, args.prompt_version)
    gold = [GoldLabel.model_validate(g) for g in json.loads(GOLD.read_text())]
    samples = collect_samples(pipeline, PATIENTS, gold)

    dataset = EvaluationDataset.from_list([
        {"user_input": s.question, "retrieved_contexts": s.contexts, "response": s.response, "reference": s.reference}
        for s in samples
    ])
    judge = LangchainLLMWrapper(ChatOpenAI(model=args.judge_model, temperature=0))
    result = evaluate(dataset=dataset, metrics=[Faithfulness(), LLMContextRecall()], llm=judge)
    print(result)


if __name__ == "__main__":
    main()
