"""Command line interface.

    python -m clinical_doc.cli run data/patients/p002.json
    python -m clinical_doc.cli evaluate --prompt-version v2
    python -m clinical_doc.cli compare                 # v1 vs v2 side by side
    python -m clinical_doc.cli --predictor openai --embedder openai evaluate
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from .config import load_config
from .embeddings import HashingEmbedder, OpenAIEmbedder
from .evaluation import evaluate, render_comparison, render_markdown
from .llm import LangChainPredictor, ReplayPredictor
from .pipeline import ClinicalDocPipeline, load_patient
from .prompts import LATEST_PROMPT_VERSION, SYSTEM_PROMPTS
from .schemas import GoldLabel

ROOT = Path(__file__).resolve().parents[2]
PATIENTS = ROOT / "data" / "patients"
GOLD = ROOT / "data" / "labels" / "gold.json"
FIXTURES = ROOT / "data" / "fixtures"
REPORTS = ROOT / "reports"


def make_pipeline(args, prompt_version: str) -> ClinicalDocPipeline:
    predictor = (ReplayPredictor(FIXTURES / f"{prompt_version}.json") if args.predictor == "replay"
                 else LangChainPredictor())
    embedder = OpenAIEmbedder() if args.embedder == "openai" else HashingEmbedder()
    return ClinicalDocPipeline(load_config(), predictor, prompt_version, embedder=embedder, engine=args.engine)


def run_eval(args, prompt_version: str):
    pipeline = make_pipeline(args, prompt_version)
    records = {p.stem: load_patient(p) for p in sorted(PATIENTS.glob("*.json"))}
    results = [f for rec in records.values() for f in pipeline.run(rec).fields]
    gold = [GoldLabel.model_validate(g) for g in json.loads(GOLD.read_text())]
    return pipeline, evaluate(results, gold, records, pipeline.config)


def cmd_run(args) -> None:
    pipeline = make_pipeline(args, args.prompt_version)
    result = pipeline.run(load_patient(Path(args.patient)))
    print(json.dumps([f.to_dict() for f in result.fields], indent=2))


def cmd_evaluate(args) -> None:
    pipeline, report = run_eval(args, args.prompt_version)
    md = render_markdown(report, pipeline.predictor.name, args.prompt_version)
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / f"eval_{args.prompt_version}.md").write_text(md)
    (REPORTS / f"eval_{args.prompt_version}.json").write_text(json.dumps(
        {"model": pipeline.predictor.name, "prompt_version": args.prompt_version, "engine": pipeline.engine,
         "overall": report.overall, "per_field": report.per_field}, indent=2))
    print(md)


def cmd_compare(args) -> None:
    reports = {v: run_eval(args, v)[1] for v in args.versions}
    md = "# Prompt version comparison\n\n" + render_comparison(reports)
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "comparison.md").write_text(md)
    print(md)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="clinical_doc")
    p.add_argument("--predictor", choices=["replay", "openai"], default="replay")
    p.add_argument("--embedder", choices=["hashing", "openai"], default="hashing")
    p.add_argument("--engine", choices=["auto", "langgraph", "local"], default="auto")
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="command", required=True)

    r = sub.add_parser("run", help="answer every field for one patient")
    r.add_argument("patient")
    r.add_argument("--prompt-version", choices=list(SYSTEM_PROMPTS), default=LATEST_PROMPT_VERSION)
    r.set_defaults(func=cmd_run)

    e = sub.add_parser("evaluate", help="score all patients against gold labels")
    e.add_argument("--prompt-version", choices=list(SYSTEM_PROMPTS), default=LATEST_PROMPT_VERSION)
    e.set_defaults(func=cmd_evaluate)

    c = sub.add_parser("compare", help="compare prompt versions on the same gold set")
    c.add_argument("--versions", nargs="+", default=list(SYSTEM_PROMPTS))
    c.set_defaults(func=cmd_compare)

    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING)
    args.func(args)


if __name__ == "__main__":
    main()
