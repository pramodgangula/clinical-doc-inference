"""Field configuration loaded from configs/fields.yaml."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, model_validator

DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "fields.yaml"


class FieldSpec(BaseModel):
    field_id: str
    question: str
    answer_type: Literal["categorical", "list"]
    allowed_values: list[str] = []
    guidance: str = ""
    sections: list[str]
    lookback_days: int = 120
    require_topic_evidence: bool = True
    keywords: list[str] = []

    @model_validator(mode="after")
    def _categorical_needs_values(self) -> "FieldSpec":
        if self.answer_type == "categorical" and not self.allowed_values:
            raise ValueError(f"categorical field {self.field_id} needs allowed_values")
        return self


class Condition(BaseModel):
    field: str
    answer: str


class ConsistencyRule(BaseModel):
    name: str
    when: Condition
    and_: Condition


class RetrievalSettings(BaseModel):
    top_k: int = 4
    stale_evidence_tolerance_days: int = 14


class Config(BaseModel):
    fields: dict[str, FieldSpec]
    consistency_rules: list[ConsistencyRule] = []
    retrieval: RetrievalSettings = RetrievalSettings()

    @model_validator(mode="after")
    def _rules_reference_real_values(self) -> "Config":
        for rule in self.consistency_rules:
            for cond in (rule.when, rule.and_):
                spec = self.fields.get(cond.field)
                if spec is None or cond.answer not in spec.allowed_values:
                    raise ValueError(f"rule {rule.name}: invalid condition {cond}")
        return self


def load_config(path: Path = DEFAULT_CONFIG) -> Config:
    raw = yaml.safe_load(Path(path).read_text())
    return Config(
        fields={fid: FieldSpec(field_id=fid, **spec) for fid, spec in raw["fields"].items()},
        consistency_rules=[
            ConsistencyRule(name=r["name"], when=r["when"], and_=r["and"])
            for r in raw.get("consistency_rules", [])
        ],
        retrieval=RetrievalSettings(**raw.get("retrieval", {})),
    )
