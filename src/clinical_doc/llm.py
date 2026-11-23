"""LLM predictors.

* LangChainPredictor: ChatOpenAI via LangChain with structured output bound to
  the LLMFieldOutput schema. Used for real runs.
* ReplayPredictor: returns saved outputs from a fixture file, so tests and the
  evaluation run offline and deterministically.

Both satisfy the same Predictor protocol, so models and prompt versions can be
compared on the same evaluation set.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from .schemas import LLMFieldOutput


@dataclass
class PredictRequest:
    system: str
    user: str
    metadata: dict = field(default_factory=dict)  # patient_id, field_id, prompt_version, attempt


class Predictor(Protocol):
    name: str

    def predict(self, request: PredictRequest) -> Any:
        """Return a dict (or JSON string) matching LLMFieldOutput."""


class LangChainPredictor:
    def __init__(self, model: str | None = None, temperature: float = 0.0):
        from langchain_openai import ChatOpenAI  # optional dependency

        self.model = model or os.environ.get("OPENAI_MODEL", "gpt-4o")
        self.name = f"openai:{self.model}"
        llm = ChatOpenAI(model=self.model, temperature=temperature)
        self._structured = llm.with_structured_output(LLMFieldOutput, method="function_calling")

    def predict(self, request: PredictRequest) -> dict:
        from langchain_core.messages import HumanMessage, SystemMessage

        out = self._structured.invoke([
            SystemMessage(content=request.system),
            HumanMessage(content=request.user),
        ])
        return out.model_dump(mode="json")


class ReplayPredictor:
    """Replays recorded outputs keyed by '<patient_id>/<field_id>'."""

    def __init__(self, fixture_path: Path):
        self.fixture_path = Path(fixture_path)
        self.name = f"replay:{self.fixture_path.stem}"
        self._outputs = json.loads(self.fixture_path.read_text())

    def predict(self, request: PredictRequest) -> Any:
        key = f"{request.metadata['patient_id']}/{request.metadata['field_id']}"
        if key not in self._outputs:
            raise KeyError(f"no recorded output for {key}")
        return self._outputs[key]
