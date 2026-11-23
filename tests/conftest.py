from pathlib import Path

import pytest

from clinical_doc.config import load_config
from clinical_doc.pipeline import ClinicalDocPipeline, load_patient
from clinical_doc.preprocessing import build_chunks
from clinical_doc.retrieval import PatientIndex

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def config():
    return load_config()


@pytest.fixture(scope="session")
def patient():
    return lambda pid: load_patient(ROOT / "data" / "patients" / f"{pid}.json")


@pytest.fixture(scope="session")
def index(patient):
    cache = {}

    def get(pid):
        if pid not in cache:
            rec = patient(pid)
            cache[pid] = PatientIndex(build_chunks(rec), rec.assessment_date)
        return cache[pid]

    return get


@pytest.fixture
def fixture_path():
    return lambda version: ROOT / "data" / "fixtures" / f"{version}.json"


class ScriptedPredictor:
    """Returns queued outputs in order and records each request."""

    name = "scripted"

    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.requests = []

    def predict(self, request):
        self.requests.append(request)
        out = self.outputs.pop(0)
        if isinstance(out, Exception):
            raise out
        return out


@pytest.fixture
def scripted():
    return ScriptedPredictor


@pytest.fixture
def local_pipeline(config):
    return lambda predictor, version="v2": ClinicalDocPipeline(config, predictor, version, engine="local")
