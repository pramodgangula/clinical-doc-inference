import json
from datetime import date

from clinical_doc.evaluation import ErrorType, classify_error, evaluate, list_prf, Scored
from clinical_doc.llm import ReplayPredictor
from clinical_doc.schemas import FieldResult, GoldLabel, Status

from .conftest import ROOT

A, INS, CON = Status.ANSWERED, Status.INSUFFICIENT_EVIDENCE, Status.CONFLICTING_EVIDENCE


def pred(status, answer=None, notes=("n1",), dates=()):
    return FieldResult(patient_id="p", field_id="f", status=status, answer=answer,
                       retrieved_note_ids=list(notes), evidence_dates=list(dates))


def gold(status, answer=None, notes=("n1",)):
    return GoldLabel(patient_id="p", field_id="f", status=status, answer=answer, evidence_note_ids=list(notes))


def test_error_taxonomy():
    g = gold(A, "PARTIAL_ASSIST")
    assert classify_error(pred(A, "PARTIAL_ASSIST"), g, []) is None
    assert classify_error(pred(A, "X"), gold(INS, notes=()), []) == ErrorType.OVER_INFERENCE
    assert classify_error(pred(INS, notes=("other",)), g, []) == ErrorType.MISSING_CONTEXT
    assert classify_error(pred(INS), g, []) == ErrorType.INCORRECT_ABSTENTION
    old = pred(A, "INDEPENDENT", dates=[date(2026, 8, 1)])
    assert classify_error(old, g, [date(2026, 10, 20)]) == ErrorType.TEMPORAL_CONTEXT
    assert classify_error(pred(A, "DEPENDENT", dates=[date(2026, 10, 20)]), g, [date(2026, 10, 20)]) == ErrorType.WRONG_VALUE
    assert classify_error(pred(INS), gold(CON), []) == ErrorType.WRONG_ABSTAIN_REASON


def test_list_items_are_normalised_to_generic_names():
    s = Scored(pred(A, ["Metoprolol succinate", "Lisinopril"]), gold(A, ["metoprolol"]), None)
    assert list_prf([s]) == {"precision": 0.5, "recall": 1.0, "f1": 0.667}


def _report(config, patient, local_pipeline, fixture_path, version):
    records = {p: patient(p) for p in ("p001", "p002", "p003")}
    pipe = local_pipeline(ReplayPredictor(fixture_path(version)), version)
    results = [f for r in records.values() for f in pipe.run(r).fields]
    g = [GoldLabel.model_validate(x) for x in json.loads((ROOT / "data/labels/gold.json").read_text())]
    return evaluate(results, g, records, config)


def test_v2_improves_on_v1_and_no_wrong_answer_is_auto_filled(config, patient, local_pipeline, fixture_path):
    v1 = _report(config, patient, local_pipeline, fixture_path, "v1").overall
    v2 = _report(config, patient, local_pipeline, fixture_path, "v2").overall
    assert v2["accuracy"] > v1["accuracy"]
    assert v1["error_types"] == {"OVER_INFERENCE": 3, "TEMPORAL_CONTEXT": 2, "MISSING_CONTEXT": 1}
    assert v2["error_types"] == {"INCORRECT_ABSTENTION": 1, "MISSING_CONTEXT": 1}
    assert v1["errors_auto_filled"] == 2 and v2["errors_auto_filled"] == 0
