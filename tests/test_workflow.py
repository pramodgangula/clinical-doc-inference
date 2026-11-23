import pytest

from clinical_doc.graph import build_field_graph
from clinical_doc.llm import ReplayPredictor
from clinical_doc.pipeline import ClinicalDocPipeline
from clinical_doc.schemas import Flag, Status

GOOD = {"status": "ANSWERED", "answer": "SUPERVISION",
        "evidence": [{"chunk_id": "p001-n01-c01", "quote": "Daughter stands nearby for safety during showers"}]}


def test_replay_pipeline_answers_every_field(config, patient, fixture_path, local_pipeline):
    result = local_pipeline(ReplayPredictor(fixture_path("v2"))).run(patient("p002"))
    by_field = {f.field_id: f for f in result.fields}
    assert list(by_field) == list(config.fields)
    assert by_field["ambulation"].answer == "PARTIAL_ASSIST"
    assert not by_field["ambulation"].needs_review


def test_prompt_shows_dates_newest_first_and_field_guidance(patient, scripted, local_pipeline):
    pred = scripted([GOOD])
    local_pipeline(pred).run(patient("p002"), field_ids=["ambulation"])
    user = pred.requests[0].user
    assert user.index("2026-10-21") < user.index("2026-08-05")
    assert "Contact guard" in user


def test_v1_and_v2_use_different_system_prompts(patient, scripted, local_pipeline):
    p1, p2 = scripted([GOOD]), scripted([GOOD])
    local_pipeline(p1, "v1").run(patient("p001"), field_ids=["bathing"])
    local_pipeline(p2, "v2").run(patient("p001"), field_ids=["bathing"])
    assert "INSUFFICIENT_EVIDENCE" not in p1.requests[0].system
    assert "INSUFFICIENT_EVIDENCE" in p2.requests[0].system


def test_invalid_output_is_retried_with_feedback(patient, scripted, local_pipeline):
    pred = scripted(["{not json", GOOD])
    res = local_pipeline(pred).run(patient("p001"), field_ids=["bathing"]).fields[0]
    assert res.answer == "SUPERVISION" and res.attempts == 2
    assert "PREVIOUS RESPONSE WAS INVALID" in pred.requests[1].user


def test_provider_error_then_give_up_routes_to_review(patient, scripted, local_pipeline):
    pred = scripted([RuntimeError("timeout"), {"status": "ANSWERED", "answer": None}])
    res = local_pipeline(pred).run(patient("p001"), field_ids=["bathing"]).fields[0]
    assert res.flags == [Flag.INVALID_OUTPUT]
    assert res.status == Status.INSUFFICIENT_EVIDENCE and res.needs_review
    assert len(pred.requests) == 2


def test_unknown_prompt_version_rejected(config, patient, scripted):
    with pytest.raises(ValueError):
        ClinicalDocPipeline(config, scripted([GOOD]), "v9", engine="local").run(patient("p001"))


def test_langgraph_engine_matches_local_engine(config, patient, fixture_path):
    pytest.importorskip("langgraph")
    rec = patient("p003")
    local = ClinicalDocPipeline(config, ReplayPredictor(fixture_path("v2")), "v2", engine="local").run(rec)
    graph = ClinicalDocPipeline(config, ReplayPredictor(fixture_path("v2")), "v2", engine="langgraph").run(rec)
    assert [f.to_dict() for f in local.fields] == [f.to_dict() for f in graph.fields]


def test_build_field_graph_requires_langgraph_only_when_called():
    assert callable(build_field_graph)
