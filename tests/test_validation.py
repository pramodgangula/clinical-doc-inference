import pytest
from pydantic import ValidationError

from clinical_doc.schemas import EvidenceRef, FieldResult, Flag, LLMFieldOutput, Status
from clinical_doc.validation import apply_consistency_rules, quote_in, validate_field


def out(status, answer=None, *evidence):
    return LLMFieldOutput(status=status, answer=answer,
                          evidence=[EvidenceRef(chunk_id=c, quote=q) for c, q in evidence])


def check(config, index, pid, field, output, retrieved=None):
    idx = index(pid)
    retrieved = set(retrieved) if retrieved is not None else set(idx.by_id)
    return validate_field(output, config.fields[field], idx, retrieved)


def test_quote_matching_ignores_case_and_punctuation():
    assert quote_in("daughter stands NEARBY, for safety", "Daughter stands nearby for safety during showers")
    assert not quote_in("", "anything")


def test_clean_answer_has_no_flags(config, index):
    o = out(Status.ANSWERED, "SUPERVISION", ("p001-n01-c01", "Daughter stands nearby for safety during showers"))
    assert check(config, index, "p001", "bathing", o) == []


def test_invalid_value_and_missing_evidence(config, index):
    o = out(Status.ANSWERED, "MOSTLY_OK")
    assert set(check(config, index, "p001", "bathing", o)) == {Flag.INVALID_VALUE, Flag.MISSING_EVIDENCE}


def test_list_field_requires_list_answer(config, index):
    o = out(Status.ANSWERED, "amlodipine", ("p001-n02-c01", "Taking amlodipine 5 mg daily"))
    assert check(config, index, "p001", "current_medications", o) == [Flag.INVALID_VALUE]


def test_fabricated_quote_is_ungrounded(config, index):
    o = out(Status.ANSWERED, "INDEPENDENT", ("p001-n01-c01", "Pt walks a mile daily"))
    assert check(config, index, "p001", "ambulation", o) == [Flag.UNGROUNDED_EVIDENCE]


def test_evidence_not_shown_to_model(config, index):
    o = out(Status.ANSWERED, "SUPERVISION", ("p001-n01-c01", "Daughter stands nearby for safety during showers"))
    assert check(config, index, "p001", "bathing", o, retrieved=["p001-n02-c01"]) == [Flag.EVIDENCE_OUTSIDE_CONTEXT]


def test_over_inference_from_another_activity_is_flagged(config, index):
    o = out(Status.ANSWERED, "SUPERVISION", ("p001-n01-c01", "Daughter stands nearby for safety"))
    assert check(config, index, "p001", "dressing", o) == [Flag.EVIDENCE_OFF_TOPIC]


def test_stale_evidence_when_newer_documentation_exists(config, index):
    o = out(Status.ANSWERED, "INDEPENDENT", ("p002-n01-c00", "Patient ambulates independently throughout the home"))
    assert check(config, index, "p002", "ambulation", o) == [Flag.STALE_EVIDENCE]


def test_abstention_with_topic_in_context_is_flagged(config, index):
    o = out(Status.INSUFFICIENT_EVIDENCE)
    assert check(config, index, "p001", "medication_management", o) == [Flag.EVIDENCE_IN_CONTEXT_IGNORED]


def test_abstention_with_topic_only_outside_context_suggests_retrieval_miss(config, index):
    o = out(Status.INSUFFICIENT_EVIDENCE)
    flags = check(config, index, "p003", "bathing", o, retrieved=["p003-n01-c02"])
    assert flags == [Flag.POSSIBLE_RETRIEVAL_MISS]


def test_true_absence_is_not_flagged(config, index):
    assert check(config, index, "p001", "dressing", out(Status.INSUFFICIENT_EVIDENCE)) == []


def test_schema_enforces_status_answer_contract():
    with pytest.raises(ValidationError):
        LLMFieldOutput(status=Status.ANSWERED, answer=None)
    with pytest.raises(ValidationError):
        LLMFieldOutput(status=Status.CONFLICTING_EVIDENCE, answer="INDEPENDENT")


def test_consistency_rule_flags_both_fields(config):
    rs = [FieldResult(patient_id="p", field_id="bathing", status=Status.ANSWERED, answer="INDEPENDENT"),
          FieldResult(patient_id="p", field_id="dressing", status=Status.ANSWERED, answer="DEPENDENT"),
          FieldResult(patient_id="p", field_id="ambulation", status=Status.ANSWERED, answer="INDEPENDENT")]
    apply_consistency_rules(rs, config.consistency_rules)
    assert [r.flags for r in rs] == [[Flag.CONSISTENCY_CONFLICT], [Flag.CONSISTENCY_CONFLICT], []]
