import pytest

from clinical_doc.retrieval import BM25


def ids(index, config, pid, field):
    return [s.chunk.chunk_id for s in index(pid).retrieve(config.fields[field], top_k=config.retrieval.top_k)]


def test_bm25_scores_only_matching_docs():
    bm = BM25([["walker", "gait"], ["pillbox"]])
    assert bm.scores(["walker"], [0, 1])[1] == 0 < bm.scores(["walker"], [0, 1])[0]


def test_metadata_filter_restricts_sections(index, config):
    field = config.fields["current_medications"]
    assert {index("p002").chunks[i].section for i in index("p002").candidates(field)} == {"MEDICATIONS"}


def test_metadata_filter_respects_lookback(index, config):
    field = config.fields["bathing"].model_copy(update={"lookback_days": 30})
    dates = {index("p002").chunks[i].date.isoformat() for i in index("p002").candidates(field)}
    assert "2026-08-05" not in dates


@pytest.mark.parametrize("pid,field,expected", [
    ("p001", "bathing", "p001-n01-c01"),
    ("p001", "medication_management", "p001-n02-c01"),
    ("p002", "ambulation", "p002-n03-c00"),
    ("p002", "current_medications", "p002-n02-c01"),
    ("p003", "medication_management", "p003-n02-c00"),
])
def test_retrieval_finds_annotated_evidence(index, config, pid, field, expected):
    assert expected in ids(index, config, pid, field)


def test_reranker_puts_newer_documentation_first(index, config):
    assert ids(index, config, "p002", "ambulation")[0] == "p002-n03-c00"


def test_known_gap_social_section_not_searched_for_bathing(index, config):
    """Documents the retrieval miss found in evaluation (see README)."""
    assert "p003-n02-c00" not in ids(index, config, "p003", "bathing")
