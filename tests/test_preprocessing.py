from clinical_doc.embeddings import HashingEmbedder
from clinical_doc.preprocessing import build_chunks, clean_text, group_sentences, split_sections


def test_clean_text_removes_signatures_bullets_and_extra_spaces():
    raw = "FUNCTIONAL STATUS:\n-  Pt   ambulates with walker.\n\n\n\nElectronically signed by: RN"
    assert clean_text(raw) == "FUNCTIONAL STATUS:\nPt ambulates with walker."


def test_split_sections_maps_headings_to_section_types():
    text = "Intro line.\nDISCHARGE MEDICATIONS:\nStarted metoprolol.\nA/P:\nFollow up."
    assert split_sections(text) == [
        ("GENERAL", "Intro line."),
        ("MEDICATIONS", "Started metoprolol."),
        ("ASSESSMENT_PLAN", "Follow up."),
    ]


def test_unheaded_note_is_grouped_by_topic_and_keeps_continuations():
    sents = [
        "Patient states she manages all of her own medications.",
        "She walks around the apartment with a cane.",
        "She was pleasant and conversational.",
        "Vital signs stable.",
    ]
    groups = group_sentences(sents, HashingEmbedder())
    assert [t for t, _ in groups] == ["MEDICATIONS", "FUNCTIONAL_STATUS", "VITALS"]
    # the pronoun continuation stays with the sentence it refers to
    assert groups[1][1] == sents[1:3]


def test_chunks_carry_metadata_and_structured_data_is_dated(patient):
    chunks = build_chunks(patient("p002"))
    by_id = {c.chunk_id: c for c in chunks}
    meds = by_id["p002-structured-c01"]
    assert meds.source == "STRUCTURED" and meds.section == "MEDICATIONS"
    assert meds.date.isoformat() == "2026-08-05"
    pt = by_id["p002-n03-c00"]
    assert (pt.doc_type, pt.section, pt.date.isoformat()) == ("PT_EVALUATION", "FUNCTIONAL_STATUS", "2026-10-21")
    assert all(c.patient_id == "p002" for c in chunks)
