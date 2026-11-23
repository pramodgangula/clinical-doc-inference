"""Field-level evaluation against clinician-annotated gold labels.

Three layers, measured separately so a failure can be traced to its component:

1. Retrieval: did the retrieved context contain the note the annotator used?
2. Answers: field-level correctness; precision/recall/F1 per field
   (macro over labels for categorical fields, micro over items for list fields);
   precision/recall/F1 of abstention.
3. Safety: of the answers that would be auto-filled, how many were wrong?

Every error is categorised so prompt, retrieval and validation work can be
targeted at the largest category.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import date, timedelta
from enum import Enum

from .config import Config
from .schemas import FieldResult, GoldLabel, PatientRecord, Status


class ErrorType(str, Enum):
    OVER_INFERENCE = "OVER_INFERENCE"            # answered without support (gold abstains)
    MISSING_CONTEXT = "MISSING_CONTEXT"          # gold evidence never reached the model
    TEMPORAL_CONTEXT = "TEMPORAL_CONTEXT"        # wrong answer from superseded documentation
    INCORRECT_ABSTENTION = "INCORRECT_ABSTENTION"  # abstained although evidence was in context
    WRONG_VALUE = "WRONG_VALUE"                  # other misinterpretation of the evidence
    WRONG_ABSTAIN_REASON = "WRONG_ABSTAIN_REASON"  # abstained, but insufficient vs conflicting mixed up


def normalize_item(item: str) -> str:
    """'Metoprolol succinate' -> 'metoprolol' (generic name, salt dropped)."""
    return item.strip().lower().split()[0] if item.strip() else ""


def label_of(status: Status, answer) -> str:
    return answer if status == Status.ANSWERED and isinstance(answer, str) else status.value


def is_correct(pred: FieldResult, gold: GoldLabel) -> bool:
    if pred.status != gold.status:
        return False
    if gold.status != Status.ANSWERED:
        return True
    if isinstance(gold.answer, list):
        return isinstance(pred.answer, list) and {normalize_item(a) for a in pred.answer} == {normalize_item(a) for a in gold.answer}
    return pred.answer == gold.answer


def classify_error(pred: FieldResult, gold: GoldLabel, gold_dates: list[date], tolerance_days: int = 14) -> ErrorType | None:
    if is_correct(pred, gold):
        return None
    pred_answered = pred.status == Status.ANSWERED
    if gold.status != Status.ANSWERED:
        return ErrorType.OVER_INFERENCE if pred_answered else ErrorType.WRONG_ABSTAIN_REASON
    if gold.evidence_note_ids and not set(gold.evidence_note_ids) & set(pred.retrieved_note_ids):
        return ErrorType.MISSING_CONTEXT
    if not pred_answered:
        return ErrorType.INCORRECT_ABSTENTION
    if pred.evidence_dates and gold_dates and min(pred.evidence_dates) < min(gold_dates) - timedelta(days=tolerance_days):
        return ErrorType.TEMPORAL_CONTEXT
    return ErrorType.WRONG_VALUE


@dataclass
class Scored:
    pred: FieldResult
    gold: GoldLabel
    error: ErrorType | None

    @property
    def correct(self) -> bool:
        return self.error is None


def _ratio(n: int, d: int) -> float | None:
    return round(n / d, 3) if d else None


def _prf(tp: int, fp: int, fn: int) -> tuple[float | None, float | None, float | None]:
    p, r = _ratio(tp, tp + fp), _ratio(tp, tp + fn)
    f1 = None if p is None or r is None else (round(2 * p * r / (p + r), 3) if p + r else 0.0)
    return p, r, f1


def categorical_prf(scored: list[Scored]) -> dict:
    """Macro precision/recall/F1 over labels (allowed values + abstention statuses)."""
    labels = sorted({label_of(s.gold.status, s.gold.answer) for s in scored}
                    | {label_of(s.pred.status, s.pred.answer) for s in scored})
    ps, rs, fs = [], [], []
    for lab in labels:
        tp = sum(label_of(s.pred.status, s.pred.answer) == lab == label_of(s.gold.status, s.gold.answer) for s in scored)
        fp = sum(label_of(s.pred.status, s.pred.answer) == lab != label_of(s.gold.status, s.gold.answer) for s in scored)
        fn = sum(label_of(s.gold.status, s.gold.answer) == lab != label_of(s.pred.status, s.pred.answer) for s in scored)
        p, r, f = _prf(tp, fp, fn)
        ps.append(p or 0.0), rs.append(r or 0.0), fs.append(f or 0.0)
    n = len(labels) or 1
    return {"precision": round(sum(ps) / n, 3), "recall": round(sum(rs) / n, 3), "f1": round(sum(fs) / n, 3)}


def list_prf(scored: list[Scored]) -> dict:
    """Micro precision/recall/F1 over extracted items."""
    tp = fp = fn = 0
    for s in scored:
        pred = {normalize_item(a) for a in s.pred.answer} if isinstance(s.pred.answer, list) else set()
        gold = {normalize_item(a) for a in s.gold.answer} if isinstance(s.gold.answer, list) else set()
        tp, fp, fn = tp + len(pred & gold), fp + len(pred - gold), fn + len(gold - pred)
    p, r, f = _prf(tp, fp, fn)
    return {"precision": p, "recall": r, "f1": f}


def summary_metrics(scored: list[Scored]) -> dict:
    auto = [s for s in scored if not s.pred.needs_review]
    errors = [s for s in scored if not s.correct]
    pred_abs = {id(s) for s in scored if s.pred.status != Status.ANSWERED}
    gold_abs = {id(s) for s in scored if s.gold.status != Status.ANSWERED}
    abs_p, abs_r, abs_f = _prf(len(pred_abs & gold_abs), len(pred_abs - gold_abs), len(gold_abs - pred_abs))
    with_evidence = [s for s in scored if s.gold.evidence_note_ids]
    return {
        "fields": len(scored),
        "accuracy": _ratio(len(scored) - len(errors), len(scored)),
        "retrieval_recall": _ratio(
            sum(bool(set(s.gold.evidence_note_ids) & set(s.pred.retrieved_note_ids)) for s in with_evidence),
            len(with_evidence),
        ),
        "abstention_precision": abs_p,
        "abstention_recall": abs_r,
        "abstention_f1": abs_f,
        "auto_fill_rate": _ratio(len(auto), len(scored)),
        "auto_fill_precision": _ratio(sum(s.correct for s in auto), len(auto)),
        "errors": len(errors),
        "errors_caught_by_review": sum(s.pred.needs_review for s in errors),
        "errors_auto_filled": sum(not s.pred.needs_review for s in errors),
        "error_types": dict(Counter(s.error.value for s in errors)),
    }


@dataclass
class EvalReport:
    scored: list[Scored]
    overall: dict
    per_field: dict


def evaluate(
    results: list[FieldResult],
    gold: list[GoldLabel],
    records: dict[str, PatientRecord],
    config: Config,
) -> EvalReport:
    gold_by_key = {(g.patient_id, g.field_id): g for g in gold}
    scored = []
    for pred in results:
        g = gold_by_key.get((pred.patient_id, pred.field_id))
        if g is None:
            raise KeyError(f"no gold label for {pred.patient_id}/{pred.field_id}")
        rec = records[pred.patient_id]
        note_dates = {n.note_id: n.date for n in rec.notes} | {f"{rec.patient_id}-structured": rec.structured.as_of}
        gold_dates = [note_dates[n] for n in g.evidence_note_ids if n in note_dates]
        scored.append(Scored(pred, g, classify_error(pred, g, gold_dates, config.retrieval.stale_evidence_tolerance_days)))

    by_field: dict[str, list[Scored]] = defaultdict(list)
    for s in scored:
        by_field[s.pred.field_id].append(s)
    per_field = {}
    for fid, items in by_field.items():
        m = summary_metrics(items)
        m.update(list_prf(items) if config.fields[fid].answer_type == "list" else categorical_prf(items))
        per_field[fid] = m
    return EvalReport(scored=scored, overall=summary_metrics(scored), per_field=per_field)


# ---------------------------------------------------------------- report


def _pct(v) -> str:
    return "–" if v is None else f"{v:.0%}"


def render_markdown(report: EvalReport, model: str, prompt_version: str) -> str:
    o = report.overall
    lines = [
        f"# Evaluation: prompt {prompt_version}, model `{model}`",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Field-level accuracy | {_pct(o['accuracy'])} ({o['fields'] - o['errors']}/{o['fields']}) |",
        f"| Retrieval recall (gold evidence note retrieved) | {_pct(o['retrieval_recall'])} |",
        f"| Abstention precision / recall / F1 | {_pct(o['abstention_precision'])} / {_pct(o['abstention_recall'])} / {_pct(o['abstention_f1'])} |",
        f"| Auto-fill rate | {_pct(o['auto_fill_rate'])} |",
        f"| **Auto-fill precision** | **{_pct(o['auto_fill_precision'])}** |",
        f"| Errors caught by review routing | {o['errors_caught_by_review']} of {o['errors']} |",
        f"| **Errors auto-filled** | **{o['errors_auto_filled']}** |",
        "",
        "## Per field",
        "",
        "| Field | Accuracy | Precision | Recall | F1 | Errors auto-filled |",
        "|---|---|---|---|---|---|",
    ]
    for fid, m in report.per_field.items():
        lines.append(f"| {fid} | {_pct(m['accuracy'])} | {_pct(m['precision'])} | {_pct(m['recall'])} "
                     f"| {_pct(m['f1'])} | {m['errors_auto_filled']} |")
    lines += ["", "## Errors", "",
              "| Patient | Field | Error type | Predicted | Gold | Routed to review | Flags |",
              "|---|---|---|---|---|---|---|"]
    for s in report.scored:
        if s.correct:
            continue
        fmt = lambda st, a: (", ".join(a) if isinstance(a, list) else a) if st == Status.ANSWERED else st.value
        lines.append(
            f"| {s.pred.patient_id} | {s.pred.field_id} | {s.error.value} | {fmt(s.pred.status, s.pred.answer)} "
            f"| {fmt(s.gold.status, s.gold.answer)} | {'yes' if s.pred.needs_review else '**NO**'} "
            f"| {', '.join(f.value for f in s.pred.flags) or '–'} |"
        )
    return "\n".join(lines) + "\n"


def render_comparison(reports: dict[str, EvalReport]) -> str:
    keys = [("accuracy", "Field-level accuracy"), ("retrieval_recall", "Retrieval recall"),
            ("abstention_f1", "Abstention F1"), ("auto_fill_precision", "Auto-fill precision"),
            ("errors_auto_filled", "Errors auto-filled")]
    versions = list(reports)
    lines = ["| Metric | " + " | ".join(versions) + " |", "|---|" + "---|" * len(versions)]
    for key, label in keys:
        vals = [reports[v].overall[key] for v in versions]
        lines.append(f"| {label} | " + " | ".join(str(x) if isinstance(x, int) else _pct(x) for x in vals) + " |")
    all_types = sorted({t for r in reports.values() for t in r.overall["error_types"]})
    for t in all_types:
        lines.append(f"| {t} | " + " | ".join(str(reports[v].overall["error_types"].get(t, 0)) for v in versions) + " |")
    return "\n".join(lines) + "\n"
