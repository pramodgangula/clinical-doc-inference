# Evaluation: prompt v1, model `replay:v1`

| Metric | Value |
|---|---|
| Field-level accuracy | 60% (9/15) |
| Retrieval recall (gold evidence note retrieved) | 92% |
| Abstention precision / recall / F1 | – / 0% / – |
| Auto-fill rate | 73% |
| **Auto-fill precision** | **82%** |
| Errors caught by review routing | 4 of 6 |
| **Errors auto-filled** | **2** |

## Per field

| Field | Accuracy | Precision | Recall | F1 | Errors auto-filled |
|---|---|---|---|---|---|
| bathing | 67% | 50% | 50% | 50% | 0 |
| dressing | 33% | 25% | 25% | 25% | 0 |
| ambulation | 67% | 33% | 50% | 40% | 0 |
| medication_management | 67% | 50% | 67% | 56% | 1 |
| current_medications | 67% | 90% | 100% | 95% | 1 |

## Errors

| Patient | Field | Error type | Predicted | Gold | Routed to review | Flags |
|---|---|---|---|---|---|---|
| p001 | dressing | OVER_INFERENCE | SUPERVISION | INSUFFICIENT_EVIDENCE | yes | EVIDENCE_OFF_TOPIC |
| p002 | ambulation | TEMPORAL_CONTEXT | INDEPENDENT | PARTIAL_ASSIST | yes | STALE_EVIDENCE |
| p002 | current_medications | TEMPORAL_CONTEXT | Lisinopril, Furosemide, Atorvastatin, Metoprolol succinate | furosemide, metoprolol, atorvastatin | **NO** | – |
| p003 | bathing | MISSING_CONTEXT | INDEPENDENT | DEPENDENT | yes | EVIDENCE_OFF_TOPIC |
| p003 | dressing | OVER_INFERENCE | INDEPENDENT | INSUFFICIENT_EVIDENCE | yes | EVIDENCE_OFF_TOPIC |
| p003 | medication_management | OVER_INFERENCE | INDEPENDENT | CONFLICTING_EVIDENCE | **NO** | – |
