# Evaluation: prompt v2, model `replay:v2`

| Metric | Value |
|---|---|
| Field-level accuracy | 87% (13/15) |
| Retrieval recall (gold evidence note retrieved) | 92% |
| Abstention precision / recall / F1 | 60% / 100% / 75% |
| Auto-fill rate | 67% |
| **Auto-fill precision** | **100%** |
| Errors caught by review routing | 2 of 2 |
| **Errors auto-filled** | **0** |

## Per field

| Field | Accuracy | Precision | Recall | F1 | Errors auto-filled |
|---|---|---|---|---|---|
| bathing | 67% | 50% | 50% | 50% | 0 |
| dressing | 100% | 100% | 100% | 100% | 0 |
| ambulation | 100% | 100% | 100% | 100% | 0 |
| medication_management | 67% | 50% | 50% | 50% | 0 |
| current_medications | 100% | 100% | 100% | 100% | 0 |

## Errors

| Patient | Field | Error type | Predicted | Gold | Routed to review | Flags |
|---|---|---|---|---|---|---|
| p001 | medication_management | INCORRECT_ABSTENTION | INSUFFICIENT_EVIDENCE | INDEPENDENT | yes | EVIDENCE_IN_CONTEXT_IGNORED |
| p003 | bathing | MISSING_CONTEXT | INSUFFICIENT_EVIDENCE | DEPENDENT | yes | POSSIBLE_RETRIEVAL_MISS |
