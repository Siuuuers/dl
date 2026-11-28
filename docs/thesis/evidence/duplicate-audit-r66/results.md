# Duplicate and unmatched output audit

The saved outputs reproduce exactly before three separate output-removal probes. The probes reveal tradeoffs; no model or threshold is promoted.

`mutual_view_support` requires the existing mutual original/reflected box link. `additional_nms_060` and `additional_nms_050` suppress completed outputs at prediction-pair IoU strictly above 0.60 or 0.50. They do not replace native NMS. All retained coordinates and confidence values remain unchanged.

## Reviewed-reference results

| Method | Probe | Misses | Extras | Duplicate subset | Both within 5 px | Both within 10 px |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Historical retained method | unchanged | 18 | 51 | 5 | 569/583 | 583/583 |
| Historical retained method | mutual_view_support | 23 | 27 | 2 | 569/583 | 583/583 |
| Historical retained method | additional_nms_060 | 22 | 37 | 0 | 569/583 | 583/583 |
| Historical retained method | additional_nms_050 | 23 | 26 | 0 | 569/583 | 583/583 |
| PR157 native control | unchanged | 18 | 49 | 4 | 568/583 | 583/583 |
| PR157 native control | mutual_view_support | 22 | 25 | 2 | 568/583 | 583/583 |
| PR157 native control | additional_nms_060 | 22 | 35 | 0 | 568/583 | 583/583 |
| PR157 native control | additional_nms_050 | 23 | 26 | 0 | 568/583 | 583/583 |
| PR157 QFL candidate | unchanged | 12 | 83 | 17 | 570/583 | 582/583 |
| PR157 QFL candidate | mutual_view_support | 16 | 46 | 5 | 570/583 | 582/583 |
| PR157 QFL candidate | additional_nms_060 | 14 | 58 | 4 | 570/583 | 582/583 |
| PR157 QFL candidate | additional_nms_050 | 16 | 42 | 0 | 570/583 | 582/583 |
| PR158 original-label arm | unchanged | 47 | 85 | 6 | 541/583 | 566/583 |
| PR158 original-label arm | mutual_view_support | 47 | 59 | 3 | 541/583 | 566/583 |
| PR158 original-label arm | additional_nms_060 | 48 | 62 | 2 | 541/583 | 566/583 |
| PR158 original-label arm | additional_nms_050 | 48 | 55 | 1 | 541/583 | 566/583 |
| PR158 reviewed-label arm | unchanged | 15 | 51 | 13 | 572/583 | 583/583 |
| PR158 reviewed-label arm | mutual_view_support | 19 | 27 | 6 | 572/583 | 583/583 |
| PR158 reviewed-label arm | additional_nms_060 | 16 | 32 | 2 | 572/583 | 583/583 |
| PR158 reviewed-label arm | additional_nms_050 | 19 | 23 | 1 | 572/583 | 583/583 |

The maximum-cardinality reference-matching bound is reported separately in the machine-readable matrix. It is a diagnostic upper bound, not a replacement scorer or a usable inference rule.

The full matrix retains all three reference versions. Paired changes identify every lost/gained reference and reassigned prediction; availability groups distinguish zero, one and two known corners. The baseline error ledger does not label unmatched regions as physical negatives.

## Interpretation limits

- All three references describe the same 20 repeatedly examined development photographs.
- An evaluator extra or duplicate is not an independent physical background or object-identity verdict.
- Historically excluded visible partial fronts remain excluded; the evaluator has no ignore-region protocol.
- Extra suppression acts on already filtered completed two-view outputs. It does not replace or replay native NMS and cannot recover discarded proposals.
- Retained boxes, scores and averaged coordinates are unchanged. Fresh evaluation matching can select a different retained record for a reference.
- The reflected image is another network response to the same photograph, not independent camera evidence.
- Maximum one-to-one matching uses reference geometry only as a cardinality bound; it changes neither the primary scorer nor inference.
- A decrease in extras with unchanged complete-pair success can still lose incomplete-corner fronts.
- No label changes, training, model inference, threshold selection, or deployment promotion are performed.
