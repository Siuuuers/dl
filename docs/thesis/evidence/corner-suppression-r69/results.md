# Corner agreement as a condition for additional suppression

This fixed saved-output experiment compares unchanged detections, the existing additional box-only IoU 0.60 control, and the same suppression qualified by two accepted corresponding corners within five pixels. All selection uses predictions alone. The R68 reflected-output union is excluded.

| Method | Variant | r67 misses | Extras | Duplicate subset | Pair within 5 px | Pair within 10 px |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| historical_retained | unchanged | 21 | 50 | 5 | 569/583 | 583/583 |
| historical_retained | additional_nms_060 | 25 | 36 | 0 | 569/583 | 583/583 |
| historical_retained | corner_qualified_nms_060 | 21 | 47 | 5 | 569/583 | 583/583 |
| pr157_native | unchanged | 21 | 48 | 4 | 568/583 | 583/583 |
| pr157_native | additional_nms_060 | 25 | 34 | 0 | 568/583 | 583/583 |
| pr157_native | corner_qualified_nms_060 | 21 | 44 | 4 | 568/583 | 583/583 |
| pr157_qfl | unchanged | 15 | 82 | 17 | 570/583 | 582/583 |
| pr157_qfl | additional_nms_060 | 17 | 57 | 4 | 570/583 | 582/583 |
| pr157_qfl | corner_qualified_nms_060 | 15 | 76 | 16 | 570/583 | 582/583 |
| pr158_original_labels | unchanged | 50 | 84 | 6 | 541/583 | 566/583 |
| pr158_original_labels | additional_nms_060 | 51 | 61 | 2 | 541/583 | 566/583 |
| pr158_original_labels | corner_qualified_nms_060 | 51 | 81 | 6 | 541/583 | 566/583 |
| pr158_reviewed_labels | unchanged | 18 | 50 | 13 | 572/583 | 583/583 |
| pr158_reviewed_labels | additional_nms_060 | 19 | 31 | 2 | 572/583 | 583/583 |
| pr158_reviewed_labels | corner_qualified_nms_060 | 18 | 44 | 11 | 572/583 | 583/583 |

The full matrix retains five methods and four references. Fresh matching can change which surviving prediction represents a target. The ledgers preserve removed outputs, their suppressors, target gains, losses, rematches and confidence-qualified known-corner errors. Outputs keep their original coordinates and scores.

The two thresholds were fixed before the first candidate outcome measurement. This is an exploratory two-corner rule inspired by pose comparison in RMPE, with no learned pose-distance parameters or coordinate merging. Agreement between predictions can be shared error. The same development photographs support all reference comparisons.
