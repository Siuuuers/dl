# Four restored partial fronts: reference-only replay

Four reviewed partial-front rows are appended to a separately versioned development reference. The six methods retain exactly the same saved outputs. This is a change in measured agreement with the reference, not a model improvement.

| Method | Reference | Fronts | Misses | Extras | Duplicate subset | Other extras | Both within 5 px | Both within 10 px |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Supplied original model | r50 | 654 | 34 | 61 | 5 | 56 | 555/583 | 577/583 |
| Supplied original model | r67 | 658 | 37 | 60 | 5 | 55 | 555/583 | 577/583 |
| Historical retained method | r50 | 654 | 18 | 51 | 5 | 46 | 569/583 | 583/583 |
| Historical retained method | r67 | 658 | 21 | 50 | 5 | 45 | 569/583 | 583/583 |
| PR157 native control | r50 | 654 | 18 | 49 | 4 | 45 | 568/583 | 583/583 |
| PR157 native control | r67 | 658 | 21 | 48 | 4 | 44 | 568/583 | 583/583 |
| PR157 QFL candidate | r50 | 654 | 12 | 83 | 17 | 66 | 570/583 | 582/583 |
| PR157 QFL candidate | r67 | 658 | 15 | 82 | 17 | 65 | 570/583 | 582/583 |
| PR158 original-label arm | r50 | 654 | 47 | 85 | 6 | 79 | 541/583 | 566/583 |
| PR158 original-label arm | r67 | 658 | 50 | 84 | 6 | 78 | 541/583 | 566/583 |
| PR158 reviewed-label arm | r50 | 654 | 15 | 51 | 13 | 38 | 572/583 | 583/583 |
| PR158 reviewed-label arm | r67 | 658 | 18 | 50 | 13 | 37 | 572/583 | 583/583 |

## Evidence and interpretation

The primary scorer replays all saved r50 image scores exactly. New evaluation matching is then performed against r67. The reference grows from 654 to 658 fronts; its known-corner strata are 18 with zero, 57 with one and 583 with two. Every inherited r50 row remains byte-identical and at the same row position.

The machine-readable matrix retains zero/one/two-known-corner detection outcomes. Separate ledgers identify all restored-front outcomes, any reassigned retained target, and any unchanged prediction whose matched/duplicate/other-extra status changes. An unchanged complete-pair count does not establish that the restored partial fronts were detected.

## Existing fixed filters after reference restoration

This separate eight-row sensitivity check uses only the PR158 reviewed-label arm. It reuses the three fixed r66 restrictions, independently: require its existing mutual reflected-view link, or add suppression at prediction-pair IoU strictly above 0.60 or 0.50. This does not replace native NMS, change retained coordinates, recompute view associations, or select a threshold.

| Reference | Fixed restriction | Misses | Extras | Duplicate subset | Partial fronts matched | Restored fronts matched |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| r50 | unchanged | 15 | 51 | 13 | 56/71 | Not in reference |
| r67 | unchanged | 18 | 50 | 13 | 57/75 | 1/4 |
| r50 | mutual_view_support | 19 | 27 | 6 | 52/71 | Not in reference |
| r67 | mutual_view_support | 22 | 26 | 6 | 53/75 | 1/4 |
| r50 | additional_nms_060 | 16 | 32 | 2 | 55/71 | Not in reference |
| r67 | additional_nms_060 | 19 | 31 | 2 | 56/75 | 1/4 |
| r50 | additional_nms_050 | 19 | 23 | 1 | 52/71 | Not in reference |
| r67 | additional_nms_050 | 22 | 22 | 1 | 53/75 | 1/4 |

The survival ledger separately records whether each restored front's unchanged supporting output remains after filtering. A reference-only match change must not be described as a newly generated detection.

## Limits

- This changes the evaluated reference population, not learned recognition or model outputs.
- All 20 photographs are repeatedly examined development images, not independent test captures.
- Three other excluded candidates remain held for ownership or corner uncertainty; the evaluator has no ignore-region mechanism.
- The corrected inventory is a bounded four-case inclusion decision, not certification that every visible front is annotated.
- Restored coordinates retain historical nominal precision; soft image edges do not establish subpixel physical accuracy.
- Complete-pair success omits incomplete-corner fronts; report detection strata and restored-front outcomes alongside it.
- Evaluation matches are recomputed from unchanged predictions. Reference changes can reassign an output or change its duplicate category.
- No training labels, historical references, scorer rules, confidence thresholds or suppression settings are changed.
