# Duplicate and unmatched cassette outputs

28 November 2026, Asia/Hong_Kong. Saved-output diagnostic; no new inference or training.

**Decision:** keep the current recognition defaults. The reviewed-label candidate remains worth investigating, but none of the tested output restrictions improves all measured outcomes. Additional suppression at IoU 0.60 removes 19 extras, including 11 duplicates, while losing one previously matched incomplete front. Complete-pair scores do not reveal that loss. The [four-cell CPU notebook](../../notebooks/baomai_duplicate_audit_r66.ipynb) reproduces the full comparison from fixed saved predictions.

## What unmatched and duplicate mean

The unchanged evaluator visits predictions by descending confidence and assigns each to the greatest-overlap unassigned same-class reference when IoU is at least 0.50. An unmatched reference is a miss. An unmatched prediction is an extra. An extra overlapping an already matched reference by at least 0.50 is a duplicate; duplicates are included in extras.

These are reference-agreement categories. An extra can be another prediction of the same front, a poorly placed box on a real front, a response on the rack, an omitted reference target, an intentional exclusion or an unresolved region. The [focused image review](annotation-review-r65.md) preserves those distinctions. Model scores alone cannot decide which physical explanation is correct.

NMS compares predictions with predictions. Evaluation compares predictions with reference boxes. Two predictions can each overlap a reference by at least 0.50 while overlapping each other by less than 0.70. The 13 reviewed-label duplicates have prediction-to-matched-prediction IoU approximately 0.475–0.700; their survival is consistent with the declared suppression rule. Twelve have their greatest reference overlap with an incomplete-corner front.

## Fixed experiment

The diagnostic loads three archives with exact size, SHA-256 and member-inventory checks: the historical reference replay, PR157's classification comparison and PR158's training-label comparison. Archived native outputs are passed through the unchanged processing and evaluator first. All saved prediction, association, suppression, case and per-image structures reproduce exactly, covering 840 archived image scores.

Five two-view output sets are compared under each of the three reference versions. Each output set receives four separate conditions:

1. Unchanged completed outputs.
2. Retain only outputs with an existing mutual original/reflected box association at IoU at least 0.50.
3. Add hard suppression of final boxes at prediction-pair IoU strictly above 0.60.
4. Add hard suppression of final boxes at prediction-pair IoU strictly above 0.50.

The probes are never combined. All retained boxes, scores and averaged corner coordinates stay unchanged. Confidence ordering uses original output index to break ties. View association is not recomputed after deletion, but evaluation matching is. Additional suppression is downstream of the existing native and postclip suppression at 0.70; it cannot restore already discarded proposals and is not a replacement-NMS experiment. These illustrative settings were examined on reused development data; no optimum is claimed or deployed.

## Reviewed-label candidate

The reviewed reference contains 654 fronts: 583 with complete corner annotations and 71 with fewer than two known corners.

| Output rule | Misses | Extras | Duplicate subset | Detected incomplete fronts | Both within 5 px | Both within 10 px |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Unchanged | 15 | 51 | 13 | 56/71 | 572/583 | 583/583 |
| Require reflected partner | 19 | 27 | 6 | 52/71 | 572/583 | 583/583 |
| Additional NMS 0.60 | 16 | 32 | 2 | 55/71 | 572/583 | 583/583 |
| Additional NMS 0.50 | 19 | 23 | 1 | 52/71 | 572/583 | 583/583 |

The 0.60 rule loses `r50/new_000056/R014`, which has one known corner. The 0.50 rule also loses `r50/new_000009/R051` and `r50/new_000089/R042` and `R043`, all with one known corner. The reflected-partner rule instead loses `r50/new_000009/R049` and `R051`, `r50/new_000056/R016`, and `r50/new_000087/R048`; `R016` has no known corner and the other three have one. No reviewed-reference match is gained by these restrictions for this arm.

The counts concern lost reference matches; they are not a new independent physical adjudication of each photograph. They nevertheless establish a task-level measurement issue: restricting evaluation to complete corner pairs would hide changes in the required partial-front detection coverage.

Reflection support is a consistency signal from the same photograph. It adds no independent camera observation and cannot establish that an unsupported output is false. Both the detector and its reflected run can agree on the same error or disagree on a real partial front.

## Other methods and reference versions

The [complete 60-row matrix](evidence/duplicate-audit-r66/matrix.csv) retains all five methods, three references and four conditions. The [readable results](evidence/duplicate-audit-r66/results.md) show every reviewed-reference row. No returned training arm is confused with the historical retained checkpoint.

Under the reviewed reference, additional suppression at 0.50 changes the native-control arm from 18/49/4 misses/extras/duplicates to 23/26/0, and QFL from 12/83/17 to 16/42/0. Their complete-pair counts stay unchanged. Thus zero scored duplicates can coexist with additional missed targets.

The older references remain sensitivity checks on the same photographs. Complete-pair counts do not remain unchanged in every older-reference condition. In particular, reflected support gains one five-pixel success for the original-label arm but loses one for the reviewed-label arm under both older references. Fresh matching can select a different retained prediction even though none of the retained coordinates was modified. This is why the original evaluator is rerun and the detailed changes are retained.

## Matching cannot repair the retained boxes

An independent maximum-cardinality bipartite calculation asks how many matches the same prediction and reference boxes permit under the unchanged same-class IoU-at-least-0.50 rule. Its maximum equals the existing greedy count in all 1200 method/reference/condition/image records. The reviewed-reference unchanged totals are 636, 636, 642, 607 and 639 for the historical, native-control, QFL, original-label and reviewed-label outputs, respectively.

This is a diagnostic upper bound using reference geometry, not a usable inference algorithm. Changing only the matching algorithm cannot recover another detection from these retained boxes. Changing eligibility thresholds or reference definitions would be a different experiment. Matching alternatives must not be chosen to obtain more favourable corner scores.

## Next work and evidence

Review object identity and extent in the documented duplicate clusters and the lost partial-front cases. Preserve the general requirement to detect each identifiable front once. A prior explicit exclusion of a visible fragment must be reconciled before a future reference revision; it must not silently become a verified background example. The historical references stay frozen.

The [primary-source research](research-duplicates-r66.md) explains TIDE's error categories, pre-NMS Soft-NMS, localization-quality ranking and the already attempted one-to-one head. A future Soft-NMS probe needs the decoded proposals before suppression. A learned ranker needs a specific score-ordering hypothesis and training control. These are conditional options, not completed improvements.

The [execution evidence](evidence/duplicate-audit-r66/README.md) records the actual notebook cells, source pin, archive checks, deterministic saved-output replay and result package. The [frozen assignment audit](score-assignment-audit.md) separately finds no missing assignment among the 3999 training fronts and does not justify an immediate box-only loss change. No annotation or application default is changed by either diagnostic.
