# PR125 return, R40 folder integration and remaining-output review

26 November 2026, Asia/Hong_Kong.

Start with [R40 findings and use](../../../docs/unmatched-output-r40.md). This evidence bundle connects three distinct operations: verification of the returned R39 replay, fresh inference through the new optional folder filter, and source-pixel review of the remaining other-unmatched outputs.

## Verified result

The PR123 candidate at confidence 0.13 and ordinary NMS IoU 0.70 returned 737 primary predictions. The optional extra NMS on final clipped boxes, fixed IoU strictly greater than 0.70, retained 728. Fresh CPU folder inference on all 20 original development images reproduced the complete saved R39 numeric arrays and original prediction IDs exactly. The retained and suppressed records reconstruct every original output, and all 20 overlay files and CSV counts agree.

| Reference | Matched | Missed | Unmatched | Duplicate subset | Other unmatched |
| --- | ---: | ---: | ---: | ---: | ---: |
| Accepted R38 | 643 | 11 | 85 | 21 | 64 |
| Historical frozen666 | 631 | 35 | 97 | 22 | 75 |

All original matched identities and corner outcomes are preserved in both comparisons. These references share the same photographs. No new training, changed annotation, automatically selected checkpoint or independent test result is introduced.

## Evidence files

| File | Content |
| --- | --- |
| [pr125-return-audit.json](pr125-return-audit.json) | Immutable PR125 archive identity, 12 member hashes, ZIP integrity and source/notebook provenance. |
| [runtime-verification.json](runtime-verification.json) | Fresh real CPU execution, source/checkpoint hashes, runtime, exact all-20 output comparison, reconstruction and two-reference scores. |
| [upstream-update-audit.json](upstream-update-audit.json) | Live check of all three upstream branches; none changed from the recorded 7 November snapshot. |
| [triage/REVIEW.md](triage/REVIEW.md) | Source-pixel interpretation of all 64 other-unmatched predictions. |
| [triage/review.json](triage/review.json) | Complete per-case review with original pIDs, exact geometry, reference associations, exclusions, source hashes and uncertainty. |
| [triage/mechanical.json](triage/mechanical.json) | The scoring/geometry stage before manual interpretation. |
| [triage/representative_cases.png](triage/representative_cases.png) | Six readable original/annotated comparisons. |

The complete retained numeric outputs already reside in the exact hashed [R39 cache](../pr123-r39-diagnosis-20261126/clipped-nms-predictions.jsonl); the [primary cache](../pr123-r37-20261126/predictions/candidate.jsonl) retains all 737. Runtime timing/overlay artifacts were generated and checked locally; their hashes are in the fresh-run verification record. They are not copied over those prior caches.

## Full 64-case crop gallery

Each sheet pairs unmarked original pixels with accepted-reference and prediction overlays. The JSON ledger maps each case to its sheet and exact original-image crop. Cyan is accepted reference material, magenta is the prediction. Prediction numbers remain original primary IDs.

| Cases | Sheet | Cases | Sheet |
| --- | --- | --- | --- |
| 1–6 | [01](triage/sheet_01.png) | 7–12 | [02](triage/sheet_02.png) |
| 13–18 | [03](triage/sheet_03.png) | 19–24 | [04](triage/sheet_04.png) |
| 25–30 | [05](triage/sheet_05.png) | 31–36 | [06](triage/sheet_06.png) |
| 37–42 | [07](triage/sheet_07.png) | 43–48 | [08](triage/sheet_08.png) |
| 49–54 | [09](triage/sheet_09.png) | 55–60 | [10](triage/sheet_10.png) |
| 61–64 | [11](triage/sheet_11.png) | | |

Full-frame context is retained in [01](triage/context_01.png), [02](triage/context_02.png), [03](triage/context_03.png) and [04](triage/context_04.png). The [087–090 top sequence](triage/top_sequence.png) establishes the white top fixture; [053–056 context](triage/top_sequence_053_056.png) preserves the uncertain clipped strips. Nearby frames are used for structural context, not unverified coordinate transfer.

## Reproduction

The [folder guide](../../../docs/predict-pose-folder.md#optional-nms-after-framework-clipping) gives the explicit PR123 checkpoint command and original-image payload requirement. The existing focused check is:

```bash
python scripts/check_predict_pose_folder.py
```

It passes all 18 checks, including unchanged disabled behavior, retained-ID JSON/CSV/overlay consistency, and exact replay of the 20-image R39 filter result. A separate fresh real-model run and independent source review also passed; these are recorded in the main R40 guide.

The crop assembly scripts take paths explicitly. From the repository root, with original image payloads present and Pillow available:

```bash
python results/evaluation/pr123-r40-unmatched-20261126/triage/prepare_triage.py \
  --root . --image-root data/images/val --output /path/to/new-r40-review
python results/evaluation/pr123-r40-unmatched-20261126/triage/finish_triage.py \
  --root . --image-root data/images/val --output /path/to/new-r40-review
python results/evaluation/pr123-r40-unmatched-20261126/triage/render_representative.py \
  --root . --image-root data/images/val --output /path/to/new-r40-review
```

The finish script checks the immutable cache and reference hashes before attaching the saved manual judgments. Running it reproduces this recorded review; it does not independently re-adjudicate physical ownership or approve new labels. Nearest-neighbor enlargement preserves the source-pixel evidence. The reviewed cases remain development data.
