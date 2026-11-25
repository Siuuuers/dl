# PR114 fixed inference and separate reference replays

This record evaluates the returned R33 native training pair from [PR114](https://github.com/Siuuuers/dl/pull/114). It preserves the original prediction order and compares each checkpoint against distinct annotation references. The later R34 replay uses the same predictions; it is **not** an R34 training result.

## Provenance

- PR114 artifact revision: `e520cc1e8aa878d0c7d127c7222dfbfbcf1f69c9`.
- Published R33 source/notebook revision: `d4631e07af44efc3c35d84f57f55c10c6ff06a65` ([PR115](https://github.com/Siuuuers/dl/pull/115)). The R33 runtime source was the earlier unpublished assembly `d4b905dc0ed7728815108fed04486bc4215f7a2b`; the published source tree was verified separately by the source publication workflow.
- PR35 retained checkpoint artifact revision: `8e8bb99cabcb44ed0ec0acd14db911aa44013fd8` ([PR35](https://github.com/Siuuuers/dl/pull/35)).
- PR113 was also verified to resolve the user's older magenta pIDs ([PR113](https://github.com/Siuuuers/dl/pull/113), revision `257ce13706586607447cc8c4d948a35def4a9eb7`). Those pIDs belong to its seed-47, partial-box-factor-2 candidate and must not be transferred to PR114 predictions.

`archive-audit.json` records validated archive and checkpoint hashes. Archive checks included wrapper hashes, ZIP integrity, each declared member size and hash, and safe paths. Both PR114 native run verifications passed. All 157 original image hashes were checked against the source manifest before inference. Models were not retrained.

The three committed prediction caches were generated in one CPU environment: Ultralytics 8.4.182, PyTorch 2.10.0+cpu, FP32, four threads. All 20 validation images, including three empty scenes, are present. Fixed inference uses size 1280, rectangular preprocessing, batch 1, native one-to-many output, NMS IoU 0.70, box confidence 0.13, max detections 300, and the repository's quantize=32 convention. Cache headers record model hashes, runtime, and settings. The GPU training runtime remains separately recorded in the archive audit.

## Results

Each row uses its stated annotation reference. Pair counts are objects with both locatable corners returned within the specified original-image pixel radius. Denominators include eligible missed objects.

| Reference | Model | Misses | Unmatched | Duplicate subset | Pairs <=5 px | Pairs <=10 px |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Frozen 666, 584 pairs | PR114 R31-label control | 36 | 108 | 30 | 516 | 573 |
| Frozen 666, 584 pairs | PR114 R33-label candidate | 35 | 105 | 30 | 516 | 573 |
| Frozen 666, 584 pairs | PR35 selected | 18 | 80 | 32 | 533 | 582 |
| Historical 663, 584 pairs | PR114 R31-label control | 32 | 107 | 30 | 516 | 573 |
| Historical 663, 584 pairs | PR114 R33-label candidate | 32 | 105 | 30 | 516 | 573 |
| Historical 663, 584 pairs | PR35 selected | 15 | 80 | 32 | 533 | 582 |
| R33 review, 661 fronts / 583 pairs | PR114 R31-label control | 15 | 92 | 32 | 571 | 582 |
| R33 review, 661 fronts / 583 pairs | PR114 R33-label candidate | 15 | 90 | 30 | 571 | 582 |
| R33 review, 661 fronts / 583 pairs | PR35 selected | 45 | 112 | 17 | 547 | 569 |
| R34 review, 661 fronts / 583 pairs | PR114 R31-label control | 16 | 93 | 32 | 570 | 582 |
| R34 review, 661 fronts / 583 pairs | PR114 R33-label candidate | 16 | 91 | 30 | 570 | 582 |
| R34 review, 661 fronts / 583 pairs | PR35 selected | 45 | 112 | 15 | 546 | 569 |

On R33, all 15 candidate misses are partial fronts: four have zero locatable corners and eleven have one. All 583 two-corner fronts match. Partial-front matching is 63/78. There are 17 matched rows with a missing available corner or corner error greater than five pixels. Nine missed fronts have no output with IoU >=0.10; six have an overlapping output below the matching threshold. These are geometric triage groups, not physical adjudications.

On the amended R34 reference, the unchanged PR114 candidate additionally misses `new_000088` row 51, and candidate p57 becomes other-unmatched. `new_000059` row 2 additionally enters the corner-error catalogue. The candidate now has 16 misses, all partial, and 18 corner-error rows. `r34-reference-delta.json` lists the exact identity changes. Score changes here describe amended labels, not model learning.

## Definitions and limits

The preserved scorer matches predictions in descending confidence to unmatched same-class reference boxes at IoU >=0.50. A missed front is an unmatched reference; an unmatched output is an unassigned prediction. A duplicate is an unmatched prediction overlapping an already matched reference at IoU >=0.50. Thus **90 unmatched = 30 duplicates + 60 other unmatched** for the R33 candidate. Duplicates must not be added to unmatched totals again. Other-unmatched outputs are not automatically hallucinations: poor localization, excluded surfaces, and omitted or uncertain references can also produce this category.

An additional maximum-cardinality pass used the same class/IoU eligibility graph. It found zero additional possible matches for every model/image/reference combination (three models, 20 images, five separately preserved references). The reported scorer was not changed. This rules out a recoverable count deficit from greedy assignment in this replay, but it does not establish the physical correctness of all equally sized assignments.

Only reference visibility 2 supplies a locatable corner target. Visibility 0 does not supply a coordinate target; a predicted hidden corner is allowed but is not scored against the placeholder origin. Endpoint confidence is 0.50. Five and ten pixels are review radii, not calibrated manipulation tolerances. Stored image y grows downward; top is ymin, bottom is ymax.

The R33/R34 references have been repeatedly reviewed during development and are not an independent unseen-session test. The frozen 666 and historical 663 comparisons remain separate. PR35 remains the retained model under the frozen protocol; this record does not promote the R33 candidate. No new NMS sweep, threshold search, loss-weight experiment, or training run was performed.

## Files and reproduction

- `predictions/*.jsonl`: original fixed-inference output order, with hashes and runtime headers.
- `comparison.json`: all three models across five references, all per-image counts, matching diagnostics, and complete candidate R33/R34 error catalogues.
- `r34-reference-delta.json`: added/removed candidate error identities caused by the R34 reference amendment.
- `pdf-inventory.json`: every R33 PDF entry, source-image hash, crop, category, and page; includes the delivered PDF hash.
- `pdf-qa.json`: 36-page render review and catalogue completeness checks; the first 34 pages are pixel-identical to the initial R33 document at the checked render scale.
- `archive-audit.json`: verified source archive and checkpoint provenance.

Replay the saved predictions without loading a model:

```bash
python3 scripts/score_pr114_review.py
```

Render the R33 diagnostic PDF after materializing the original images through Git LFS:

```bash
python3 scripts/render_pr114_failures.py --output /tmp/pr114-r33-misses-and-errors.pdf
```

The first 34 PDF pages preserve the R33 catalogue: 15 missed fronts, 17 corner-error rows, and all 90 unmatched outputs, with cyan reference and magenta candidate overlays. Two separately numbered appendix pages give the R34 cached-rescore summary and original cropped views of the two newly flagged cases. The earlier R33 pages and their 1–34 numbering are unchanged. PDFs, images, and weights are not added to this evaluation directory.
