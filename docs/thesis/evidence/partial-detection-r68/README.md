# Partial-front diagnosis and fixed reflected-output union

Completed CPU analysis, using saved outputs and unchanged labels. The [four-cell notebook](../../../../notebooks/baomai_partial_detection_r68.ipynb) ran all four actual cells unchanged from published implementation [0e1f545a2cf33022b86bf316866b08cacfa55477](https://github.com/Siuuuers/dl/commit/0e1f545a2cf33022b86bf316866b08cacfa55477). Execution used local IPython, a clean source checkout and five preseeded, hash-checked archives. Hosted Colab, live downloads and its browser download interface were not tested. See the [execution receipt](notebook-execution.json), [executed notebook](executed-notebook.ipynb) and [package manifest](package-manifest.json).

The [fixed protocol](protocol.json) was recorded after focused stage inspection and a bounded support check, before scoring the full union. This experiment changes no training, model weights, reference labels, confidence thresholds or existing coordinates. No method is promoted.

## Three restored misses

The [r67 lineage](reference-lineage.json.gz) identifies these previously excluded partial fronts:

| Historical identity | Current r67 row | Nominal target size | Historical original-view stage |
|---|---|---:|---|
| r34/new_000009/R056 | new_000009/R055 | 18 × 31 px | Eligible box below confidence cutoff |
| r37/new_000009/R052 | new_000009/R056 | 15 × 14 px | No decoded box reaches matching IoU 0.50 |
| r37/new_000053/R001 | new_000053/R022 | 47 × 10 px | Eligible boxes below confidence cutoff |

Only the historical adapted checkpoint's original view has a complete decoded tensor before confidence and native NMS. The archived tensor contains 19,320 proposals for each 1280 × 720 photograph; its padded network input is 736 × 1280. The [stage diagnostic](stage-diagnostic.json) reuses the existing r63 continuity checks and checks matching against r67 on the two relevant photographs.

| Target | Best decoded IoU | Best-box confidence | Best-box size | IoU-eligible boxes in complete pool |
|---|---:|---:|---:|---:|
| new_000009/R055 | 0.608489 | 0.00215215 | 26.808 × 31.255 px | 1 |
| new_000009/R056 | 0.409067 | 0.33002254 | 26.030 × 13.897 px | 0 |
| new_000053/R022 | 0.690867 | 0.00009662 | 64.124 × 10.609 px | 2 |

A confidence cutoff can only remove existing candidates. In this fixed original-view pool it cannot recover the lower009 target, because no box reaches the unchanged matching IoU. For the other two targets, a cutoff low enough to admit the highest-scoring eligible box admits at least 947 or 778 decoded candidates in the respective photograph before NMS. These are finite-pool bounds, not measured postprocessing results, proposed operating cutoffs or estimates of false outputs.

Newer checkpoints have more limited saved stages. PR157 caches were collected above confidence 0.001 after native NMS, with a 300-output cap; the r60 and PR158 native caches already apply confidence above 0.25. [Available native support](available-native-support.json) records the limits and cap flags explicitly. A missing candidate in those caches cannot establish whether raw geometry, raw confidence, native suppression or the cap caused its absence. [Final geometry](final-focused-geometry.json) reports the greatest positive-overlap saved box, or null when none overlaps; it does not assign a neighboring prediction to the physical target.

The QFL reflected view has a lower009 box with IoU 0.504753 and score 0.290007. It is already mutually associated with an original box of score 0.408069; their box IoU is 0.733817. Its `reflected_pair_unavailable` status concerns corner averaging, not the existence of the box link. The unpaired-only union therefore excludes it. Even an unrestricted union at NMS 0.70 would suppress it behind the original. The [paired-box counterexample](paired-counterexample.json) preserves both records and the exact association.

## One fixed union experiment

Other missed fronts have eligible unpaired reflected support, as recorded by the [preliminary support gate](union-support-gate.json). This justifies a separate all-front experiment. Retain every completed original output, including its existing associated-corner averaging; append every reflected output whose index is unused by any mutual box link; then apply the existing score-ranked postclip NMS at IoU strictly above 0.70. All scores and corner records remain unchanged. No reference is used to select additions, and no new final output cap is introduced.

The [40-row matrix](matrix.csv) compares baseline and union for five methods under r50, r67, supplied664 and frozen666. Its r67 results are:

| Method | Misses before → after | Extras before → after | Duplicate subset before → after | Both corners ≤5 px | Both corners ≤10 px |
|---|---:|---:|---:|---:|---:|
| Historical retained | 21 → 16 | 50 → 62 | 5 → 9 | 569 → 569 | 583 → 583 |
| PR157 native control | 21 → 15 | 48 → 60 | 4 → 9 | 568 → 568 | 583 → 583 |
| PR157 QFL | 15 → 11 | 82 → 115 | 17 → 33 | 570 → 570 | 582 → 582 |
| PR158 original labels | 50 → 48 | 84 → 93 | 6 → 8 | 541 → 541 | 566 → 567 |
| PR158 reviewed labels | 18 → 10 | 50 → 60 | 13 → 18 | 572 → 572 | 583 → 583 |

Every pair result uses the same 583 complete-reference pairs. All three focused restorations remain missed. The union offers a coverage/extra tradeoff and adds no evidence of improvement on new photographs.

For the reviewed-label arm, 23 reflected outputs are appended and 20 survive. Two original outputs, both previously other extras under r67, are suppressed. Eight previously missed reference fronts become matched and no previously matched r67 target is lost. The net output count rises from 690 to 708. The [selection ledger](union-selection.json.gz) preserves source-view IDs and each suppression witness, including the two original removals.

### Partial-corner quality matters

The eight recovered reviewed-label fronts comprise two with no known corner and six with one known corner. All six annotated corners have an available prediction under the existing endpoint threshold. Four are within 5 pixels; five are within 10 pixels. Their errors are 1.325, 1.426, 1.648, 4.089, 5.205 and 11.303 pixels. The last belongs to new_000008/R007. An unchanged complete-pair score therefore does not establish accurate known-corner placement on every newly detected partial front. Exact records appear in [paired target changes](union-target-changes.json.gz).

The conclusion also depends on the annotation reference. For the same reviewed-label outputs, the union changes misses/extras by −8/+10 on r50 and r67, but by −2/+16 on supplied664 and frozen666. Extras mean unmatched outputs under the specified reference; this count alone does not prove that each depicts a physically false front. [Output-category changes](union-output-changes.json.gz) distinguish additions, suppressed originals, duplicate changes and matching changes. The QFL r67 comparison also contains two target rematches, which are separate from its four recovered misses.

## Descriptive training support

The [training size screen](training-support.json) verifies the unchanged 142 PR157 training-label files: 3,999 fronts, including 465 with exactly one known corner. At nominal original-image coordinates, the one-known-corner counts within width/height bounds are 10 fronts across eight images for ≤18 × ≤31 pixels, zero for ≤15 × ≤14 pixels, and zero for ≤47 × ≤10 pixels. A 0.001-pixel boundary tolerance leaves these three counts unchanged.

These bins were chosen from observed development targets. They describe label sizes before augmentation; equal size does not establish similar appearance or effective training exposure. The result supports checking representative training examples before assuming a particular loss is the cause. Restoring evaluation rows does not add training examples. Complete raw original/reflected outputs from newer checkpoints remain necessary for equivalent stage attribution there.

## Reproducibility

The [summary](summary.json) binds five input archives, 21 project source modules, candidate-reference bytes, model identities and runtime. The [generated result explanation](results.md) and [result ZIP](baomai-partial-detection-r68-results.zip) come directly from the pinned driver; neither was manually amended. The ZIP is 2,289,832 bytes, SHA-256 `befe5da298de4432e74e90c8fbac0867bdf001493475679f40efad3f8ec8ad32`.

The package contains all comparison outputs, all four label references, 800 per-image score objects, 76 target-change records, 556 output-change records, the 39 saved-native-stage rows and 18 focused final-output geometry comparisons. Historical archived scoring and the r67 baseline replay are checked before the union is interpreted. The [independent verification](independent-verification.json) separately checks the numerical results and final source/notebook/package bindings.
