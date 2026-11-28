# Partial-front inclusion: exact saved-output replay

Four physically reviewed partial fronts are restored in a new r67 evaluation reference. All 654 r50 rows retain their exact bytes and row positions; the four restored historical lines are appended. The original r50, supplied664 and frozen666 references and all training labels remain unchanged. The three other reviewed candidates remain explicit holds, without ignore-region semantics.

## What changed in the measured results

For every one of the six unchanged methods, the reference-only change adds four targets, adds one match, adds three misses and removes one other-unmatched output. Duplicate counts and complete-pair scores stay unchanged. There are no retained-target reassignments.

The newly supported output is the existing `new_000010/P029`, matched to historical `r37/new_000010/R030`, now candidate `r67/new_000010/R034`. Each method had already returned its own P029 record. Its reclassification is not a newly generated detection. The other three restored fronts remain unmatched under the fixed protocol.

| Method | R67 matches / 658 | Misses | Extras | Duplicate subset | Both within 5 px | Both within 10 px | Partial fronts matched / 75 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Supplied original model | 621/658 | 37 | 60 | 5 | 555/583 | 577/583 | 41/75 |
| Historical retained method | 637/658 | 21 | 50 | 5 | 569/583 | 583/583 | 54/75 |
| PR157 native control | 637/658 | 21 | 48 | 4 | 568/583 | 583/583 | 54/75 |
| PR157 QFL candidate | 643/658 | 15 | 82 | 17 | 570/583 | 582/583 | 60/75 |
| PR158 original-label arm | 608/658 | 50 | 84 | 6 | 541/583 | 566/583 | 39/75 |
| PR158 reviewed-label arm | 640/658 | 18 | 50 | 13 | 572/583 | 583/583 | 57/75 |

R67 contains 18 fronts with no known corner, 57 with one known corner and 583 with both known corners. Missing corner flags remain unchanged. The complete-pair denominator therefore remains 583 even though the detection population increases to 658.

## Fixed suppression sensitivity

The separate eight-row [suppression sensitivity](suppression-sensitivity.csv) applies the existing r66 mutual-support gate and additional post-output NMS rules at 0.60 and 0.50 to the PR158 reviewed-label arm, independently under r50 and r67. None removes the newly supported P029 output. Under r67, the gate, 0.60 rule and 0.50 rule produce respectively `[misses, extras, duplicates] = [22,26,6], [19,31,2], [22,22,1]`; their partial-front detection is 53/75, 56/75 and 53/75. The unchanged arm is `[18,50,13]` with 57/75 partial fronts detected. All four retain 572/583 five-pixel and 583/583 ten-pixel success.

These are the same three illustrative restrictions used in r66. They act on completed outputs and do not replace native NMS. They do not introduce a threshold search or alter retained averaged coordinates or cross-view associations. Evaluation matching is recomputed for each reference and restriction.

## Reproducibility and execution

The implementation and candidate labels are pinned to `2c3886271c12e501d236a5eff54e3eb78007d1b0`. The [four-cell notebook](../../../../notebooks/baomai_partial_inclusion_r67.ipynb) was executed unchanged in one local IPython namespace. All four cells passed; the isolated checkout remained clean. The [execution receipt](notebook-execution.json) binds the notebook, each code cell, logs, environment and result package.

Acquisition was preseeded from the exact local Git source and three locally available immutable archives. The actual cells verified all archive identities and inventories. This does not claim a live GitHub acquisition test, hosted Colab execution or browser-download UI test. No photographs or weights were loaded by the replay, and no neural inference or training occurred.

The replay first reproduces all 840 archived per-image score objects across the existing methods and references. It then directly checks the six selected r50 methods against their 120 saved per-image primary scores. The r50/r67 comparison records 12 metric rows, 240 per-image score objects, 7,872 target states, 24 restored-target outcomes, six output-category transitions and zero retained-target changes. The filter supplement records eight rows and 32 restored-target survival states. Fourteen imported local Python modules form the complete declared analysis-source hash set.

Notebook: `10640` bytes; SHA-256 `67fea3563efaef53683ebeb2fabd4019eb0577800c4cdad732360defa67c224c`.

Result package: `baomai-partial-inclusion-r67-results.zip`, `1093811` bytes; SHA-256 `18c0e5491bb59b2b242853fc3df2c0ef3655099e5c98a08194846814923665cd`.

## Evidence map

- [Reference manifest](../../../../data/reviews/fronts-2026-11-28-r67/manifest.json): every candidate label identity, exact restorations, unchanged-parent relationships and three holds.
- [Review decisions](review-decisions.json), [independent physical review](independent-physical-review.json) and [historical provenance](excluded-front-provenance.json): ownership decisions, nominal geometry and source history.
- [Summary](summary.json), [matrix](matrix.csv) and [results](results.md): primary metric comparisons, availability strata, source/input identities and interpretation.
- `reference-labels.json.gz` and `reference-lineage.json.gz`: exact r50/r67 text and stable identities for all 658 candidate rows.
- `measurements.json.gz`, `per-image.json.gz` and `target-states.json.gz`: full primary measurements and individual reference states.
- `restored-target-outcomes.json.gz`, `retained-target-changes.json.gz` and `prediction-transitions.json.gz`: all reference-only changes; the retained-change ledger is empty.
- `predictions.json.gz`: the six unchanged completed output collections.
- [Suppression sensitivity](suppression-sensitivity.json) and `suppression-restored-target-survival.json.gz`: the bounded PR158 reviewed-label supplement.
- [Execution receipt](notebook-execution.json), [executed notebook](executed-notebook.ipynb), four cell logs and [package manifest](package-manifest.json): actual execution evidence.
- [Independent verification](independent-verification.json): separately implemented scoring and selectors reproduce all 12 primary rows, eight supplemental rows and change ledgers; final source, candidate inputs, notebook cells, logs and package identities also pass.
- [Result ZIP](baomai-partial-inclusion-r67-results.zip): numeric replay package with verified member inventory and immutable source binding. Its inventory covers the packaged numeric records; separate review images, review receipts and document artifacts are not implied to be package members.

## Scope

These remain repeatedly reviewed development photographs. Four resolved inclusion decisions do not certify exhaustive annotation of every image. Nominal historical pixel coordinates do not establish subpixel physical accuracy. The three held regions are not verified background and do not become ignore regions by omission. A physical positioning claim still requires camera and scene geometry. No model or threshold is promoted by this replay.
