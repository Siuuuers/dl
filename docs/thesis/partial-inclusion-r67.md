# Four partial fronts restored: review and reference-only replay

28 November 2026, Asia/Hong_Kong. **Completed:** original-image review, a separately versioned evaluation reference, six-method replay, fixed-filter sensitivity and all four actual notebook cells.

The review restores four identifiable partial cassette fronts across three photographs. It follows the existing [visible-front definition](../annotation-definition.md): include an identifiable observed front and record each actual lower corner only when locatable. The [research note](research-partial-inclusion-r67.md) checks this interpretation against the supplied Baomai guidance and primary sources. Image evidence determines these four inclusion decisions; model predictions and new scores did not determine their geometry.

## What the new reference contains

The [r67 manifest](../../data/reviews/fronts-2026-11-28-r67/manifest.json) records **658 fronts on the same 20 development photographs**: 18 with no known corner, 57 with one and 583 with both. This gives **1,223 known corners** and preserves the **583-front complete-pair denominator**. All **654 r50 rows remain byte-identical and at their existing positions**. The four additions use exact historical label tokens, appended in a deterministic order. Training labels and the supplied664, frozen666 and r50 evaluation snapshots retain their earlier bytes.

| Historical row | Appended r67 row | Observed front | Known corner retained |
| --- | --- | --- | --- |
| R34 `new_000009:56` | `new_000009:55` | Right-jaw fragment | Right `(823,516)` |
| R37 `new_000009:52` | `new_000009:56` | Lowest visible band left of the gripper | Left `(601,502)` |
| R37 `new_000010:30` | `new_000010:34` | Bottom-clipped band below the broader front | Neither |
| R37 `new_000053:1` | `new_000053:22` | Small wedge at the top image boundary | Left approximately `(640.8,8.33)` |

The historical version prefix matters: old R34 or R37 row numbers can name different objects from the same-numbered current r50 row. Exact boxes, point values, line hashes and source-image identities are preserved in the [provenance](evidence/partial-inclusion-r67/excluded-front-provenance.json) and [decision ledger](evidence/partial-inclusion-r67/review-decisions.json).

Raw photographs and crops support four distinct fronts, including the top wedge and bottom band whose enclosing rectangles overlap neighbouring labels. That overlap alone does not establish duplicate ownership. The accepted coordinates remain nominal image annotations; soft boundaries and decimal serialization do not establish subpixel physical accuracy. Unknown corners retain their exact `0 0 0` slots. See the [independent physical review](evidence/partial-inclusion-r67/independent-physical-review.json) and [raw/overlay evidence inventory](evidence/partial-inclusion-r67/review-evidence.json).

Three cases remain held: **R34 `new_000009:57` and R37 `new_000086:57` / `new_000088:57`**. The central lower patch has competing object-ownership interpretations; the two thumb-adjacent slivers retain uncertainty about frontal extent and landmark ownership. These omissions are explicit limits of the reference. They do not establish background or create ignore regions. The focused review does not certify every annotation or measure human annotation repeatability.

The changed-photo PDF covers only `new_000009`, `new_000010` and `new_000053`, with additions highlighted. Its source and output identities are recorded in [PDF validation](evidence/partial-inclusion-r67/pdf-validation.json); it is delivered separately from the numeric replay package.

## What changes when predictions stay fixed

The replay uses the six methods' existing completed outputs and the unchanged primary scorer. For **every method**, adding the four reference rows produces **one more match, three more misses and one fewer other-unmatched output**. Duplicates and five-/ten-pixel complete-pair counts remain unchanged. No existing reference target is reassigned. The [full matrix](evidence/partial-inclusion-r67/matrix.csv) retains r50 beside r67, and the [summary](evidence/partial-inclusion-r67/summary.json) includes all corner-availability strata.

| Method | R67 misses | Extras | Duplicate subset | Both within 5 px | Both within 10 px |
| --- | ---: | ---: | ---: | ---: | ---: |
| Supplied original | 37 | 60 | 5 | 555/583 | 577/583 |
| Historical retained | 21 | 50 | 5 | 569/583 | 583/583 |
| PR157 native | 21 | 48 | 4 | 568/583 | 583/583 |
| PR157 QFL | 15 | 82 | 17 | 570/583 | 582/583 |
| PR158 original labels | 50 | 84 | 6 | 541/583 | 566/583 |
| PR158 reviewed labels | 18 | 50 | 13 | 572/583 | 583/583 |

Each method had already emitted its own `new_000010/P029` record. It now matches the restored bottom band, changing from other-unmatched to matched. The other three restored fronts remain unmatched. This is a measured reference-population effect: no model learned a new detection, and no output was added or removed. The [restored-target outcomes](evidence/partial-inclusion-r67/restored-target-outcomes.json.gz) and [prediction transitions](evidence/partial-inclusion-r67/prediction-transitions.json.gz) identify every change.

## The suppression risk remains

The separate [eight-row sensitivity check](evidence/partial-inclusion-r67/suppression-sensitivity.csv) repeats the three fixed r66 restrictions for the PR158 reviewed-label arm under both references. All three retain the newly supported P029. Their losses of other partial-front matches remain visible under r67:

| Fixed restriction | R67 misses | Extras | Duplicate subset | Partial fronts matched |
| --- | ---: | ---: | ---: | ---: |
| Unchanged | 18 | 50 | 13 | 57/75 |
| Require existing mutual reflected support | 22 | 26 | 6 | 53/75 |
| Additional suppression above IoU 0.60 | 19 | 31 | 2 | 56/75 |
| Additional suppression above IoU 0.50 | 22 | 22 | 1 | 53/75 |

Every row still scores 572/583 within five pixels and 583/583 within ten. Complete-pair scores therefore continue to conceal losses among incomplete-corner fronts. The restrictions act separately on completed outputs, preserve retained coordinates and existing view associations, and use fresh evaluation matching. They do not replace native NMS or select a new operating threshold. See the [restored-output survival ledger](evidence/partial-inclusion-r67/suppression-restored-target-survival.json.gz) and the preceding [duplicate audit](duplicate-audit-r66.md).

## Reproduce the completed analysis

Run all four code cells in the [CPU notebook](../../notebooks/baomai_partial_inclusion_r67.ipynb). The implementation and candidate labels are pinned to **`2c3886271c12e501d236a5eff54e3eb78007d1b0`**. Analysis uses the Python standard library and three hash-verified archives; it needs no photograph or model download, GPU, new inference or training.

All four actual cells completed unchanged in one local IPython session with a clean pinned checkout and preseeded immutable input archives. This verifies the notebook execution and saved-input analysis; live network acquisition, hosted Colab and its browser-download interface were not exercised. The [execution receipt](evidence/partial-inclusion-r67/notebook-execution.json) binds code cells, logs and outputs. [Independent verification](evidence/partial-inclusion-r67/independent-verification.json) reproduces all 12 primary rows, eight supplemental rows and change ledgers and checks the final source, notebook and package identities.

The [evidence guide](evidence/partial-inclusion-r67/README.md) maps the full measurements, per-image scores, stable row lineage and exact before/after label text. The [numeric result ZIP](evidence/partial-inclusion-r67/baomai-partial-inclusion-r67-results.zip) packages those records with input/source hashes; the [package manifest](evidence/partial-inclusion-r67/package-manifest.json) records its identity. Earlier result matrices remain available under their original reference names. No model or deployment default is promoted by this review.
