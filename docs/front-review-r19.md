# Round 19: PR92 audit and cassette-front corrections

22 November 2026, Asia/Hong_Kong.

**PR35 remains selected. The current training entry is the r19 notebook.** The [PR92 audit](../results/evaluation/pr92-r18-20261122/README.md) verifies both returned runs and preserves their pre-return evaluation. This amendment pass saves **460 row replacements across 34 photographs**, after inspecting all 40 supplied nonempty photos in 081–130. No r19 GPU training was run during preparation.

## What was corrected

The user's examples distinguish the front-face corner from the outer side silhouette. The previous pass put 22 left points onto the wrong outer edge. Each was reinspected against original pixels and restored to the supported front-face junction. The two earlier right-point moves on 096 rows 42/45 remain visually supported and were retained. Photo 128 rows 31–33 keep their accepted left-point coordinates, and photo 086 row 3 keeps its accepted box.

Boxes now exclude the supported rack and finger margins while retaining real projecting front material and rims. There are **446 changed box groups, 23 changed left-point groups, and 1 changed right-point groups**. One row can contain more than one changed group. All object counts and row identities are preserved. The only visibility change restores the explicitly requested left corner of 127 row 32.

| User example | Saved decision |
| --- | --- |
| 094 row 1 | xmin 146, ymin 398, xmax 298; existing ymax about 441.05 retained. The projecting **upper** rim requires smaller ymin in downward image Y. Extending ymax would enter the next cassette. |
| 094 rows 2/3 | xmax copies immutable r18 row 7 lpt.x / row 9 xmin, about 301.58 / 309.19 pixels. |
| Other named 094 and 089 edges | Tighten the visually supported rack margins; keep cassette rims and independently placed corners. Every named clause has a saved disposition. |
| 087 row 52 / 090 row 43 | xmax 522 / 461 removes the clear finger extension. Their missing right points stay missing. |
| 089 row 20 | Correct the right point from the dark separator to x 766; set xmax 766.5 while preserving point Y and visibility. |
| 114 rows 15/16 | Left X returns to about 962.60 / 955.20 at the front-face junction. Row 15 agrees with row 14's left X around 963.17. |
| 127 row 32 | Add the visible left point near (941, 518) pixels with visibility 2 on the existing object. |

Visual estimates are pixel-based judgments; serialized decimals do not imply subpixel physical accuracy. Axis-aligned boxes around slanted fronts can contain background in their triangular corners. The correction removes unsupported outer margins without cutting actual cassette material. A lower-corner X is therefore not a universal box boundary. See the [annotation definition](annotation-boundaries-r19.md).

## The annotations are saved and used

The [complete r19 review](../data/reviews/fronts-2026-11-22-r19/manifest.json) contains all **157 annotation files under its labels directory**. The [r19 candidate manifest](../data/candidates/front-review-2026-11-22-r19/manifest.json) selects the eligible training supervision, and the notebook stages that candidate explicitly. The PDF is rendered from these same saved labels. Original data/labels remains the preserved source dataset.

All 117 unrequested files are byte-identical to r18. Within the inspected range, these photos remain unchanged: **081, 099, 100, 102, 103, 112**. IDs 091–093 and 104–110 are absent from the prepared source; they are not empty scenes. No new questions, cases, holds or amber markers were created. Existing 17 displayed-case approvals remain inherited.

The [exact operations](../results/review/front-review-20261122-r19/operations.json) record immutable r18 row identities, exact before/after lines, image and label hashes, and physical reasons. [Request coverage](../results/review/front-review-20261122-r19/request-coverage.json) records all 40 photos and explicit user clauses. [Verification](../results/review/front-review-20261122-r19/verification.json) replays every label and confirms the single point restoration and retained reference examples.

### Changed rows by photo and field group

| Photo | Box rows | Left-point rows | Right-point rows |
| --- | --- | --- | --- |
| 082 | 23–25, 28 | — | — |
| 083 | 3, 5, 8, 10, 12–13 | — | — |
| 084 | 21–28, 30 | — | — |
| 085 | 24–33 | — | — |
| 086 | 20, 22–23, 34–35, 44–45 | — | — |
| 087 | 20, 34–35, 44, 52 | — | — |
| 088 | 20, 34–35, 42 | — | — |
| 089 | 6–9, 20, 28–39, 44–45 | — | 20 |
| 090 | 3, 6–7, 20, 34–35, 43 | — | — |
| 094 | 1–3, 5–7, 9, 16–17, 19, 24, 31–34 | — | — |
| 095 | 1–2, 5–9, 21–23, 28–30, 40–43 | — | — |
| 096 | 1–3, 5–9, 15–24, 28–32, 35–38, 43–44 | — | — |
| 097 | 1–9, 21–24, 26–30, 41–44 | — | — |
| 098 | 26 | — | — |
| 101 | 16, 18, 20 | — | — |
| 111 | 1–3, 25, 27 | — | — |
| 113 | 1–3, 5–8, 13–19, 23–27, 29 | — | — |
| 114 | 1–3, 17, 19, 24 | 15–16 | — |
| 115 | 1–3, 5, 10, 12, 14–19 | 14–16 | — |
| 116 | 1–3, 5–7, 9–11, 16, 18, 20 | 15–17 | — |
| 117 | 1–4, 6, 10, 12–19, 23 | 14–16 | — |
| 118 | 1–3, 21–22 | — | — |
| 119 | 25 | — | — |
| 120 | 2–3, 5–9, 15, 17, 19, 25–26, 28–32, 43–45, 49–51 | — | — |
| 121 | 2–3, 5–7, 16–18, 20, 26, 29–33, 44–45, 50–52 | — | — |
| 122 | 1–5, 16–23, 26–27, 29–33, 43, 49–52 | — | — |
| 123 | 1–4, 6–8, 15–20, 25–26, 28, 30, 43–46 | — | — |
| 124 | 1–3, 6–8, 15–17, 24, 29, 40–43 | — | — |
| 125 | 2–3, 5–9, 15–24, 26, 29–34, 44–45, 47–52 | — | — |
| 126 | 13–16, 20, 40–42, 46–47 | — | — |
| 127 | 7–8, 15–16 | 7–8, 29–35 | — |
| 128 | — | 7–8, 27 | — |
| 129 | 2–8, 14–18, 20, 23–32, 42–44, 47–49 | — | — |
| 130 | 1, 3–13, 15–19, 21–26, 29–33, 44–45, 50–52 | — | — |

## Changed-only PDF

The PDF has **35 pages**, including its linked index and 34 changed photos. Every current front on each included photo is displayed: **1440 current row tags**. Magenta marks changed groups and cyan marks retained geometry. There are no amber or question markers. Original decoded RGB pixels, saved coordinates, row identities, links, label placements and footers are checked. [PDF delivery and verification](../results/review/front-review-20261122-r19/pdf-delivery.json).

## What PR92 established

Both returned arms completed the fixed 30-epoch experiment on the same A100 runtime. The actual archive, all indexed files, exact training source and notebook, staged inputs and checkpoint bundles passed verification. Fresh inference was performed for all three models. PR35 and the fresh control produced exactly equal prediction arrays on the 20 development photos.

| Reference | Model | Misses | Unmatched outputs | Pairs ≤5 px | Pairs ≤10 px |
| --- | --- | ---: | ---: | ---: | ---: |
| Frozen primary, 666 fronts | PR35 / fresh control | 18 | 80 | 533/584 | 582/584 |
| Frozen primary, 666 fronts | r18 candidate | 34 | 103 | 522/584 | 575/584 |
| Pre-return r18 review, 662 fronts | PR35 / fresh control | 36 | 102 | 538/584 | 573/584 |
| Pre-return r18 review, 662 fronts | r18 candidate | 15 | 88 | 551/584 | 583/584 |

The historical 663-front reference also favors PR35/control: 15 misses versus 31 for r18. All three models produce zero outputs on the three empty scenes. Unmatched outputs are matching outcomes, not a claim that every output is independently confirmed background. Eligible missed objects stay in the corner-pair denominator.

On identical pre-return r18 labels, the preceding r17 candidate had 14 misses, 87 unmatched outputs, and 549/584 five-pixel pairs; r18 has 15, 88, and 551/584. Ten-pixel successes are unchanged. The sole lost match is 011 row 55 crossing the fixed IoU boundary from about 0.50572 to 0.49452. Training hardware changed from L4 to A100 between generations, so this does not isolate the annotation amendment's effect.

There is a separate reference-only effect: changing r16 labels to r18 changes the score of unchanged r17 predictions at 090 row 43. The audit reports that independently. **No r19 label enters the PR92 scores.** The conflicting reference rankings leave PR35 selected under the fixed protocol. [Full measured results and reproducible evidence](../results/evaluation/pr92-r18-20261122/README.md).

## Actual next training inputs

| Materialized training input | Reference | Candidate |
| --- | ---: | ---: |
| Photographs | 137 | 137 |
| Empty scenes | 13 | 13 |
| Front instances | 3,771 | 3,784 |
| Locatable endpoints | 6,841 | 6,838 |
| Missing endpoint slots | 701 | 730 |

Exactly **92 training-label files differ** between arms. The complete review has 3,786 training fronts; eligible candidate training has 3,784, because photo 015 retains PR35 supervision in both arms. The separate 31 old-series treatment exclusions also retain PR35 supervision. All photos remain present. [Actual staging and readback](../results/review/front-review-20261122-r19/staging-verification.json).

Both arms initialize independently from original epoch 173. Keep seed 46, 30 epochs, batch 4, input 1280, FP32, AdamW with initial learning rate 0.0001, and final-epoch selection. The resource trial cannot initialize either full run. Original trainer validation stays at 20 photos / 664 fronts; frozen primary and historical scoring stay at 666 / 663. The separately pinned r19 development review has 662 fronts and is exploratory.

## Run the notebook

[Open r19 in Colab](https://colab.research.google.com/github/Siuuuers/dl/blob/codex/front-review-r19-20261122/notebooks/baomai_front_review_r19.ipynb), or download [baomai_front_review_r19.ipynb](../notebooks/baomai_front_review_r19.ipynb).

1. Select **Runtime → Change runtime type → GPU**.
2. Select **Run all** and connect Drive when offered.
3. Let verification, the resource trial and both fresh arms complete.
4. Return **baomai-front-review-r19-results.zip** for evaluation.

The final notebook pins a full source commit and exact protocol hash. Its input preflight checks original images and initializer, exact annotation replay, the point restoration, actual arm composition, split boundaries and references. [Notebook preparation receipt](../results/review/front-review-20261122-r19/notebook-verification.json). CUDA installation, Drive interaction and GPU training were not executed during preparation.

The experiment compares PR35 supervision with the complete eligible r19 review; it does not isolate this amendment round. Repeated development scenes do not establish performance on an unseen capture session. Pixel radii measure image-space agreement, not millimetre positioning or robot grasp success. See the [physical geometry guide](robot-grasp-readiness.md) for that later boundary.
