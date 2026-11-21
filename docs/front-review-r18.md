# Round 18: PR90 audit, saved corrections and next notebook

22 November 2026, Asia/Hong_Kong.

**PR35 remains selected.** The [PR90 evaluation](../results/evaluation/pr90-r17-20261122/README.md) verifies the returned r17 training and reports both frozen and reviewed-reference results. This round saves **36 coordinate replacements across 14 photographs**, then pins a new notebook to those exact inputs. No r18 GPU training was run during preparation.

## Saved annotation changes

All 40 supplied nonempty photographs in 081–130 were inspected against original pixels: 35 training and five development photographs (086–090). IDs 091–093 and 104–110 are absent from the prepared source. The other 26 inspected photographs retain their r17 bytes.

Only confident changes were saved: **12 box groups, 22 left-point groups and two right-point groups**. There are no added/deleted objects, changed visibility values, new cases, new holds, or uncertainty flags. All 117 unrequested review files and 143 of the 157 complete review files are byte-identical to r17. Row numbers retain their r17 identities.

In this table, `a ← b` means copy the existing coordinate b into a. Each listed row is modified once.

| Photograph | Split | xmin ← lpt.x | lpt.x ← xmin | rpt.x ← xmax |
| --- | --- | --- | --- | --- |
| new_000086 | development | 52 | — | — |
| new_000087 | development | 52 | — | — |
| new_000088 | development | 52 | — | — |
| new_000089 | development | 50 | — | — |
| new_000090 | development | 43 | — | — |
| new_000094 | training | 2; 3 | — | — |
| new_000095 | training | 2; 3; 28 | — | — |
| new_000096 | training | 2; 3 | — | 42; 45 |
| new_000114 | training | — | 15; 16 | — |
| new_000115 | training | — | 14; 15; 16 | — |
| new_000116 | training | — | 15; 16; 17 | — |
| new_000117 | training | — | 14; 15; 16 | — |
| new_000127 | training | — | 7; 8; 29; 30; 31; 33; 34; 35 | — |
| new_000128 | training | — | 7; 8; 27 | — |

The box trims remove unsupported rack/background margins. The point moves correct endpoints visibly inset into the front face. Existing sloped sides and projecting owned rims remain independently represented. Equality between a box edge and a corner is not a universal geometric constraint. Uncertain fields are left unchanged and are not enumerated as new review cases.

The [complete r18 review](../data/reviews/fronts-2026-11-22-r18/manifest.json) has all 157 annotation files under `labels/`. The [exact operation ledger](../results/review/front-review-20261122-r18/operations.json) retains original and replacement lines, source hashes, physical reasons and immutable row identities. [Coverage](../results/review/front-review-20261122-r18/request-coverage.json) records every inspected photo; [verification](../results/review/front-review-20261122-r18/verification.json) replays every saved label. Original `data/labels/` remains the preserved source dataset.

## Changed-only PDF

The PDF contains the 14 changed, nonempty photographs plus a clickable index. Every current front on each included photograph is shown. Magenta marks only changed fields; cyan marks retained geometry. There are no amber or question markers. The renderer reads saved labels, preserves all original PNG pixels, and cannot move the geometry. [PDF delivery and verification](../results/review/front-review-20261122-r18/pdf-delivery.json).

## PR90 result and decision

Both returned arms completed the pinned 30-epoch run. Archive, source, notebook, staged inputs and checkpoint checks passed. All three models were freshly inferred at the fixed operating point.

| Reference | Model | Misses | Unmatched outputs | Pairs ≤5 px | Pairs ≤10 px |
| --- | --- | ---: | ---: | ---: | ---: |
| Frozen primary, 666 fronts | PR35 | 18 | 80 | 533/584 | 582/584 |
| Frozen primary, 666 fronts | Fresh control | 20 | 83 | 531/584 | 583/584 |
| Frozen primary, 666 fronts | r17 candidate | 34 | 103 | 521/584 | 575/584 |
| Pre-return review, 662 fronts | PR35 | 35 | 101 | 538/584 | 573/584 |
| Pre-return review, 662 fronts | Fresh control | 38 | 105 | 534/584 | 573/584 |
| Pre-return review, 662 fronts | r17 candidate | 15 | 88 | 549/584 | 583/584 |

The historical 663-front comparison also favors PR35 over the candidate, with 15 versus 31 misses. The same predictions yield the different reviewed-reference ranking. Against r15 on identical reviewed labels, r17 gains one match but adds two unmatched outputs and one duplicate; total corner-pair successes tie. These results do not satisfy the frozen promotion rule. The full audit preserves matching identities, all three reference inventories, counts by locatable corners, original empty scenes and reproducible predictions.

These PR90 results use the references pinned before the return. The later r18 development amendments cannot silently replace that comparison.

## Actual next training inputs

| Materialized training input | Reference | Candidate |
| --- | ---: | ---: |
| Photographs | 137 | 137 |
| Empty scenes | 13 | 13 |
| Front instances | 3,771 | 3,784 |
| Locatable endpoints | 6,841 | 6,837 |
| Missing endpoint slots | 701 | 731 |

Exactly **92 training-label files differ** between arms. The complete review has 3,786 training fronts; the candidate has 3,784 because the inherited photo 015 decision retains PR35 supervision in both arms. The separate 31 old-series treatment exclusions also retain PR35 supervision. No photo is newly removed or held.

| Fixed choice | Value |
| --- | --- |
| Initialization | Original epoch 173, independently for both arms |
| Recipe | Seed 46, 30 epochs, batch 4, input 1280 |
| Optimizer | AdamW, initial learning rate 0.0001, FP32 |
| Selection | Final epoch; resource trial cannot initialize either arm |
| Trainer validation | Original 20 photographs / 664 fronts |
| Primary scoring | Frozen 666 fronts |
| Historical scoring | Frozen 663 fronts |
| r18 development review | 662 fronts, separately pinned and exploratory |

## Run the notebook

[Open r18 in Colab](https://colab.research.google.com/github/Siuuuers/dl/blob/codex/front-review-r18-20261122/notebooks/baomai_front_review_r18.ipynb), or download [baomai_front_review_r18.ipynb](../notebooks/baomai_front_review_r18.ipynb).

1. Select **Runtime → Change runtime type → GPU**.
2. Select **Run all** and connect Drive when offered.
3. Let input verification, the resource trial and both independently initialized arms complete.
4. Return **`baomai-front-review-r18-results.zip`** for fixed evaluation.

The delivered notebook has a full source-commit pin and exact protocol hash. Its preflight checks original images and weights, exact label replay, actual arm composition, split boundaries, inherited decisions and frozen references. [Preparation receipt](../results/review/front-review-20261122-r18/notebook-verification.json). CUDA installation, Drive interaction and GPU training are not executed during preparation.

This experiment compares PR35 supervision with the complete eligible r18 review. It does not isolate the r17-to-r18 changes. Repeated development photos do not establish unseen-session reliability; a separate capture session labeled before model inspection would supply stronger evidence. Pixel thresholds do not establish robot positioning tolerance in millimetres; see the [physical geometry guide](robot-grasp-readiness.md).
