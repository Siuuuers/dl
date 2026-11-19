# Recovered cassette-front review and PR76 decision

**Keep PR35 as the selected model. Review the focused r9 packet before a new training run.** The completed PR76 experiment and this later annotation reconstruction have distinct input versions.

## What was recovered

The preserved source is r8 commit `817927dacf5cda2232e9286ec83660abe6700caa`. The interrupted follow-up's coordinate ledger was not saved in the accessible repository or recovered files. This version therefore reconstructs the requested follow-up from preserved r8 labels, the marked screenshots, and the original camera pixels. It does not claim byte-for-byte recovery of the interrupted revision, and it does not use its reported 52-photo or 35-hold totals as targets.

[Amendment summary](../results/review/front-review-20261119-r9-recovery/amendment-summary.json), [operations](../results/review/front-review-20261119-r9-recovery/operations.json), and [case ledger](../results/review/front-review-20261119-r9-recovery/review-status.json) describe the saved decisions.

| Saved reconstruction | Count |
| --- | ---: |
| Original photographs | 157 |
| Photographs with numeric coordinate changes | 9 |
| Edited rows | 25 |
| Changed boxes / left fields / right fields | 23 / 8 / 9 |
| Added or deleted instances in this follow-up | 0 |
| Open cases | 61 |
| Held training photographs | 36 |
| Photographs with unresolved development review cases | 6 |
| Focused photograph pages | 47 |
| Unchanged photographs without an open case, omitted | 110 |

Field counts overlap within a row. The PDF has 48 pages including its introduction and linked index. Its 47 photo pages show all 1,597 current objects on those photos, including unchanged neighboring objects.

## Physical target and concrete corrections

The task remains one visible cassette front plus two actual lower-front corners, ordered by image x. An oblique camera view changes the projected positions; it does not change the physical landmark identities to farther side-face corners. An unlocatable corner keeps its own slot as exact `0 0 0`. The cassette's own frontal rim belongs to its box. A connected rack rail, rear perforated plane, hand, or gripper does not become a target corner.

This follows the upstream [baomai two-point task](https://github.com/an94-doll/embedding_vision/blob/351eb3cf960ccc34829e2bc72e1fa4645acc6295/README.md) and this project's [visible-boundary convention](annotation-boundaries-r8.md). The separately linked upstream [yolo-pose branch](https://github.com/an94-doll/embedding_vision/blob/72a99dafa1b14bfabd101e639b9ddb5f5aa38d52/README.md) has a different five-point/front-back task and is not the schema used here.

| Current row identity | Correction |
| --- | --- |
| `new_000004:46L` | Remove the point on the left white gripper; store `0 0 0`. |
| `new_000004:51R` | Remove the point on the right white gripper; trim the supported visible-fragment box. |
| `new_000014:12R` | Remove the point on the foreground gripper; trim the adjacent rail excess from the box. |
| `new_000005:29L`, `new_000012:4R` | Restore the exact earlier r7 landmark tokens supported by the original pixels. |
| `new_000007:8,9,46–49`; `new_000008:34,35` | Separate neighboring cassette tiers where inherited boxes crossed their boundaries. |
| `new_000008:6,8`; `new_000012:4,15,24`; `new_000045:10` | Include the cassette's visible raised frontal rim where the inherited top clipped it. |
| `new_000005:45`; `new_000013:1,4,20` | Remove rear top-plane material while retaining the own frontal lip. |

Other small contour corrections are recorded individually in the operations ledger. Source blur limits these placements to roughly two or three pixels; the decimal serialization is not a claim of subpixel physical certainty. Each operation records the exact source line, replacement line, source identity, changed fields, and rationale. Unchanged row bytes are retained.

Screenshot numbers are not always current label-row numbers. In the marked `new_000005` screenshot, row 27 maps to current row 29 and row 43 maps to current row 45. The mapping follows physical image content. Directly copying a screen rectangle or using its row number against another label revision would edit the wrong target.

## Holds narrowed and retained

The [old/late audit](../results/review/front-review-20261119-r9-recovery/old-and-late-audit.json) inspected all 31 old photographs. Seeing an attached lower rim did not justify a blanket redefinition of all their corners. Their coordinates remain unchanged. Twenty-nine old photos now have no open case; two retain specific box-only questions:

- `old_000010`, row 1: frame-clipped upper boundary and lip ownership.
- `old_000042`, row 10: a thin upper strip adjoining the rack and reflections.

`old_000013` remains byte-identical to the original user-confirmed label file.

A new box-only hold was added for `new_000002`, row 45. Material is visible above the hand-drawn rectangle, while a separated left sliver and the lowest right projection have uncertain ownership across the gripper. Its coordinates remain unchanged. All other unresolved gray-tip, gripper-gap, thumb-side, and frame-edge cases remain explicit. Correctly missing corners do not, by themselves, make an instance invalid; this new hold concerns box ownership.

## What the actual staged inputs contain

The [verification report](../results/review/front-review-20261119-r9-recovery/verification.json) was produced by reading materialized files from both arms. The [input inventory](../results/review/front-review-20261119-r9-recovery/training-input-inventory.json) preserves all 314 image/label hash records.

| Actual input | PR35 control | Reconstructed candidate |
| --- | ---: | ---: |
| Training photographs | 137 | 137 |
| Empty training photographs | 13 | 13 |
| Training fronts | 3,771 | 3,770 |
| Locatable training endpoint slots | 6,841 | 6,847 |
| Validation photographs / fronts | 20 / 664 | 20 / 664 |

The two arms differ on 58 training photographs. Every one of the 36 held training photographs uses the exact same PR35 bytes in both arms. This controls the annotation treatment; it does not certify those inherited labels as physical truth.

The full review contains 3,788 training fronts and 663 exploratory development fronts. Those review totals are not the actual training-input totals: the held-photo fallback changes the training composition, and the original validation labels remain fixed at 664 fronts. Both independent scoring-reference versions remain unchanged at 666 and 663 fronts. All 157 original image hashes and original label files passed verification.

The new review, candidate, and input protocol have `status: pending_review` and `release_ready: false`. No training was run. The existing notebook still names r8; it has not silently switched to these labels.

## PR76 result

The [complete PR76 audit](../results/evaluation/pr76-recovery-20261119/README.md) verified the returned archive and both 30-epoch runs, reran all three checkpoints, and reproduced the saved-prediction comparison byte for byte.

| Model, fixed 666-front reference | Matches | Scored misses | Unmatched outputs | Both corners within 5 px | Within 10 px |
| --- | ---: | ---: | ---: | ---: | ---: |
| Retained PR35 | 648 | 18 | 80 | 533/584 | 582/584 |
| Fresh r8 control | 646 | 20 | 83 | 531/584 | 583/584 |
| PR76 r8 candidate | 640 | 26 | 89 | 535/584 | 580/584 |

The candidate adds six scored misses and six unmatched outputs relative to its fresh control. Its four additional five-pixel corner pairs do not offset the loss in coverage and ten-pixel pairs under the existing decision rule. PR35 remains selected.

The actual PR76 experiment retained 64 training photographs and differed from control on 59 photos. Its results must not be attributed to this later 36-hold, 58-difference reconstruction. A scored miss can be an overlapping proposal just below the fixed IoU threshold; an unmatched output can be a duplicate or extent disagreement. Neither term alone is a physical adjudication. The 584 denominator comprises fronts with two locatable reference corners, including missed eligible fronts as failures, and is not a complete-scene success rate.

## Reproduce the review

For an independent rebuild, use a separate clean checkout of the preserved r8 base with all original PNGs available. Copy `build_front_review_r9_recovery.py` and the four audit JSON files from this branch into their matching paths, then run:

```bash
python3 -B scripts/build_front_review_r9_recovery.py \
  --audit results/review/front-review-20261119-r9-recovery/early-secondary-audit.json \
  --audit results/review/front-review-20261119-r9-recovery/middle-audit.json \
  --audit results/review/front-review-20261119-r9-recovery/old-and-late-audit.json \
  --audit results/review/front-review-20261119-r9-recovery/root-audit.json \
  --build
```

The builder refuses to overwrite existing generated artifacts. A fresh staging directory lives outside the repository. After generation, `--verify-only` rereads the materialized arms and reconciles their inventory with the saved receipt.

Render the focused packet from the saved review:

```bash
python3 -B scripts/build_front_review_r9_pdf.py \
  --output /tmp/cassette-r9-raw.pdf \
  --qa-output /tmp/cassette-r9-layout.json
python3 -B scripts/optimize_review_pdf.py \
  --input /tmp/cassette-r9-raw.pdf \
  --output /tmp/cassette-r9-focused.pdf \
  --audit-output /tmp/cassette-r9-lossless.json
```

The optimizer requires fresh output paths. It preserves every decoded image pixel, vector content stream, page geometry, link, and bookmark. It does not resize images or JPEG-compress them.

The [layout audit](../results/review/front-review-20261119-r9-recovery/pdf-layout-qa.json) verifies all 1,597 number tags: zero overlapping tag rectangles and zero clipped tags. Geometry renders first; labels and leader lines render afterward. Moving 603 tags leaves every saved box and endpoint unchanged. Magenta marks changed fields, amber marks unresolved fields, and cyan means inherited/unchanged. The new PDF gives priority to identifying each object while keeping all neighboring labels visible.

Each amber case has a [short display note](../results/review/front-review-20261119-r9-recovery/pdf-case-notes.json) stating its actual remaining uncertainty. The renderer verifies that these notes cover exactly the 61 held case IDs and match the saved case-ledger hash. This prevents a methods sentence or a description of a supported correction from obscuring why a different field is still held. Full evidence remains in the case ledger.

## Next decision

Review the magenta fields and answer the specific amber ownership questions using current photo IDs and row numbers. A useful correction states the field, the physical contour or occluder that justifies it, and whether the corner is actually locatable. Once those decisions are saved, a separate candidate and notebook can be pinned for a new controlled experiment. Independent capture sessions and physical camera calibration would be needed before making unseen-scene or millimetre-level positioning claims.

For manual drawing, the existing [HTML labeler](../tools/cassette_labeler/README.md) accepts a working folder containing the original `images/train,val` plus the matching r9 review `labels/train,val`. Use copies of the labels under `data/reviews/fronts-2026-11-19-r9-recovery/`, which contain the full proposed review. The staged training candidate intentionally falls back to PR35 on held photos and is therefore a different working set. Keep the original `data/labels/` and frozen scoring references intact. No new application or model architecture is needed for this review workflow.
