# Improve training supervision, then measure the result

> **Update — 2026-11-11:** The seed45 pair returned in [PR32](https://github.com/Siuuuers/dl/pull/32) and its [full comparison is complete](annotation-comparison-2026-11-11.md). On the revised reference at confidence 0.25, the candidate has 24 misses/55 extras versus its control's 25/58, ties at 10 px both-corner success, and loses one 5 px success. Epoch173 remains the working default. The next action is the separately pinned [seed46 confirmation](training-annotations-seed46.md). The preparation-stage protocol and evidence below remain as the historical record; statements that training had not yet occurred describe that stage.

**The next intervention is ready: five omitted cassette fronts are added, four gripper-only labels are removed, and two redundant rows are consolidated with their useful corner marks preserved.** Three retained rows are amended across eight training images. The [candidate manifest](../data/candidates/annotations-2026-11-10/manifest.json) and [review record](../data/candidates/annotations-2026-11-10/review.json) contain the exact edits. No new model has been trained in this preparation workspace, which has no CUDA GPU.

Run the [reviewed-annotations notebook](../notebooks/baomai_reviewed_annotations.ipynb) in the established GPU workflow and return `baomai-reviewed-annotations-seed45-results.zip`. The notebook pins an immutable implementation revision and the exact candidate manifest, downloads the original inputs through Git LFS, performs a short resource trial, trains the matched pair and packages both final checkpoints.

## Why this change comes next

The [completed reference amendment](reference-amendment-2026-11-10.md) improved the development labels and replayed unchanged predictions. On that revised reference, epoch173 at box confidence 0.25 has **27 misses and 54 extras**, with **575/584 eligible objects** returning both lower corners within the provisional 10 px ceiling. Those are measurements of the existing model against revised labels; they are not evidence that it learned anything new.

The earlier [box experiment across seeds43/44](box-repeatability-comparison-2026-11-10.md) did not reproduce the seed42 detection gain. Repeating the same sixteen box edges would not test a new intervention. This pass instead addresses examples where the model was given no target for an identifiable front, conflicting targets for the same front, or a target located on the gripper. These errors are plausible sources of confusing supervision. Whether this small correction set improves predictions remains an empirical question.

The user prioritizes missed objects and authorizes assistant-marked annotations. The task remains cassette localization in the supplied D405 still images with two actual lower-front corners. No architecture or hardware-integration change is needed for this experiment.

## Accepted changes

All row numbers below refer to the **original frozen label file**, before deletions. New objects append after retained rows; the manifest contains the complete original-to-candidate map. Image links point to the existing original pixels at the reviewed parent revision.

| Original image | Accepted change | Corner handling |
| --- | --- | --- |
| [new_000065](https://github.com/Siuuuers/dl/blob/fda37dc364f8c6ea54ad23819482dbf21be0534a/data/images/train/new_000065.png) | Add the top-clipped left front above row15 and the hand-occluded front below row14 | Top front: new visible marks `(194,12)` and `(340,17)`; lower fragment: both missing |
| [new_000067](https://github.com/Siuuuers/dl/blob/fda37dc364f8c6ea54ad23819482dbf21be0534a/data/images/train/new_000067.png) | Delete aggregate row25, which spans four fronts already represented by rows3/5/8/12; transfer its useful point to row12 | Exact left tokens from25 preserved on12; right remains missing; existing row12 box retained |
| [new_000068](https://github.com/Siuuuers/dl/blob/fda37dc364f8c6ea54ad23819482dbf21be0534a/data/images/train/new_000068.png) | Delete gripper-only rows15 and16; retain the real front fragment in14 | No visible point removed |
| [new_000072](https://github.com/Siuuuers/dl/blob/fda37dc364f8c6ea54ad23819482dbf21be0534a/data/images/train/new_000072.png) | Delete gripper-only35; consolidate36 into34 and refit34 to the observed right fragment `[686,458,702,472]` | Exact right tokens from36 preserved on34; left remains missing |
| [new_000097](https://github.com/Siuuuers/dl/blob/fda37dc364f8c6ea54ad23819482dbf21be0534a/data/images/train/new_000097.png) | Add omitted front beneath its slotted upper surface in the jaw opening; box `[681,483,734,514]` | Both physical lower corners hidden |
| [new_000101](https://github.com/Siuuuers/dl/blob/fda37dc364f8c6ea54ad23819482dbf21be0534a/data/images/train/new_000101.png) | Add omitted front in the jaw opening; box `[678,483,734,515]` | Both physical lower corners hidden |
| [new_000102](https://github.com/Siuuuers/dl/blob/fda37dc364f8c6ea54ad23819482dbf21be0534a/data/images/train/new_000102.png) | Refit row53 from excessive gripper extent to observed front `[694,499,731,515]` | Preserve its original missing-corner tokens |
| [new_000103](https://github.com/Siuuuers/dl/blob/fda37dc364f8c6ea54ad23819482dbf21be0534a/data/images/train/new_000103.png) | Delete source50 on solid gripper; separately add omitted front at `[698,503,730,515]` | Both corners hidden; deletion and addition have separate provenance |

Five additions and six deletions produce a net loss of one row. **The net count does not describe the correction:** the important changes are five recovered positive examples and removal or reconciliation of six incorrect identities. All **6,839 existing visible endpoint occurrences** survive with their exact original tokens. The top-clipped added front contributes the only two new visible endpoints.

| Training quantity | Original-label control | Annotation candidate |
| --- | ---: | ---: |
| Images | 137 | 137 |
| Objects | 3,772 | 3,771 |
| Visible endpoints | 6,839 | 6,841 |
| Missing endpoints | 705 | 701 |
| Empty images | 13 | 13 |

Both training arms use the identical original validation split: **20 images, 664 objects, 1,227 visible endpoints, 101 missing endpoints, three empty images**. The separately versioned PR30 reference is used for post-training reporting, not silently substituted inside one arm.

## What the review does and does not establish

The focused pass triaged **32 training images containing 1,031 original rows**: actual training IDs061–070,071–075,079–085 and094–103. Original pixels were inspected before using source overlays for row correspondence. Proposed changes received detailed crop review and independent assistant review of the exact final files. No model prediction was copied into the labels, and no enhanced or generated pixels supplied evidence.

The rule remains [baomai-visible-front-v1](annotation-policy.md): one identifiable, positively observed front per row; an observed-extent box; actual lower-front corners ordered by image X. A front can be a valid detection target when both corners are hidden. Its hidden corners stay `0 0 0`; an intersection with a finger or jaw is not a physical corner. A row lying entirely on opaque gripper material is not justified by assumed stack spacing.

New integer boundaries are manual marks on blurred pixels. Approximately 1–2 px judgment describes their practical precision; it is not a measured statistical uncertainty. **The 10 px evaluation ceiling is not permission to place annotations loosely.** For an eligible object, both returned corners must each be within 10 original-image pixels of their corresponding reference corner; the 5 px diagnostic remains.

This is a bounded intervention, not full dataset certification. The review records unresolved identity/extent questions in066,079,082 and image-boundary fragments, plus legacy boxes that extend into roofs or occluders. Row12 in067 retains its existing box deliberately. The 137-image training set has not been fully redrawn. The historical eleven-row deduplication and sixteen-edge candidates remain separate; known old duplicates in075 and102 are not silently folded into this experiment. A future combined candidate would require its own explicit comparison.

## Fixed experiment

Experiment: `reviewed-annotations-seed45-v1`. Seed45 is selected prospectively, before observing these results. Run one matched pair first; a useful directional result may justify a separately recorded seed46 confirmation. Do not search seeds for a favorable result.

| Setting | Both arms |
| --- | --- |
| Initialization | Exact original epoch173 checkpoint, SHA-256 `e0aab1b377491700318e0b14cd4c558ec3c1684087a13b7dde2298ebdbf2847e` |
| Epochs and selection | 30 completed epochs; finalized `last.pt` from epoch30 |
| Image size and batch | 1280, rectangular loading, batch4 |
| Precision and optimizer | FP32, AdamW, nominal `lr0=0.0001` |
| Bias warmup | `warmup_bias_lr=0.0001` |
| Seed | 45 |
| Remaining numerical settings | Established low-bias recipe, unchanged |
| Trainer | Pinned Ultralytics8.4.182; matched recorded CUDA runtime |
| Scientific difference | Original versus this reviewed training-label overlay |

Recipe SHA-256: `34b8eda9e9b986a2ba19d9ba07661eef1792027ef8d4465d10763a8eb2cd2850`.

Each arm starts independently from the original checkpoint, including after the resource trial. Neither the trial, the other arm, nor a historical fine-tune initializes it. This is fresh fine-tuning, not optimizer resume. One pair is two full runs, 60 total epochs and 8,220 nominal training-image presentations; these are design counts, not runtime estimates.

The notebook preserves installed CUDA Torch/torchvision versions while installing the pinned trainer. It verifies the entire input inventory, runs four real candidate batches under the full schedule, checks optimizer updates and a parameter change, and then starts the full runs. Trial and full-run stop budgets are limits, not duration promises; the trial does not guarantee that final validation or checkpoint finalization will fit. Failed runs remain ineligible. Keep their receipts/logs, use fresh output folders for retries, and keep both arms in the same source/runtime. No automatic overwrite or resume occurs.

## How the returned pair will be judged

Perform fresh inference for **M0 epoch173, M1 original-label control and M2 annotation candidate** after both training arms finish. Use the same pinned compatible script bytes for training, inference, quality diagnostics and the final audit. All three prediction/quality runs must identify one clean evaluation revision; old M0 results do not satisfy the fresh comparison.

Preserve image size1280, rectangular FP32 preprocessing, batch1, NMS IoU0.70, matching IoU0.50, endpoint confidence0.50 and `max_det=300`. **Box confidence0.25 is primary**. Confidence0.15 and0.40 remain fixed diagnostics; no threshold is chosen after seeing the result.

Report original and PR30 revised references separately, with revised-reference results as the current development interpretation and original results retained for historical comparability. For each model, report:

1. Missed objects, including zero-, one- and two-locatable-corner groups.
2. Extra detections and empty-image results.
3. Missing visible endpoints.
4. Both-corner success at5/10 px, with all two-corner-eligible objects in the denominator, including misses.
5. Outputs on reference-unlocatable sides, where hidden coordinate error cannot be measured.
6. Conditional errors and changes on the same returned endpoint identities.

Misses are first priority. A candidate that reduces misses while adding false positives presents a tradeoff to review; an acceptable extra-detection increase has not been numerically specified. The new report therefore uses **`review-required-no-promotion`**. It does not reuse the earlier historical five-measure adoption rule as if the user had selected it for this new priority. Keep epoch173 as the working default pending the full comparison.

A one-seed result on this repeatedly used development set cannot establish independent-test accuracy. Related frames and endpoints are correlated. Correcting labels can improve the measurement and training supervision; it cannot manufacture new independent scenes. Preserve unfavorable results and inspect changed failure cases before spending on another experiment.

### Replay entry point for the returned evidence

The generic result ZIP contains `reference/` and `candidate/` completed bundles. Extract or copy them under `checkpoints/{reference,candidate}` in the evaluation evidence directory. Produce fresh `m0/`, `m1/`, `m2/` prediction runs and corresponding `m0-quality/`, `m1-quality/`, `m2-quality/` directories using the existing evaluator/quality commands. All three runs use the same clean source revision.

```bash
python scripts/rescore_annotation_comparison.py \
  --root . \
  --evidence ../annotation-evidence/seed-45 \
  --candidate data/candidates/annotations-2026-11-10/manifest.json \
  --recipe reviewed-annotations-seed45-v1 \
  --reference-dir data/references/visible-front-2026-11-10 \
  --evaluation-revision "$ANNOTATION_EVAL_REVISION" \
  --output ../annotation-evidence/seed-45/comparison-original-and-pr30.json
```

`ANNOTATION_EVAL_REVISION` is the full commit SHA used by all fresh prediction and quality runs. The wrapper validates complete paired GPU receipts, candidate and recipe identity, frozen/revised label hashes, source compatibility and chronology before scoring. It preserves all threshold results and reports model effects separately from label-reference effects. It neither loads weights nor trains a model.

## Preparation evidence

Immutable training implementation: [`1c5d2e4`](https://github.com/Siuuuers/dl/tree/1c5d2e48ab26c578c5f6660010d6ecfb33231eb8). The delivered notebook pins that commit; its subsequent delivery commit adds the notebook and this verification record without altering training code.

| Check | Result |
| --- | --- |
| Mixed-label, existing-box, checkpoint, runner, paired-report and result-package checks | 77 passing focused checks |
| Final annotation review | All eight changed images accepted; all 6,839 existing visible corner marks preserved |
| Actual input preflight | All157 original PNGs and epoch173 bytes verified |
| Complete paired staging | 314 copied images and314 copied labels verified; original validation identical in both arms |
| Delivered notebook | Seven code cells and embedded preflight compile; exact input preflight executes locally |
| GPU training or new-model inference | Not performed here |

The [verification record](../results/verification/baomai-training-annotations-20261110/verification.json) records the final source hashes, exact checks, real-input preflight and complete paired staging. It distinguishes synthetic checkpoint fixtures and notebook setup probes from real training. Final annotation review confirms exact source immutability and endpoint preservation. No completed seed45 training checkpoint or model-gain claim exists at this preparation stage.
