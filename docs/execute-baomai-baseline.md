# Continue the baomai image-localization experiment

**Current decision, 2026-11-07 (Asia/Hong_Kong): retain epoch173 as the default and preserve the lower-bias warmup model as a promising candidate.** The GPU pair returned in [PR #16](https://github.com/Siuuuers/dl/pull/16) has now been audited and evaluated. Warmup improves all five primary measures against M0, but misses 37 objects versus the fresh control's 36, so it does not pass the predeclared requirement against both comparators.

The completed result, figures, replay evidence and notebook setup repair are in [the latest comparison PR](https://github.com/Siuuuers/dl/pull/17) and [the pinned report](https://github.com/Siuuuers/dl/blob/2799b8332bb46ba07d0ef95652d7f398559573c9/docs/warmup-comparison-2026-11-07.md). Do not repeat the successful GPU pair solely because the notebook setup cell was repaired.

The targeted follow-up is now complete in **[PR #18](https://github.com/Siuuuers/dl/pull/18)** and [its pinned illustrated review](https://github.com/Siuuuers/dl/blob/284c62377d3a210f3deeadbbedbd7b749002768a/docs/warmup-detection-review-2026-11-07.md). All nine lost and eight gained matches were reviewed. A recovered 320-frame package supplies additional images and labels, but a model-blind three-image pilot found boxes extending into the rack/base. **Next: a separately versioned annotation adaptation before supplementary scoring.**

The user's current duty is finding/localizing cassette objects in supplied baomai D405 images, preserving the existing one-class, two-lower-corner target. Camera settings are unknown. Camera/robot calibration and physical grasp qualification are outside this deliverable and are not prerequisites for image training.

## Latest completed comparison

PR #16's original archive is `results/gpu/baomai-warmup-a100-001/baomai-warmup-results.zip` at commit `dc388867bb3b59ab376c8cdc1875d96e1665f99d`. It has 39,776,499 bytes and SHA-256 `2e8ca425c585a17468d8c651014e811681ccefe2a0f1a6a579a920178d560a6a`. Its complete inventory, payload hashes, CRC and both finalized checkpoint bundles passed validation.

The fresh control and warmup runs use original reference labels. Both started from the exact same epoch173 bytes and completed 30 epochs, 1,050 batches, 4,110 image presentations and 78 actual optimizer updates on the recorded A100 environment. Clean training source is `679b87b38e7b0bca5fbd02fb1519a90fb2379541`. Of 114 resolved settings, only bias warmup and three run paths differ. The only scientific change is `warmup_bias_lr: 0.1 -> 0.0001`; nominal `lr0` remains 0.0001.

Fresh M0/control/warmup inference ran together in one pinned CPU environment from that clean revision. Each completed 60 image/threshold pairs. The comparison was replayed from clean published code `2f0260c19835fcd60eb2f12040b7c77ebd2ca890`, then preserved in the latest result PR. Finalized epoch-30 `last.pt`, not the training validator's automatic best checkpoint, remains the selection contract.

At primary box confidence 0.25, matching IoU 0.50 and endpoint confidence 0.50:

| Measure | M0 epoch173 | Fresh control | Warmup |
| --- | ---: | ---: | ---: |
| Missed objects | 39 | 36 | 37 |
| Extra detections | 63 | 80 | 60 |
| Missing visible endpoints | 43 | 43 | 37 |
| Conditional median error, px | 1.611790 | 2.468005 | 1.610761 |
| Conditional p95 error, px | 5.161683 | 6.326759 | 5.004297 |
| Both corners within illustrative 5 px / 584 | 521 | 479 | 524 |

The warmup model largely recovers the original fine-tune's corner regression. Its median improvement over M0 is only about 0.00103 px. It retains all 1,184 endpoints returned by M0 and adds six, but on those same 1,184 endpoints improves 588 and worsens 596, with mean error change +0.00727 px. Do not describe it as uniformly more accurate.

The fresh control reproduces all 60 historical M1 prediction records exactly after excluding only timing fields. This strengthens the bounded recipe comparison. It does not establish checkpoint byte identity or general improvement across seeds. At secondary confidence 0.40 warmup passes both comparisons; that does not replace the declared primary 0.25 criterion.

The net one-object deficit against control consists of **nine lost matches and eight gained matches**. Against M0, warmup loses one and gains three. The sole M0-detected loss is `new_000010`, one-based label row 14: an object largely covered by the gripper, with neither reference corner locatable. It still counts in object recall under the adopted visible-front policy.

## Completed case review and additional-frame intake

The complete 17-case review spans ten validation originals. Seven of eight warmup gains have a high-confidence control prediction overlapping the target region below the fixed IoU gate; their qualifying warmup IoUs are only 0.500357–0.533900. In one illustrated case, every predicted box edge moves by less than one pixel while the match flips. Original images raise reference-box extent questions on both gained and lost sides, without establishing model predictions as annotation truth.

| Reference point availability | Objects | M0 misses | Control misses | Warmup misses |
| --- | ---: | ---: | ---: | ---: |
| Both corners visible | 584 | 14 | 14 | 11 |
| One corner visible | 59 | 15 | 15 | 15 |
| Neither corner locatable | 21 | 10 | 7 | 11 |

These groups partition the original 664 objects. They do not redefine occlusion categories or exempt difficult objects from recall. No duplicate deletion or corrected coordinate is approved by this review; the original comparison and adoption rule remain intact.

The actual upstream reviewed320 ZIP was recovered from `an94-doll/embedding_vision`, grasp-vision commit `9a6bf2432582af517d789c94eab2ac27e03a304b`. All 320 images and 320 labels match the pinned source hashes; structural checks cover 12,987 objects with no reported violation. Source ZIP: 43,424,432 bytes, SHA256 `b28bcce5fffe91f3e486ff11ec3085a67c868fe09e65c14788bc6fe494af4dd0`. [The inventory](https://github.com/Siuuuers/dl/blob/284c62377d3a210f3deeadbbedbd7b749002768a/docs/additional-frame-intake-2026-11-07.json) and the review capsule preserve source identities, lineage and a bounded recovery recipe.

The 320 raw-image hashes do not overlap the frozen 157. Nevertheless, upstream's 48-test listing reuses 15 current-training and three current-validation images from its combined 477-source pool. Re-encoding those PNGs as JPEGs does not remove their lineage. The 320 additions also have historical upstream model usage and a random frame split with possible neighboring-frame overlap; they are not a certified independent test. The original excluded five final-test and nineteen gap images remain unidentified as those specific sets.

The three-image intake pilot inspected raw originals before supplied-label overlays, without model predictions. Target identity and endpoint intent appear compatible, but `frame_0160` rows 41–43 extend beneath the white fronts into the metal rack/base. Its row 43 box ends at y=666.12 px while the supplied lower endpoints are near y=647.36/647.93 px; raw pixels identify the extra region as rack. These supplied labels are not ready as-is under `baomai-visible-front-v1`. The pilot does not approve all 129 rows or the remaining 317 images.

## Next bounded assignment

**Goal:** prepare a separately versioned annotation-adaptation candidate from the recovered additional frames, then freeze a supplementary reference before scoring the fixed checkpoints.

Preserve the upstream source image/label hashes and original one-based row identities. Review each included physical front, its observed front-face box, and each actual lower-front endpoint independently under the adopted policy. Preserve canonical missing states; do not infer hidden points, automatically snap boxes to endpoints, or use model agreement as annotation truth. Record or quarantine unresolved images before releasing a scoring reference.

Freeze eligible image IDs and exact adapted label bytes with review dispositions. Exclude all 157 old-image families even when re-encoded. Record capture/session and prior-use limits; if independent capture grouping cannot be established, label the future result as supplementary frame validation with correlated-scene limits.

Keep this reference separate from the frozen 20-image development manifest. The current `preflight_dataset` guard deliberately requires that original reference and should remain intact. A supplementary evaluator needs an explicit separate manifest while reusing the existing scorer and checkpoint-provenance checks. Compare M0, control and warmup in one common runtime at the same primary settings, and preserve predictions for replay.

M0 remains the working model and warmup stays frozen. No additional training is needed to prepare or run that checkpoint comparison. Another training change should follow a specific reviewed finding. The original GPU pair does not need to be repeated for the repaired notebook setup cell.

## Reproduce or inspect the completed work

The latest [case/intake review](https://github.com/Siuuuers/dl/blob/284c62377d3a210f3deeadbbedbd7b749002768a/docs/warmup-detection-review-2026-11-07.md) and [structured verification record](https://github.com/Siuuuers/dl/blob/284c62377d3a210f3deeadbbedbd7b749002768a/docs/warmup-detection-review-2026-11-07.json) are preserved with a 34-file capsule at `results/review/baomai-warmup-detection-20261107/review.zip`. All 17 case figures and six intake figures reproduce byte-for-byte, and fresh extraction reproduces all seven cohort counts. The published tree and all 14 changed-file bytes were read back. This phase ran no new inference or training.

Use the [latest report](https://github.com/Siuuuers/dl/blob/2799b8332bb46ba07d0ef95652d7f398559573c9/docs/warmup-comparison-2026-11-07.md) and [verification record](https://github.com/Siuuuers/dl/blob/2799b8332bb46ba07d0ef95652d7f398559573c9/docs/warmup-comparison-verification-2026-11-07.json). The compact archive is `results/evaluation/baomai-warmup-a100-001/evaluation.zip`. Its inventory and whole-archive checksum were read back; a fresh extraction replays successfully. The standard-library `scripts/summarize_warmup_comparison.py` validates original saved scores, correct recipe roles, completed runs and matched provenance before deriving the decision. It needs original validation PNGs/labels but no weights or machine-learning runtime.

For new inference, use `scripts/evaluate_pose.py` with each explicit trained-checkpoint manifest; omitting it selects the frozen M0. Keep image size 1280, rectangular preprocessing, FP32, ordinary NMS at IoU 0.70, batch 1, `max_det=300`, box confidences 0.15/0.25/0.40, matching IoU 0.50 and endpoint confidence 0.50. Use the same device/runtime across models. The baseline replacement rule requires all five unrounded metrics to be no worse and at least one better. Warmup must pass against both M0 and its matched control. Missing statistics do not pass.

The [repaired Colab notebook](https://colab.research.google.com/github/Siuuuers/dl/blob/2799b8332bb46ba07d0ef95652d7f398559573c9/notebooks/baomai_bias_warmup.ipynb) and [handoff](https://github.com/Siuuuers/dl/blob/2799b8332bb46ba07d0ef95652d7f398559573c9/docs/gpu-warmup-handoff.md) preserve the experiment for reproduction. The setup cell now writes installed Torch version constraints directly, removing the nested-Python syntax error reported in PR #16. Two focused checks cover correct constraints before pip and a missing-metadata failure. All notebook code cells compile. The training source pin is unchanged, and the returned pair passed its scientific source/settings audit.

## Earlier work and project invariants

| Work | Source |
| --- | --- |
| Original experiment and conservative adoption rule | [PR #1](https://github.com/Siuuuers/dl/pull/1) |
| Baseline evaluator and first real inference | [PR #7](https://github.com/Siuuuers/dl/pull/7) |
| Validation review and direct-pixel diagnostics | [PR #8](https://github.com/Siuuuers/dl/pull/8) |
| Thirteen-image training pilot and eleven-row candidate | [PR #9](https://github.com/Siuuuers/dl/pull/9) |
| Adopted annotation policy and paired-data staging | [PR #10](https://github.com/Siuuuers/dl/pull/10) |
| Training runner and completed-checkpoint provenance | [PR #11](https://github.com/Siuuuers/dl/pull/11) |
| Portable input/return packaging | [PR #12](https://github.com/Siuuuers/dl/pull/12) |
| Original-label / cleaned-label GPU models | [PR #13](https://github.com/Siuuuers/dl/pull/13) |
| First matched comparison, retaining M0 | [PR #14](https://github.com/Siuuuers/dl/pull/14) |
| Isolated warmup recipes and original handoff | [PR #15](https://github.com/Siuuuers/dl/pull/15) |
| Fresh control and warmup GPU return | [PR #16](https://github.com/Siuuuers/dl/pull/16) |
| Warmup comparison, evidence and setup repair | [Latest comparison PR](https://github.com/Siuuuers/dl/pull/17) |
| Completed detection review, recovered320 intake and semantic pilot | [PR #18](https://github.com/Siuuuers/dl/pull/18) |

The latest review is stacked on PR #17, whose comparison is stacked on #15/#14/#12/#11/#10/#9/#8/#7. PRs #13 and #16 contain separate main-based result archives. Inspect current heads before continuing; no merge is implied by this workflow update. Use the appropriate branch rather than rebuilding code missing from main.

The baseline hash is `e0aab1b377491700318e0b14cd4c558ec3c1684087a13b7dde2298ebdbf2847e`. Its reference has 137 training images, 20 validation images, 3,772/664 objects and 13/3 empty scenes. Preserve `cassette_front`, `kpt_shape: [2, 3]`, `flip_idx: [1, 0]` and image-X corner order. The frozen reference uses raw 1280×720 PNGs; preserve the additional pool's original JPEG source bytes separately. Unlocatable endpoints remain exactly `0 0 0`; model predictions do not supply hidden reference coordinates.

The adopted `baomai-visible-front-v1` policy retains identifiable visible fronts even when both target corners are unlocatable, and boxes observed front-face extent. Frozen legacy labels have not been completely migrated or certified. The separate deletion-only candidate has 3,761 objects and retains all 6,839 visible training endpoints. PR #2's historical 3,765-object candidate remains unrecovered and distinct. The first GPU comparison did not show that the eleven-row cleanup improved the model; its report remains an unchanged historical milestone.

The browser label editor and historical label recovery can continue independently. Browser Save/reopen acceptance needs a fresh check of the current editor branches using disposable annotation copies; older access trouble did not establish an HTML defect.

This remains a one-seed development comparison on data already used for selection. Related objects, endpoints and images are not independent deployment trials. No independent-test, millimeter-accuracy or physical-grasp claim follows from these results.
