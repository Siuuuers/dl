# Continue the baomai image-localization experiment

**Current result, 2026-11-07 (Asia/Hong_Kong): retain epoch173 after the completed reviewed-box experiment.** The returned pair in [PR #23](https://github.com/Siuuuers/dl/pull/23) has passed artifact and source audits, fresh three-model inference, independent metric recount and saved-prediction replay. [The evaluation PR](https://github.com/Siuuuers/dl/pull/24) and [pinned result report](https://github.com/Siuuuers/dl/blob/5f2adb2a2d4aeecf87c29545406b50645e91ab5d/docs/box-supervision-comparison-2026-11-07.md) preserve the result, examples and reproducible evidence. No rerun of the completed seed-42 GPU pair is needed.

At the primary confidence 0.25, M0 epoch173 has 39 misses and 63 extras; the fresh original-label reference has 37 and 60; the corrected-box candidate has 36 and 59. The candidate improves all five primary measures against M0, but its p95 corner error is 5.096371 px versus the matched reference's 5.004297 px. It therefore fails the predeclared requirement against both comparators. The higher p95 remains when comparing exactly the same 1,190 returned endpoints. Keep the candidate and its 16 reviewed box corrections as promising; keep epoch173, box confidence 0.25 and NMS 0.70 as defaults.

**Next bounded question:** determine whether the detection/corner tradeoff repeats across predeclared matched training seeds. Before another GPU run, fix the additional seed list and paired summary method, keep the same original/corrected label arms and finalized-checkpoint selection, and report every result. Do not select the best seed, change the old adoption rule, or treat replication on this reused development set as unseen-scene validation. This is a proposed next experiment; no new seed study has been executed. No architecture change or additional user answer is required to finish the present evaluation.

Working branch: `codex/baomai-box-evaluation-20261107`, delivery `5f2adb2a2d4aeecf87c29545406b50645e91ab5d`. Training/inference source is `918b2fdb811bd595fc09de9a423df6c40258c55a`; the clean comparison source is `66ba9760935701694ce115b4b6c14ffeb146e5d9`. The original [PR #22 plan](https://github.com/Siuuuers/dl/blob/99c8df3116a17984dcf5d0e0421495b6c633b95d/docs/box-supervision-2026-11-07.md) remains the record of what was specified before full training.

**Current inference deliverable:** [PR #21](https://github.com/Siuuuers/dl/pull/21) completes the interrupted existing-data NMS comparison and adds an optional `--nms-iou 0.68` setting to the verified folder command. The fresh-extraction replay and independent implementation review passed. **Keep epoch173, box confidence 0.25 and default NMS 0.70.** Use the latest folder guide below.

**Earlier warmup decision:** retain epoch173 as the default and preserve the lower-bias warmup model as a promising candidate. The GPU pair returned in [PR #16](https://github.com/Siuuuers/dl/pull/16) has now been audited and evaluated. Warmup improves all five primary measures against M0, but misses 37 objects versus the fresh control's 36, so it does not pass the predeclared requirement against both comparators.

The completed result, figures, replay evidence and notebook setup repair are in [the latest comparison PR](https://github.com/Siuuuers/dl/pull/17) and [the pinned report](https://github.com/Siuuuers/dl/blob/2799b8332bb46ba07d0ef95652d7f398559573c9/docs/warmup-comparison-2026-11-07.md). Do not repeat the successful GPU pair solely because the notebook setup cell was repaired.

The targeted detection review is complete in [PR #18](https://github.com/Siuuuers/dl/pull/18). Its three selected additional frames now have a complete 129-row annotation review and separately versioned partial corrections in **[PR #19](https://github.com/Siuuuers/dl/pull/19)** and [the pinned report](https://github.com/Siuuuers/dl/blob/664d4af5cd76f77c52076c96b306f1e90d193e4b/docs/additional-annotation-pilot-2026-11-07.md). The proposal changes 87 boxes and five corner positions, preserving all objects and visibility states. **All three images remain quarantined for 36 unresolved box boundaries; no supplementary scoring reference has been released.** The annotation follow-up remains pending where source material cannot settle those physical boundaries; it does not block the folder-inference deliverable.

The user's current duty is finding/localizing cassette objects in supplied baomai D405 images, preserving the existing one-class, two-lower-corner target. Camera settings are unknown. Camera/robot calibration and physical grasp qualification are outside this deliverable and are not prerequisites for image training.

## Completed still-image folder deliverable

The user selected **a folder of still images**, **current repository material only**, and **no priority between missed and extra detections**. Those choices are sufficient for this delivery. Unknown camera settings and unresolved reference surfaces do not prevent image-space inference.

[PR #20](https://github.com/Siuuuers/dl/pull/20) introduced `scripts/predict_pose_folder.py`; [PR #21](https://github.com/Siuuuers/dl/pull/21) adds the optional NMS setting. That folder-inference release is on `codex/baomai-postprocess-review-20261107`, published at [`79069d3`](https://github.com/Siuuuers/dl/commit/79069d31a25d5229166cdc2ba2b2cfed7ca47a25). Follow [the current folder guide](https://github.com/Siuuuers/dl/blob/79069d31a25d5229166cdc2ba2b2cfed7ca47a25/docs/predict-pose-folder.md) for the correct checkout, model-only Git LFS download and isolated runtime. The old evaluator-only checkout does not contain this command.

```bash
.venv/bin/python scripts/predict_pose_folder.py \
  --input /path/to/original-images \
  --output /path/to/new-results \
  --device cpu
```

Use the Python from an existing compatible CUDA environment and `--device 0` when desired. The command writes original-size annotated PNGs, raw predictions with endpoint display flags, a per-image CSV and explicit run status. Annotation TXT files are unnecessary. Require process exit 0 and `status: complete`; unreadable images, aborted runs and successful zero-detection images remain distinct.

Real CPU inference at clean source `df9254ac16f61eb6f5f858dfd4737459d4133a7e` produced 53 predictions on `new_000007.png` and zero on `new_000146.png`. Both complete raw prediction lists exactly match the existing evaluator in the same runtime. Eleven focused checks passed and were independently rerun during recovery. For PR #20, the final executable hashes matched its tested source; its last commit only clarified checkout and interpreter instructions.

The [demo ZIP](https://github.com/Siuuuers/dl/blob/c234a30bf6208261e973fd39ca33165efa029a31/results/prediction/baomai-folder-20261107/demo.zip) preserves original inputs, outputs and command evidence. Its 19 entries, CRC and all inventoried payload hashes passed inspection. The [verification record](https://github.com/Siuuuers/dl/blob/c234a30bf6208261e973fd39ca33165efa029a31/results/prediction/baomai-folder-20261107/verification.json) retains the actual executed revision. The published candidate tree matches the reviewed local files.

The original folder-inference publication is complete. The optional-NMS continuation below is also published and replay-verified. Prediction counts do not measure false positives, false negatives or accuracy without a valid reference. The existing 0.25 box / 0.50 endpoint thresholds remain operating settings, not a proven optimal error tradeoff.

## Completed existing-data NMS comparison and readiness assessment

The [pinned result and failure examples](https://github.com/Siuuuers/dl/blob/79069d31a25d5229166cdc2ba2b2cfed7ca47a25/docs/postprocess-experiment-2026-11-07.md) preserve the fixed, pre-inference plan, all 240 image/settings predictions and the failed primary hypothesis. At box confidence 0.25:

| NMS IoU | Role | Missed objects | Extra detections | Missing visible endpoints | Declared guard |
| --- | --- | ---: | ---: | ---: | --- |
| 0.70 | Default control | 39 | 63 | 43 | Reference |
| 0.60 | Primary hypothesis | 40 | 46 | 44 | Fails |
| 0.68 | Secondary candidate | 39 | 61 | 43 | Passes on this reference |
| 0.50 | Secondary candidate | 41 | 34 | 45 | Fails |

At 0.68 all 625 matched objects retain identical full prediction payloads and corner errors. Seven unmatched outputs disappear and five appear across four images; duplicate FP stays 17, while other FP falls 46 to 44. Describe this as two fewer extras, not two duplicates removed. The 0.60/0.50 settings lose one/two reference matches in `new_000009`. Keep default NMS 0.70; the optional command adds `--conf 0.25 --nms-iou 0.68`.

The [evidence archive](https://github.com/Siuuuers/dl/blob/79069d31a25d5229166cdc2ba2b2cfed7ca47a25/results/evaluation/baomai-postprocess-20261107/evidence.zip) is 1,551,707 bytes, SHA-256 `761b4c94a812b3933a10ac9ca5fe30a8c63d62eb96143fb459d726977090cc6a`. Recovery independently checked all 33 unique safe entries, CRC and 32 inventoried payload hashes. Fresh extraction reproduced all 240 score records and 12 aggregate records; predictions, summary, per-image results and CSV are byte-identical. All 60 historical baseline payloads and the six saved folder-run payloads match exactly. Seven experiment and 14 folder checks passed again. Published tree `906ddc53edb82cfbb8df80faa8d90db6e78ea662` matches the audited source; the six executed helpers retain their recorded hashes. No model rerun or training was needed for recovery.

**Readiness remains limited to a human-reviewed still-image workflow.** All 17 nonempty development images have at least one measured FN or FP. Objects with one/neither corner locatable account for 25 of 39 misses, an association rather than a causal occlusion claim. Conditional corner error excludes missing endpoints; endpoint confidence does not certify physical visibility. Replaying the same evidence verifies reproducibility, not independent generalization.

## Completed reviewed-box GPU comparison

The review and training handoff in [PR #22](https://github.com/Siuuuers/dl/pull/22) covered all 104 rows in three training images, plus a genuine empty control. Its separate candidate changes 16 supported box edges: eight in `new_000030`, six in `new_000012`, and two in `new_000128`. All object identities, classes and endpoint tokens are retained. Seven legacy instance identities and several other bounds remain explicitly unresolved and unchanged. The original eleven-row deletion proposal and quarantined additional-frame proposal remain separate.

The return in [PR #23](https://github.com/Siuuuers/dl/pull/23), commit `161f5f20f3aef543cbe2a291db8d510b1123e769`, contains both finalized epoch-30 checkpoints. Both record 1,050 batches, 4,110 image presentations, 78 actual optimizer updates and successful validation/finalization. Clean source `918b2fd`, the runtime, all source hashes and 114 resolved settings match except three run/data paths. The third sampled batch includes two corrected images in both arms and records differing losses and real parameter updates, establishing that the changed supervision reached training. AMP is disabled, but cuDNN TF32 is enabled and deterministic mode is warning-only; no strict bitwise GPU guarantee is claimed.

Both arms use `box-supervision-warmup-v1`, seed 42, batch 4, image size 1280, AdamW and bias warmup LR 0.0001, starting independently from the original epoch173. Both retain 137 training images, 3,772 objects and 6,839 visible endpoints. The 20-image validation reference remains identical. All 13 returned archive members, CRCs and payload hashes passed; recorded staging inventories were reconciled with pinned source bytes. The ZIP is 39,775,283 bytes, SHA-256 `20d3138d5dcc35b03ab218f9cdd7e6e9b7cc327795f34f26be225a765b2c3009`.

Fresh M0/M1/M2 inference completed all 180 image/model/threshold records in one CPU environment. At box confidence 0.25, matching IoU 0.50, endpoint confidence 0.50 and default NMS 0.70:

| Measure | M0 epoch173 | M1 original-label reference | M2 corrected-box candidate |
| --- | ---: | ---: | ---: |
| Missed objects | 39 | 37 | 36 |
| Extra detections | 63 | 60 | 59 |
| Missing visible endpoints | 43 | 37 | 36 |
| Conditional median error, px | 1.611790 | 1.610761 | 1.602810 |
| Conditional p95 error, px | 5.161683 | 5.004297 | 5.096371 |
| Both corners within illustrative 5 px / 584 | 521 | 524 | 519 |

**Decision: retain M0.** The candidate passes against M0 but fails p95 against M1. The isolated label effect is M2 versus M1; M2 versus M0 also includes fine-tuning. M1's baseline pass does not authorize adopting a previous warmup result that failed its own predeclared experiment. No threshold, tolerance or rule is changed after seeing the result.

M2 retains every M1 match and gains `new_000009`, row 52. Its qualifying confidence 0.265436 is close to the fixed cutoff. On all 1,190 shared returned endpoints, p95 still rises from 5.004297 to 5.096383 px; the new endpoint has error 3.485439 px and does not cause this increase. At the illustrative 5 px radius, six two-corner successes become failures and one failure becomes a success. At the primary box confidence of 0.25, all 17 nonempty validation images retain at least one measured FN or FP under every model. Three empty scenes produce no detections at all three thresholds. These are development-set observations, not application acceptance thresholds.

The [result report](https://github.com/Siuuuers/dl/blob/5f2adb2a2d4aeecf87c29545406b50645e91ab5d/docs/box-supervision-comparison-2026-11-07.md), [unrounded comparison](https://github.com/Siuuuers/dl/blob/5f2adb2a2d4aeecf87c29545406b50645e91ab5d/docs/box-supervision-comparison-2026-11-07.json), and [verification record](https://github.com/Siuuuers/dl/blob/5f2adb2a2d4aeecf87c29545406b50645e91ab5d/results/evaluation/baomai-box-supervision-a100-001/verification.json) link the evidence capsule. The independent recount implements matching separately; the clean `66ba976` summarizer enforces this recipe, candidate, source and both-comparator rule. Fresh extraction reproduces the result from saved predictions without loading weights. Fresh M0 and M1 prediction records also match their historical M0/warmup counterparts after excluding only timing fields.

The earlier failed CPU trials and successful one-batch smoke remain documented in the original handoff; they are no longer the latest training status. The completed GPU return qualifies this one experiment. Keep the original notebook for reproduction, not as an outstanding request to rerun it. The next proposed study concerns repeatability across matched seeds; no changes to the still-image architecture are justified by this result alone.

## Completed warmup checkpoint comparison

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

## Completed three-image annotation adaptation

The [candidate manifest and all-row ledgers](https://github.com/Siuuuers/dl/tree/664d4af5cd76f77c52076c96b306f1e90d193e4b/data/candidates/additional-pilot-2026-11-07) retain exact upstream source labels beside three separately versioned proposals. Original pixels were inspected before supplied overlays; model predictions were not used. Three reviewers covered every row and missing/extra/duplicate-instance scans, followed by root comparison inspection and a separate consistency audit.

| Frame | Rows | Changed boxes | Corrected endpoint positions | Unresolved rows |
| --- | ---: | ---: | ---: | ---: |
| frame_0001 | 43 | 6 | 1 | 13 |
| frame_0160 | 43 | 39 | 4 | 11 |
| frame_0383 | 43 | 42 | 0 | 12 |
| Total | 129 | 87 | 5 | 36 |

No objects were added or removed. All 256 visible endpoints and two canonical missing endpoints remain in the same visibility states. The frame_0001 missing right corner on row 42 and missing left corner on row 43 stay missing behind the gripper. The corrected points are the frame_0001 row 30 right corner and frame_0160 row 34 left, row 40 right, and both row 42 corners.

Thirty-five unresolved upper extents concern exposed gray material that could be a front-facing lip or upper/roof/rim surface. Connection to the cassette does not establish surface orientation. Frame_0001 row 43 also has an unresolved left box extent beside the gripper. The relevant source bounds are retained only as unaccepted placeholders, while independently supported other-edge and corner corrections are saved. Four initially proposed frame_0383 top expansions were withdrawn after the physical interpretation was challenged. Every affected whole image is quarantined; no difficult object is removed or converted into an empty negative.

This completes the selected three-image review, not all 320 images. The remaining 317 do not have a complete row review. Two additional originals, frame_0022 and frame_0354, were viewed only for surface context; no labels or coordinates were transferred from them. The recovered pool's prior-use and correlated-frame limits remain.

## Annotation follow-up and evidence limits

**Goal:** resolve the recorded physical surface ambiguities under the existing visible-front policy, then continue a separately versioned, explicitly selected annotation batch before freezing any supplementary scoring reference.

Start from the exact candidate bytes and unresolved row lists in the new manifest. Seek an unequivocal original view or part reference that distinguishes the front-facing lip from the upper/roof/rim plane. Apply that physical interpretation consistently across related cases; gray front material remains target regardless of brightness. Resolve the row 43 left visible extent separately. If the available evidence remains insufficient, retain whole-image quarantine and document the limitation. Do not force a top coordinate, infer a hidden corner, discard hard rows to make an image scoreable, or change the target definition merely to fit supplied boxes.

Review complete images and record source hashes, one-based row identities, each box edge and each endpoint state. Freeze eligible IDs and exact adapted label bytes only after their complete dispositions are resolved. Exclude old-image families even when re-encoded. Capture/session and prior-use limits must accompany the future supplementary reference; no independent-test claim is established.

The candidate currently has an empty eligible-image list, so it cannot support training or reference-based scoring. Label-free inference can still run on original images for inspection; its predictions do not approve the candidate labels. Keep the existing 20-image development manifest, scorer and preflight guard intact. When a nonempty reference is ready, use an explicit separate manifest and the same scorer/checkpoint-provenance rules to compare fixed M0/control/warmup checkpoints in one common runtime with preserved raw predictions. Another training change should follow a specific reviewed finding.

## Reproduce or inspect the completed work

The new [annotation report and verification](https://github.com/Siuuuers/dl/blob/664d4af5cd76f77c52076c96b306f1e90d193e4b/docs/additional-annotation-pilot-2026-11-07.md) link a review ZIP containing the three original JPEGs, lossless decoded-RGB PNG/TXT editor pairs, exact source/proposal labels, all-row ledgers, nine comparisons and integrity tools. PNG exports preserve decoded RGB pixels and 1280x720 coordinates; they are not native original PNGs. The package retains quarantine and makes no browser Save/reopen acceptance claim. See its structured verification record for archive hashes and fresh-restore results.

The latest [case/intake review](https://github.com/Siuuuers/dl/blob/284c62377d3a210f3deeadbbedbd7b749002768a/docs/warmup-detection-review-2026-11-07.md) and [structured verification record](https://github.com/Siuuuers/dl/blob/284c62377d3a210f3deeadbbedbd7b749002768a/docs/warmup-detection-review-2026-11-07.json) are preserved with a 34-file capsule at `results/review/baomai-warmup-detection-20261107/review.zip`. All 17 case figures and six intake figures reproduce byte-for-byte, and fresh extraction reproduces all seven cohort counts. The published tree and all 14 changed-file bytes were read back. This phase ran no new inference or training.

Use the [latest report](https://github.com/Siuuuers/dl/blob/2799b8332bb46ba07d0ef95652d7f398559573c9/docs/warmup-comparison-2026-11-07.md) and [verification record](https://github.com/Siuuuers/dl/blob/2799b8332bb46ba07d0ef95652d7f398559573c9/docs/warmup-comparison-verification-2026-11-07.json). The compact archive is `results/evaluation/baomai-warmup-a100-001/evaluation.zip`. Its inventory and whole-archive checksum were read back; a fresh extraction replays successfully. The standard-library `scripts/summarize_warmup_comparison.py` validates original saved scores, correct recipe roles, completed runs and matched provenance before deriving the decision. It needs original validation PNGs/labels but no weights or machine-learning runtime.

For a fresh comparison on the frozen labeled development set, use `scripts/evaluate_pose.py` with each explicit trained-checkpoint manifest; omitting it selects the frozen M0. Keep image size 1280, rectangular preprocessing, FP32, ordinary NMS at IoU 0.70, batch 1, `max_det=300`, box confidences 0.15/0.25/0.40, matching IoU 0.50 and endpoint confidence 0.50. Use the same device/runtime across models. The baseline replacement rule requires all five unrounded metrics to be no worse and at least one better. Warmup must pass against both M0 and its matched control. Missing statistics do not pass.

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
| Complete three-frame annotation review, partial corrections and quarantine | [PR #19](https://github.com/Siuuuers/dl/pull/19) |
| Label-free folder inference, verified examples and completed publication | [PR #20](https://github.com/Siuuuers/dl/pull/20) |
| Fixed existing-data NMS comparison, optional 0.68 and completed recovery | [PR #21](https://github.com/Siuuuers/dl/pull/21) |
| Reviewed training boxes, pinned GPU handoff and real candidate smoke | [PR #22](https://github.com/Siuuuers/dl/pull/22) |
| Completed original/corrected-box GPU return | [PR #23](https://github.com/Siuuuers/dl/pull/23) |
| Reviewed-box evaluation, independent recount and replay evidence | [Evaluation PR](https://github.com/Siuuuers/dl/pull/24) |

The reviewed-box evaluation is stacked on PR #22, whose experiment is stacked on PR #21. The optional-NMS delivery is stacked on PR #20, whose folder-inference delivery is stacked on PR #19. The annotation proposal is stacked on PR #18, which is stacked on PR #17; the earlier comparison is stacked on #15/#14/#12/#11/#10/#9/#8/#7. PRs #13, #16 and #23 contain separate main-based result archives. Inspect current heads before continuing; no merge is implied by this workflow update. Use the appropriate branch rather than rebuilding code missing from main.

The baseline hash is `e0aab1b377491700318e0b14cd4c558ec3c1684087a13b7dde2298ebdbf2847e`. Its reference has 137 training images, 20 validation images, 3,772/664 objects and 13/3 empty scenes. Preserve `cassette_front`, `kpt_shape: [2, 3]`, `flip_idx: [1, 0]` and image-X corner order. The frozen reference uses raw 1280×720 PNGs; preserve the additional pool's original JPEG source bytes separately. Unlocatable endpoints remain exactly `0 0 0`; model predictions do not supply hidden reference coordinates.

The adopted `baomai-visible-front-v1` policy retains identifiable visible fronts even when both target corners are unlocatable, and boxes observed front-face extent. Frozen legacy labels have not been completely migrated or certified. The separate deletion-only candidate has 3,761 objects and retains all 6,839 visible training endpoints. PR #2's historical 3,765-object candidate remains unrecovered and distinct. The first GPU comparison did not show that the eleven-row cleanup improved the model; its report remains an unchanged historical milestone.

The browser label editor and historical label recovery can continue independently. Browser Save/reopen acceptance needs a fresh check of the current editor branches using disposable annotation copies; older access trouble did not establish an HTML defect.

This remains a one-seed development comparison on data already used for selection. Related objects, endpoints and images are not independent deployment trials. No independent-test, millimeter-accuracy or physical-grasp claim follows from these results.
