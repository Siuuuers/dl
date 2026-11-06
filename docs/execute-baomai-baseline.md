# Execute the baomai evaluation and improvement workflow

Start here in the next conversation. The user confirmed that the supervisor requires the upstream `baomai` task and that `Siuuuers/dl` is the working repository. The first offline baseline, focused validation review and targeted training-image pilot have now completed; continue from the recorded evidence. Intended deployment and its tolerances remain unconfirmed.

**Completed on 2026-11-06:** [PR #7](https://github.com/Siuuuers/dl/pull/7) implements the evaluator and records real inference on all 20 unchanged validation images at box confidence 0.15 / 0.25 / 0.40. The executed code was `3bc783f1e53a477a095ac29160674f07e393348d`, with a clean working tree. Read the [measured results report](https://github.com/Siuuuers/dl/blob/2ffd7b9788f29edf057335f2745e2d6efd3ce35f/docs/baseline-2026-11-06.md) and [reproduction guide](https://github.com/Siuuuers/dl/blob/2ffd7b9788f29edf057335f2745e2d6efd3ce35f/docs/evaluate-pose.md) from that PR until merged.

At confidence 0.25, the declared matcher returned 625 TP, 39 FN, 63 FP, and 1,184 / 1,227 visible endpoints (96.50% coverage); conditional median/p95 error was 1.61 / 5.16 px. All 43 missing scored endpoints were on unmatched objects. The 36 focused checks passed; replay reproduced all aggregate and per-image metrics. These remain development-validation diagnostics, not independent final-test accuracy.

**Quality review also completed on 2026-11-06:** [PR #8](https://github.com/Siuuuers/dl/pull/8), stacked on PR #7, adds [quality criteria and the review decisions](https://github.com/Siuuuers/dl/blob/1c201e35a33727ac0ae59f1d11a3d7fbb72b5ae7/docs/model-quality.md) plus source-hashed evidence. All 12 queued rows in `new_000009`, `new_000056` and `new_000090`, and one additional large-error case in `new_000008`, were visually reviewed. No label edits were accepted. The queued prediction for `009` row 8 is a duplicate of neighboring row 9; do not treat it as a certified same-object corner error. The audit of all 137 training label text files passed mechanical checks, but did not visually inspect training PNGs.

At the **illustrative** 5 px radius, box confidence 0.25 and endpoint confidence 0.50, both corners succeed on 521 / 584 fully visible objects (89.21%), retaining missed objects in the denominator. At that setting 61 / 66 reference-unlocatable endpoints on matched objects still receive eligible coordinates. That is a conditional availability disagreement, not hidden-point geometric error. A clean replay at `6ae48d57865611fc9d829e429bbf3b5b387fcf24` completed 60 image/threshold pairs; 12 focused checks passed and the calculations agreed exactly. See the [verification record](https://github.com/Siuuuers/dl/blob/1c201e35a33727ac0ae59f1d11a3d7fbb72b5ae7/docs/quality-replay-verification-2026-11-06.json). No new model inference or training was needed for this stage.

**Training pilot completed on 2026-11-06:** [PR #9](https://github.com/Siuuuers/dl/pull/9), stacked on PR #8, records [the review and proposed overlay](https://github.com/Siuuuers/dl/blob/628741ea0cfe607772bfd3d84f76fb04d74bf7ec/docs/training-pilot-2026-11-06.md). Actual bytes and dimensions were verified for 13 selected originals. The bounded visual scope covered 54 target rows plus one empty-scene control, not every row in those images. Primary and independent agent reviews identified eleven duplicate physical-instance rows in five training images. The proposed overlay retains every other row byte-for-byte: 3,772 → 3,761 training objects, all 6,839 visible endpoints retained, 705 → 683 unlocatable entries. Frozen labels, validation, split and checkpoint remain unchanged. This is a proposed correction, not a human approval or measured accuracy improvement.

The all-training overlap audit checked 75,145 pairs. Twelve exceeded or equaled IoU 0.50, and one additional duplicate was found below that threshold. Two overlapping pairs—`new_000030` rows 6/29 and `new_000123` rows 32/33—are distinct neighboring fronts and must remain. Do not apply prediction-style suppression to annotations. The read-only candidate verifier and four invalid-input probes passed their expected outcomes on clean published revision `42eb13332c081c2c9aa21836fd831bfff509a6aa`; see [the verification record](https://github.com/Siuuuers/dl/blob/628741ea0cfe607772bfd3d84f76fb04d74bf7ec/docs/training-candidate-verification-2026-11-06.json).

**Next task:** review the exact proposed deletions, resolve remaining box-extent/inclusion and application acceptance decisions, and prepare a controlled data comparison. The candidate is a five-file label overlay, not a runnable dataset. Materialize a separate complete paired `images/` and `labels/` tree, apply only the five proposed replacements, verify actual image bytes, and verify the trainer's resolved label paths. A candidate YAML pointing at the original image tree can load the original labels. Keep the frozen manifest and preflight intact. Add explicit candidate-checkpoint provenance support before evaluation. No training or new inference ran during the pilot; retain the baseline until a controlled comparison supports a change.

## 1. Preserve the task and reference

- Inspected workspace: `b4df76935178de5a122119aea48e85d2d661effc`.
- Upstream: [`an94-doll/embedding_vision`, `baomai` at `351eb3cf960ccc34829e2bc72e1fa4645acc6295`](https://github.com/an94-doll/embedding_vision/tree/351eb3cf960ccc34829e2bc72e1fa4645acc6295).
- Model: `weights/cassette_lower_epoch173.pt`, YOLO26s-Pose. SHA-256: `e0aab1b377491700318e0b14cd4c558ec3c1684087a13b7dde2298ebdbf2847e`; recorded size: 106,838,542 bytes.
- One `cassette_front` class; `kpt_shape: [2, 3]`; `flip_idx: [1, 0]`. Predict the actual lower corners of the white front face, ordered by image X. Upper-lip, tray and basket corners are different features.
- Use original 1280×720 PNGs. A label row is `0 cx cy width height left_x left_y left_v right_x right_y right_v`. Coordinates are normalized. Visible points use `v=2`; an unlocatable point uses `0 0 0`.
- The reference has 137 train / 20 validation images, 3,772 / 664 instances, and 13 / 3 empty scenes. `data/annotated/` contains 1280×750 previews with a footer; never use those as model input.

Read [data/README.md](../data/README.md), [data/data.yaml](../data/data.yaml), [model_card.json](../metadata/model_card.json), [training_config.json](../metadata/training_config.json), [weights/base.json](../weights/base.json), and the predictor setup in [view_d405_yolo.py](../scripts/view_d405_yolo.py). Preserve the supplied labels, split and checkpoint. Inspect current main and open PRs before adding a second implementation. Use an isolated branch from the appropriate existing dependency; PR #9 currently depends on PR #8, which depends on PR #7. Do not lose or duplicate that work by assuming every new task must start from main.

At the inspected revision, the committed `data/`, `metadata/`, `models/` and `weights/` trees match upstream. This compared Git blobs, including LFS pointers; downloaded image and checkpoint bytes still require verification at execution time.

## 2. Bounded assignments

| Task / owner | Scope | Depends on | Acceptance evidence |
| --- | --- | --- | --- |
| Baseline / completed in PR #7 | Preflight, offline evaluator, smoke run, 20 images × 3 thresholds | Verified reference bytes and pinned CPU runtime | Saved predictions, metrics, overlays, completion record and exact replay |
| Label recovery / independent assistant if useful | Locate the earlier candidate labels, inventory and compare small text files | Candidate folder, archive or earlier session artifact if available | Actual files, hashes, split, changed rows and review status; otherwise an explicit unresolved location |
| Quality review / completed in PR #8 | 12 queued rows plus one localization case; 137 training-label text files; radius and availability diagnostics | PR #7 originals and saved predictions | Review decisions with zero accepted edits; source-hashed evidence; 12 focused checks and exact replay comparison |
| Targeted training pilot / completed in PR #9 | 13 verified originals, 54 target rows and an empty control; all-training overlap audit | PR #8 source and original pixels | Eleven proposed duplicate removals, five exact label replacements, source-hashed decisions and read-only verification |
| Controlled data comparison / next assistant | Review proposal, resolve remaining policy/tolerance decisions, stage paired dataset, support candidate provenance and follow PR #1 | PR #9 overlay, full verified image bytes and suitable compute | Correct resolved training label paths, recorded dataset/checkpoint hashes and matched comparison against unchanged data |

Label recovery can run in parallel. Its unresolved location does not block evaluation against the committed reference. The targeted review is complete. Do not change the model architecture, expand the target to five points or 3D, or add editor features without an observed need. Begin the separate controlled-training stage only after the proposed dataset is reviewed, fully materialized and verified. A model-confidence cutoff is not a probability of physical correctness; the recorded cutoff replay already exposes substantial tradeoffs. Reuse the completed baseline unless a concrete code, data, runtime or settings change requires another comparison.

## 3. Baseline reproduction, when needed

1. Record the checked-out commit. Recheck the model card, current scripts, and open PRs; this handoff may outlive the inspected revision.
2. For baseline reproduction, obtain only the needed model and validation original PNGs first. In a fresh clone, skip automatic LFS smudging and fetch the selected LFS paths. Training PNGs and annotated previews are not needed to reproduce the baseline. The pilot verified thirteen training originals; a future training run separately requires all intended training image bytes and their verified image/label pairing. Detect pointer text and verify actual bytes against the model card and validation records in `data/manifest.json`.
3. Use an isolated headless inference environment with **Ultralytics 8.4.182**. Record Python, PyTorch, CUDA, device and precision. Reuse a compatible runtime where available; do not silently upgrade dependencies. Offline evaluation does not require a RealSense device or the live viewer's desktop/CUDA setup. Select CPU or an available GPU explicitly and use the same device throughout a comparison.
4. Reuse `scripts/evaluate_pose.py` and its separate `scripts/pose_scoring.py` from PR #7. They did not exist in the starting main revision above, but are now implemented. Check current main and the PR before adding or changing an evaluator. The reproduction guide defines preflight, smoke, sweep and standard-library replay commands.
5. Preflight label semantics: exactly 11 finite fields, class 0, positive box dimensions, normalized fields in range, visibility 0 or 2, canonical `0 0 0` for missing points, and left X ≤ right X when both points are visible. Verify image/label pairing and dimensions; preserve genuine empty files. Report semantic failures for review instead of silently rewriting the reference or swapping endpoints.
6. Use bounded checks for one-to-one matching, duplicates, empty scenes, missed objects with visible endpoints, missing-point denominators, out-of-frame predictions, deterministic tie breaks, and null statistics. Do not add a full training or hardware CI matrix.

For local files, select a new output directory; refuse accidental overwrite. Preserve raw predictions so scoring can be rerun without another inference pass.

## 4. Fixed evaluation contract

Run one original validation image first and verify numerical predictions plus an overlay. Then evaluate all 20 unchanged validation images. An empty image must still appear in the output.

| Setting | Initial value |
| --- | --- |
| Model | Supplied epoch 173 |
| Image size / rectangular preprocessing | `imgsz=1280`, `rect=True` |
| Inference path | `nms=True`; verify the loaded predictor is not still end-to-end |
| NMS IoU / detection limit | `iou=0.70`, `max_det=300` |
| Precision | FP32, `quantize=32` in Ultralytics 8.4.182 (`half=False` is deprecated) |
| Box confidence | Start at 0.25; then compare 0.15, 0.25 and 0.40 |
| Endpoint confidence | 0.50 throughout the box sweep |
| Object matching IoU | 0.50, separate from the NMS threshold |

These are reproducible diagnostic defaults, not supervisor acceptance tolerances. The completed run used batch 1, CPU, four actual PyTorch threads after predictor setup, and one excluded explicit warm-up. Keep other settings fixed and record the same image order/device. Do not substitute the current draft exporter's differently configured predictions for this evaluation.

For each image, sort predictions by descending box confidence, with original prediction index breaking ties. Match each prediction to the unmatched same-class reference box with the greatest IoU ≥ 0.50; break IoU ties by reference row order. Each object participates in at most one pair. Matched pairs are TP, unmatched reference objects are FN, and unmatched predictions are FP. An unmatched prediction overlapping an already matched reference at IoU ≥ 0.50 is a duplicate FP; other unmatched predictions are other FP, which can include localization errors as well as background detections.

The endpoint denominator is **every reference endpoint marked visible**, including points on missed objects. Count a point as returned only within a matched pair, with confidence ≥ 0.50, finite coordinates and an in-frame position (`0 ≤ x < 1280`, `0 ≤ y < 720`). Preserve left/right correspondence. Missing reference points are excluded from visible-endpoint localization denominators. PR #8 separately reports returned coordinates on reference-unlocatable endpoints, using only those on matched objects as that availability diagnostic's denominator; no hidden coordinate error is computed. Report returned count, missing count and coverage; use `null` if a denominator is zero.

Measure Euclidean endpoint error in original-image pixels for returned, matched, reference-visible points. Label median/p95 as conditional on returned endpoints. Compute quantiles by linear interpolation at sorted index `q × (n − 1)`. Use `null`, never zero, when no errors are available. Keep coverage and miss counts beside those conditional errors. No pixel or millimetre acceptance tolerance has been specified; do not invent one or translate pose mAP into a physical-accuracy claim.

## 5. Deliver and choose one next experiment

Save a compact result bundle containing:

- Raw predictions keyed by image and threshold, including zero-detection images.
- Per-image and aggregate TP/FN/duplicate-FP/other-FP counts; visible-endpoint denominator, returned/missing counts and coverage; conditional median/p95 pixel error.
- Radius success over all visible endpoints, both-corner success on the same fully visible object, and returned coordinates on reference-unlocatable sides. Use PR #8's replay diagnostic on saved predictions. Until the application tolerance is confirmed, 2/5/10 px are illustrative, not pass/fail requirements.
- A small selection of failure overlays and their source image names. Explain misses, duplicate detections and wrong-edge points against originals.
- Code revision; model SHA-256; image/label hashes and split; runtime/device; inference and matching settings; commands, exit status and output locations.
- Timing definitions, warm-up exclusion and sample count. Separate preprocessing, prediction and postprocessing where available. Offline timing is not live camera/display FPS.
- Limits: what ran, what did not, and whether results are newly reproduced or merely quoted from the upstream model card.

**Done for the first milestone:** all 20 reference images are accounted for, the three confidence settings are comparable, a fresh run can reproduce the procedure, and the report recommends one bounded action from observed errors. A blocked runtime is not a completed baseline: finish the runnable implementation and focused checks, record the exact blocker, and provide a concrete continuation command for a compatible environment.

Fine-tuning is conditional on verified training-label changes or useful new training data. The completed validation review accepted zero label changes. PR #9 separately supplies eleven proposed training-row removals; its five-file overlay needs review and full dataset staging before training. Leave validation images out of training. Compare unchanged epoch 173, an unchanged-data reference fine-tune, and a reviewed-data candidate using the same evaluation. The fuller proposal is in [PR #1](https://github.com/Siuuuers/dl/pull/1), `docs/refine-cassette-pose.md`; read its branch until merged. Retain the baseline when evidence is inconclusive. The 20-image validation set already selected epoch 173; the five excluded final-test images or genuinely new independent scenes are needed for final generalization claims.

## 6. Editor readiness and annotation acceptance

[PR #3](https://github.com/Siuuuers/dl/pull/3), [PR #4](https://github.com/Siuuuers/dl/pull/4) and [PR #5](https://github.com/Siuuuers/dl/pull/5) are merged. Current HTML covers original PNG/pose-TXT loading; add/delete and box/point edits; missing endpoints; selected-row text editing; Apply/Reset; one-step Undo; explicit Save; Save & Next after successful write-close; and hiding annotations for raw-image inspection. No additional frontend architecture is needed for a supervised correction pilot.

Fresh checks on source revision `b4df76935178de5a122119aea48e85d2d661effc`:

- `node tools/cassette_labeler/check.mjs`: exit 0, 18/18 scenarios passed with DOM/canvas/file-handle substitutes.
- `python3 scripts/check_export_pose_drafts.py`: exit 0; fake-model export and existing-output preservation passed.
- A scratch probe loaded the real 54-row `new_000007.txt`, targeted row 37 and round-tripped an unchanged save. It also confirmed that manually typed reversed endpoints and nonzero missing-point coordinates are accepted by the HTML. Require the semantic preflight above before accepting labels for training; a successful Save alone does not establish correctness.

Native browser acceptance remains unverified in this execution environment. The browser could not reach the local test server and rejected `file:` URLs under its security policy. These access limits do not establish an HTML defect. Do not bypass that policy or repeatedly redesign the editor to address it.

Before scaling a browser annotation session, use an accessible desktop browser and a disposable working folder with actual PNG bytes. Open a real original plus copied labels; target a middle row, edit and Apply, hide/show overlays, Save, reopen, and compare the saved text. Check Save & Next on a second image and one invalid Apply. Use copied PNGs if absolute image symlinks from the exporter do not work in that environment; symlink compatibility has not been verified here. Keep original reference labels unchanged.

The current exporter refuses an existing `data_pred/`, exports validation only, and does not share all evaluation inference settings. Its drafts support review, not scoring or new training examples. The candidate files reported in [PR #2](https://github.com/Siuuuers/dl/pull/2), `docs/handoff-cassette-pose-ste100.md`, remain to be recovered; the reported 3,765 candidate objects and 17 pending validation images are historical claims, not newly verified completion.

## Next-conversation instruction

Read this workflow, PR #7's baseline report, PR #8's quality decisions and PR #9's training pilot; inspect current `Siuuuers/dl` and open PRs before editing. Preserve the confirmed baomai two-corner task and frozen reference. The 12-row validation review, extra `008` localization example and 13-image training pilot are complete. Reuse their recorded findings instead of restarting them. Review the exact eleven-row proposal, then build and verify a separate complete candidate dataset for PR #1's controlled comparison. Run `python3 scripts/verify_pose_candidate.py` on the candidate branch; it verifies labels and metadata, not image bytes or physical correctness.

The first-use goal and pixel or calibrated physical tolerance remain unconfirmed. Treat supervised investigation as a working scope, not user-confirmed deployment approval. Keep proposed box rules separate from adopted policy. Historical upstream instructions support visible-front boxes in the earlier epoch-127 workflow, but inheritance by the current epoch-173 data is not confirmed. Preserve `0 0 0` for unlocatable points, and do not alter a supported label to match a confident prediction. The new 3,761-object proposal is distinct from PR #2's unrecovered historical 3,765-object candidate. Retain the baseline unless the controlled experiment supports an improvement. Do not silently weaken IoU or choose a tolerance just to make a score pass.

Reuse the implemented evaluator and quality replay. The current evaluator is pinned to the frozen reference; evaluating future candidate checkpoint bytes requires explicit provenance support and validation, not overwriting the reference model card or hashes. Follow PR #1's controlled comparison only when candidate data is ready. Recover PR #2's earlier files independently if their actual location becomes available. Native-browser Save/reopen acceptance remains pending; use an accessible desktop browser and disposable copies before scaling a browser correction session.
