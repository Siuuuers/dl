# Execute the baomai baseline evaluation

Start here in the next conversation. The user confirmed that the supervisor requires the upstream `baomai` task and that `Siuuuers/dl` is the working repository. There is enough information to implement and run the first offline baseline; do not repeat the architecture interview.

**Next task:** implement a small reproducible evaluator, run unchanged epoch 173 on the 20 reference validation images, and inspect its errors. This document is an execution handoff, not a completed model experiment.

## 1. Preserve the task and reference

- Inspected workspace: `b4df76935178de5a122119aea48e85d2d661effc`.
- Upstream: [`an94-doll/embedding_vision`, `baomai` at `351eb3cf960ccc34829e2bc72e1fa4645acc6295`](https://github.com/an94-doll/embedding_vision/tree/351eb3cf960ccc34829e2bc72e1fa4645acc6295).
- Model: `weights/cassette_lower_epoch173.pt`, YOLO26s-Pose. SHA-256: `e0aab1b377491700318e0b14cd4c558ec3c1684087a13b7dde2298ebdbf2847e`; recorded size: 106,838,542 bytes.
- One `cassette_front` class; `kpt_shape: [2, 3]`; `flip_idx: [1, 0]`. Predict the actual lower corners of the white front face, ordered by image X. Upper-lip, tray and basket corners are different features.
- Use original 1280×720 PNGs. A label row is `0 cx cy width height left_x left_y left_v right_x right_y right_v`. Coordinates are normalized. Visible points use `v=2`; an unlocatable point uses `0 0 0`.
- The reference has 137 train / 20 validation images, 3,772 / 664 instances, and 13 / 3 empty scenes. `data/annotated/` contains 1280×750 previews with a footer; never use those as model input.

Read [data/README.md](../data/README.md), [data/data.yaml](../data/data.yaml), [model_card.json](../metadata/model_card.json), [training_config.json](../metadata/training_config.json), [weights/base.json](../weights/base.json), and the predictor setup in [view_d405_yolo.py](../scripts/view_d405_yolo.py). Preserve the supplied labels, split and checkpoint. Work on a new branch from the then-current main; inspect existing work before adding a second implementation.

At the inspected revision, the committed `data/`, `metadata/`, `models/` and `weights/` trees match upstream. This compared Git blobs, including LFS pointers; downloaded image and checkpoint bytes still require verification at execution time.

## 2. Bounded assignments

| Task / owner | Scope | Depends on | Acceptance evidence |
| --- | --- | --- | --- |
| Baseline / executing assistant | Preflight, offline evaluator, one-image smoke run, all 20 validation images, confidence comparison | Reference files and usable inference runtime | Saved predictions, per-image/aggregate metrics, failure overlays, run record |
| Label recovery / independent assistant if useful | Locate the earlier candidate labels, inventory and compare small text files | Candidate folder, archive or earlier session artifact if available | Actual files, hashes, split, changed rows and review status; otherwise an explicit unresolved location |
| Integration / one coordinator | Own evaluator, shared configuration and result interpretation | Baseline results | One reviewed PR and an exact-revision evidence summary |

Label recovery can run in parallel. Its unresolved location does not block evaluation against the committed reference. Do not train, change the model architecture, expand the target to five points or 3D, or add more editor features as part of this first milestone.

## 3. Preflight and implementation

1. Record the checked-out commit. Recheck the model card, current scripts, and open PRs; this handoff may outlive the inspected revision.
2. Obtain only the needed model and validation original PNGs first. In a fresh clone, skip automatic LFS smudging and fetch the selected LFS paths. Do not download training images or annotated previews for this milestone. Detect pointer text and verify actual bytes against the model card and validation records in `data/manifest.json`.
3. Use an isolated headless inference environment with **Ultralytics 8.4.182**. Record Python, PyTorch, CUDA, device and precision. Reuse a compatible runtime where available; do not silently upgrade dependencies. Offline evaluation does not require a RealSense device or the live viewer's desktop/CUDA setup. Select CPU or an available GPU explicitly and use the same device throughout a comparison.
4. Add a small CLI, proposed path `scripts/evaluate_pose.py`, if no suitable evaluator exists. This file and its CLI do **not** exist in the inspected revision. Separate deterministic scoring from model inference so matching and missing-point accounting can be checked without loading weights.
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
| Precision | FP32, `half=False` |
| Box confidence | Start at 0.25; then compare 0.15, 0.25 and 0.40 |
| Endpoint confidence | 0.50 throughout the box sweep |
| Object matching IoU | 0.50, separate from the NMS threshold |

These are reproducible diagnostic defaults, not supervisor acceptance tolerances. Keep other settings fixed and record the same image order/device. Do not substitute the current draft exporter's differently configured predictions for this evaluation.

For each image, sort predictions by descending box confidence, with original prediction index breaking ties. Match each prediction to the unmatched same-class reference box with the greatest IoU ≥ 0.50; break IoU ties by reference row order. Each object participates in at most one pair. Matched pairs are TP, unmatched reference objects are FN, and unmatched predictions are FP. An unmatched prediction overlapping an already matched reference at IoU ≥ 0.50 is a duplicate FP; other unmatched predictions are other FP, which can include localization errors as well as background detections.

The endpoint denominator is **every reference endpoint marked visible**, including points on missed objects. Count a point as returned only within a matched pair, with confidence ≥ 0.50, finite coordinates and an in-frame position (`0 ≤ x < 1280`, `0 ≤ y < 720`). Preserve left/right correspondence. Missing reference points are excluded from endpoint denominators. Report returned count, missing count and coverage; use `null` if a denominator is zero.

Measure Euclidean endpoint error in original-image pixels for returned, matched, reference-visible points. Label median/p95 as conditional on returned endpoints. Compute quantiles by linear interpolation at sorted index `q × (n − 1)`. Use `null`, never zero, when no errors are available. Keep coverage and miss counts beside those conditional errors. No pixel or millimetre acceptance tolerance has been specified; do not invent one or translate pose mAP into a physical-accuracy claim.

## 5. Deliver and choose one next experiment

Save a compact result bundle containing:

- Raw predictions keyed by image and threshold, including zero-detection images.
- Per-image and aggregate TP/FN/duplicate-FP/other-FP counts; visible-endpoint denominator, returned/missing counts and coverage; conditional median/p95 pixel error.
- A small selection of failure overlays and their source image names. Explain misses, duplicate detections and wrong-edge points against originals.
- Code revision; model SHA-256; image/label hashes and split; runtime/device; inference and matching settings; commands, exit status and output locations.
- Timing definitions, warm-up exclusion and sample count. Separate preprocessing, prediction and postprocessing where available. Offline timing is not live camera/display FPS.
- Limits: what ran, what did not, and whether results are newly reproduced or merely quoted from the upstream model card.

**Done for the first milestone:** all 20 reference images are accounted for, the three confidence settings are comparable, a fresh run can reproduce the procedure, and the report recommends one bounded action from observed errors. A blocked runtime is not a completed baseline: finish the runnable implementation and focused checks, record the exact blocker, and provide a concrete continuation command for a compatible environment.

Fine-tuning is conditional on verified training-label changes or useful new training data. Compare unchanged epoch 173, an unchanged-data reference fine-tune, and a reviewed-data candidate using the same evaluation. The fuller proposal is in [PR #1](https://github.com/Siuuuers/dl/pull/1), `docs/refine-cassette-pose.md`; read its branch until merged. Retain the baseline when evidence is inconclusive. The 20-image validation set already selected epoch 173; the five excluded final-test images or genuinely new independent scenes are needed for final generalization claims.

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

Read this workflow from its PR branch, inspect current `Siuuuers/dl`, and execute the first offline baseline milestone. Preserve the confirmed `baomai` task and reference assets. Implement the evaluator and bounded scoring checks, then run real inference if the model files and runtime are available. Recover earlier candidate labels independently. Report the tested revision, results, reproducibility evidence and concrete limits; do not present a proposed experiment or editor test as model performance.
