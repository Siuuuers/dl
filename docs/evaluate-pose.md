# Evaluate the baomai baseline and completed training arms

`scripts/evaluate_pose.py` defaults to the unchanged epoch-173 model against all 20 reference validation images. An explicit `--checkpoint-manifest` selects a verified completed training arm while keeping the reference validation set fixed. It needs no RealSense device or desktop display. The task remains one `cassette_front` class and two actual lower-front corners, ordered by image X. Continue from [workflow PR #6](https://github.com/Siuuuers/dl/pull/6).

**Latest model comparison:** the [completed warmup comparison](warmup-comparison-2026-11-07.md) evaluates the audited [PR #16](https://github.com/Siuuuers/dl/pull/16) return alongside M0 and a fresh control. Warmup improves all five primary measures against M0, but fails the required control comparison on missed objects: 37 versus 36. Keep epoch173 as the default and retain warmup as a promising candidate. Use that report's recorded revision and commands to reproduce its comparison.

The [follow-up detection review](warmup-detection-review-2026-11-07.md) completes all nine lost and eight gained cases and records 320 recovered additional frames with labels. Their raw hashes do not overlap the frozen 157 images. A three-image pilot without model predictions found that `frame_0160` rows 41–43 extend into the metal rack/base, violating the visible-front box policy. This unscored pool is not ready as supplied and is not an independent test: prepare a separately versioned annotation adaptation, preserve source labels and record capture/session and prior-use limits before freezing and scoring a supplementary reference. The commands below retain the frozen development reference.

Historical milestones remain in [the 2026-11-06 baseline report](baseline-2026-11-06.md) and [the first GPU comparison](gpu-comparison-2026-11-07.md). The latter evaluated the original-label and cleaned-label fine-tunes, both of which failed the replacement rule. Its replay evidence and comparison summarizer preserve that earlier result.

Do not use these images as an independent final test: they already participated in selecting epoch 173. Do not change reference labels to improve this score. The user clarified that the current duty is finding/localizing objects in the supplied baomai D405 images. Keep the two-corner target; camera/robot calibration is outside this evaluation. The [warmup handoff](gpu-warmup-handoff.md) now preserves reproduction instructions. Its notebook setup syntax repair does not require rerunning the completed GPU pair. Earlier candidate-label recovery and a supervised editor acceptance session are separate work.

## 1. Obtain the reference bytes

Run from a checkout containing this evaluator. For a fresh clone, skip automatic LFS downloads, then fetch only the checkpoint and validation originals:

```bash
GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/Siuuuers/dl.git
cd dl
# Pin the published evaluator with schema-2 support and both comparison summaries:
GIT_LFS_SKIP_SMUDGE=1 git checkout --detach 2f0260c19835fcd60eb2f12040b7c77ebd2ca890
git lfs install --local
git lfs pull --include='data/images/val/*.png,weights/cassette_lower_epoch173.pt' --exclude=''
python3 scripts/evaluate_pose.py --preflight-only
```

The default preflight requires all 20 originals and labels, including three genuinely empty scenes. It checks SHA-256 values, checkpoint size, PNG dimensions, unique image/label pairs, manifest counts, and label semantics before loading the model. A Git LFS pointer is an explicit error. Training images and annotated preview PNGs are not required. The verified reference contains 664 objects, 1,227 visible endpoints and 101 unlocatable endpoints.

The semantics check requires 11 finite fields, class 0, positive normalized box dimensions, coordinates in [0, 1], visibility 0 or 2, `0 0 0` for a missing point, and nondecreasing left/right X when both are visible. It does not silently repair labels. `--image ID` restricts required image/label bytes to selected validation records for a smoke run; omitting it evaluates the entire validation set.

## 2. Use an isolated runtime

The checked CPU environment uses Python 3.12.14, PyTorch 2.10.0+cpu, torchvision 0.25.0+cpu and Ultralytics 8.4.182. Keep a compatible existing environment if available. An example fresh CPU installation is:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.10.0 torchvision==0.25.0
.venv/bin/python -m pip install ultralytics==8.4.182 numpy==2.5.4 Pillow==13.0.0 opencv-python==5.0.0.93
```

These commands are for offline evaluation. The live D405 setup in the main README has additional CUDA, camera and display requirements. The evaluator explicitly checks the Ultralytics version. It records the installed versions, requested and actual device, and actual CPU thread count. When using a compatible CUDA environment, choose `--device 0` explicitly and use that device for the whole comparison. GPU timings are not established by a CPU run.

## 3. Check scoring, then run real inference

The focused checks use only the Python standard library, temporary fixtures and fake checkpoint bytes; they never load a model:

```bash
python3 scripts/check_evaluate_pose.py
```

Run one real image and inspect its overlay before the complete sweep:

```bash
.venv/bin/python scripts/evaluate_pose.py --device cpu --threads 4 --image new_000007 --conf 0.25 --overlays 1 --output reports/smoke-001
```

Then evaluate all 20 images at the three fixed box-confidence thresholds:

```bash
mkdir -p reports
.venv/bin/python scripts/evaluate_pose.py --device cpu --threads 4 --conf 0.15 0.25 0.40 --output reports/baseline-001 > reports/baseline-001.log 2>&1
.venv/bin/python -m pip freeze > reports/baseline-001/runtime-freeze.txt
```

Choose a fresh output path for every run. An existing directory is refused, including one from a failed run. Do not interpret a partial `predictions.jsonl` as completion: require process exit 0 and `run_status.json` with `status: complete` and 60 scored image/threshold pairs for the default sweep. Inspect the log for runtime warnings, including NMS time limits.

The predictor performs an explicit blank-image warm-up before measurement. It verifies ordinary NMS (`backend_end2end: false`) and FP32 (`backend_fp16: false`). Ultralytics resets CPU threads while selecting the device, so the evaluator restores and records `--threads` after setup and before measured predictions.

## 4. Fixed settings and metric definitions

| Setting | Value |
| --- | --- |
| Model | Original epoch-173 checkpoint; hash checked against the model card |
| Original inputs | 1280 × 720 PNGs; no annotated-preview footer |
| Resize / padding | `imgsz=1280`, `rect=True` |
| Batch | 1; one file per prediction call |
| Ordinary box NMS | `nms=True`, `iou=0.70`, `max_det=300` |
| Precision | FP32 via `quantize=32` in pinned Ultralytics 8.4.182 |
| Box-confidence comparison | 0.15, 0.25, 0.40 |
| Endpoint-confidence cutoff | 0.50 |
| Evaluation matching IoU | 0.50; distinct from the NMS setting |

The upstream version uses `quantize=32` for explicit FP32; `half=False` is its deprecated spelling. Prediction coordinates from `Results.boxes.xyxy` and `Results.keypoints.data` are already in original-image pixels. Labels are denormalized once; predictions are not denormalized again. Returned detections retain Results order. The pinned ordinary NMS path uses a strict box-confidence comparison, and replay validates that saved detections exceed their claimed threshold.

**Object matching:** sort predictions by descending confidence, breaking ties by original prediction index. Match each to the unmatched same-class reference box with greatest IoU at least 0.50; break equal-IoU ties by reference row order. This is deterministic greedy matching. Each pair is a TP, every unmatched reference is an FN, and every unmatched prediction is an FP. An FP overlapping any finally matched reference at the matching threshold is a duplicate FP. Remaining FPs can be background detections or localization errors; the name alone does not distinguish them.

**Endpoint coverage:** the denominator includes every reference-visible endpoint, including those on missed objects. A point is returned only when its object is matched, its corresponding left/right prediction has confidence at least 0.50, and its finite coordinates satisfy `0 <= x < width`, `0 <= y < height`. Unlocatable reference points are excluded. Left and right are never swapped to reduce error. Report returned and missing counts alongside coverage.

**Conditional pixel errors:** calculate Euclidean distance for returned, matched, reference-visible points. Pool all distances before computing median and p95 by linear interpolation at sorted position `q * (n - 1)`. These statistics do not include missing endpoints and must be read with coverage and FN counts. No eligible points means `null`, not zero. No pixel or physical acceptance tolerance has been supplied; these diagnostics do not establish millimetre accuracy.

**Coordinate limits:** saved predictions are the framework's postprocessed and clipped outputs, not network logits or pre-clipping coordinates. In this version coordinates may be clipped to the inclusive frame boundary; the evaluator's strict in-frame rule rejects x=width or y=height. It cannot reconstruct how far an unclipped point originally lay outside the image.

**Timing:** `stage_ms` contains Ultralytics preprocessing, model inference and postprocessing times per image. These exclude image-file decoding, result serialization, overlays and model setup. `predict_call_wall_ms` covers the full measured prediction call, including file input and result construction, but excludes subsequent JSON writing and scoring. Warm-up is excluded. Report means with sample count and device. They are neither a live D405 pipeline measurement nor a camera/display FPS guarantee.

## 5. Saved evidence and replay

| Output | Purpose |
| --- | --- |
| `predictions.jsonl` | Run header plus every image/threshold, including zero-detection scenes; original-pixel postprocessed predictions |
| `summary.json` | Aggregate counts, endpoint statistics, timing definitions, runtime, code revision/file hashes and reference hashes |
| `per_image.json` | Per-image metrics, matching indices, unmatched indices and endpoint errors |
| `per_image.csv` | Compact per-image table |
| `overlays/` | By default, the same five selected images at each threshold; green reference, magenta prediction, red missed object, orange extra prediction |
| `run_status.json` | Completion or explicit failure, exit code and scored-pair count |

Overlay selection ranks the 0.25 results by FN+FP, missing endpoints, then conditional p95, with image ID breaking ties. They are diagnostic examples, not a random sample. Indices in JSON are zero-based; `g` and `p` overlay labels use one-based row indices. Retain the original images and inspect them without annotations when classifying failures.

Replay requires the reference originals and labels but does not deserialize the checkpoint or import PyTorch/Ultralytics. With `--overlays 0`, it needs only the standard library:

```bash
python3 scripts/evaluate_pose.py --replay reports/baseline-001/predictions.jsonl --overlays 0 --output reports/replay-001
```

Replays check dataset/model identity, image/label hashes, dimensions, inference configuration, actual threshold availability and one record for each selected image/threshold. They preserve original inference provenance and timings, including when replaying a replay; the new source revision records the scoring code. Optional `--conf` selects already inferred thresholds. It cannot invent a new threshold run. Changing `--endpoint-conf` or `--matching-iou` changes the scoring criteria and must be stated in comparisons. Checksums establish file identity; a replay is not another model inference.

## 6. Evaluate a completed training arm explicitly

The [training runner](train-pose-comparison.md) produces `checkpoint-manifest.json` only after a complete paired fine-tune has passed its provenance checks. A resource trial does not produce an eligible manifest. The focused checks below use visibly synthetic receipts, results and fake checkpoint bytes; they do not demonstrate a trained candidate or its accuracy:

```bash
python3 scripts/check_pose_checkpoint.py
```

To evaluate a completed run, use its generated bundle:

```bash
.venv/bin/python scripts/evaluate_pose.py --checkpoint-manifest ../baomai-reference-001/checkpoint-manifest.json --preflight-only
.venv/bin/python scripts/evaluate_pose.py --checkpoint-manifest ../baomai-reference-001/checkpoint-manifest.json --device cpu --threads 4 --conf 0.15 0.25 0.40 --output reports/reference-finetune-001
```

The same command pattern applies to the candidate arm and to the warmup experiment's `control/` and `warmup/` bundles. For the completed warmup result, use the report's recorded revision and role-specific manifests. Use matching evaluation settings and a suitable common device. Omitting the manifest continues to select and verify the original epoch173 checkpoint; the frozen model card is never replaced.

The manifest identifies the finalized `last.pt` by content hash and byte size, the pose/class/two-point/flip contract, the training receipt and the copied staging manifest. Its artifact paths are relative and confined to the run bundle. The verifier checks:

- The initialization, frozen dataset/model-card hashes, selected training arm and its dataset YAML identity.
- Both staged file inventories against the frozen reference and the repository's versioned candidate proposal, including all validation image/label hashes.
- The fixed paired recipe, including float `fraction=1.0`. In the pinned runtime integer `1` selects one image, so that substitution is rejected.
- A clean recorded code revision, runner/recipe hashes, runtime identity, observed optimizer updates and validation.
- Successful worker exit without a resource stop or input-verification error, a contiguous completed epoch sequence 1–30, successful finalization and final-epoch selection.
- Hashed results/log files, with exactly 30 epoch rows and finite numeric results, plus the actual checkpoint bytes before inference.

The loaded model's task, class names and actual `[2, 3]` keypoint head are checked before prediction, including when a scene returns no detections. The checkpoint format cannot independently prove the physical meaning of its points; training-data provenance and annotation review establish that meaning.

The evaluator's top-level `model_sha256` identifies the evaluated checkpoint. Its `model_card_sha256` and `manifest_sha256` continue to identify the frozen reference. A trained run also carries `checkpoint_identity`, binding its manifest, receipt, staged data, arm, initialization and final epoch. The quality diagnostic exposes that identity as `evaluated_checkpoint`, separately from the unchanged `reference` object.

Candidate replay requires the same explicit bundle:

```bash
python3 scripts/evaluate_pose.py --checkpoint-manifest ../baomai-reference-001/checkpoint-manifest.json --replay reports/reference-finetune-001/predictions.jsonl --overlays 0 --output reports/reference-replay-001
python3 scripts/diagnose_pose_quality.py --checkpoint-manifest ../baomai-reference-001/checkpoint-manifest.json --predictions reports/reference-finetune-001/predictions.jsonl --output reports/reference-quality-001
```

Replay verifies the bundle's small evidence files and the reference validation bytes without loading or requiring either checkpoint file, training images, PyTorch or Ultralytics. It still needs the versioned reference/candidate text metadata. It never adopts a candidate identity from the prediction header alone, and it preserves original inference identity through repeated replay. Existing schema-version-1 baseline replays remain supported without a checkpoint manifest.

Keep the receipt, copied staging manifest, results and log together with the manifest. Historical absolute training paths remain provenance text; replay does not load data from those paths. Checksums and consistency checks make the experiment traceable, but are not external attestation that a training claim or physical grasp succeeded. A complete bundle qualifies a checkpoint for the planned comparison, not for deployment.

## 7. Continue from measured errors

Continue from the [completed case review and additional-frame intake](warmup-detection-review-2026-11-07.md). The supplied labels require a separately versioned annotation adaptation under the visible-front policy before supplementary scoring. Preserve source labels and the existing lower-front corner target, record capture/session and prior-use limits, then freeze reviewed eligible image IDs and adapted label bytes before scoring the fixed checkpoints. The current evaluator's frozen-reference guard and the existing comparison remain intact. Store supported annotation proposals separately with review status and source hashes. A further training change should follow a specific finding from this evidence.

Fine-tuning requires actual reviewed training changes or useful new training data, as described in [PR #1](https://github.com/Siuuuers/dl/pull/1). Historical candidate counts in [PR #2](https://github.com/Siuuuers/dl/pull/2) do not establish that those files were recovered. A final generalization claim needs independent scenes or the excluded final-test data, not further optimization on these same 20 validation images.

Pinned implementation references: [inference configuration](https://github.com/ultralytics/ultralytics/blob/ee00414e1d86e111ab17fe2aa12a29452eb4b55d/ultralytics/cfg/__init__.py), [predictor](https://github.com/ultralytics/ultralytics/blob/ee00414e1d86e111ab17fe2aa12a29452eb4b55d/ultralytics/engine/predictor.py), [pose coordinate scaling](https://github.com/ultralytics/ultralytics/blob/ee00414e1d86e111ab17fe2aa12a29452eb4b55d/ultralytics/models/yolo/pose/predict.py), [NMS](https://github.com/ultralytics/ultralytics/blob/ee00414e1d86e111ab17fe2aa12a29452eb4b55d/ultralytics/utils/nms.py), and [device/thread selection](https://github.com/ultralytics/ultralytics/blob/ee00414e1d86e111ab17fe2aa12a29452eb4b55d/ultralytics/utils/torch_utils.py).
