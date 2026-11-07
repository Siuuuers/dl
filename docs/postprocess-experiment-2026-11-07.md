# Cassette postprocessing experiment without new training data

## Plan recorded before fresh candidate inference

**Goal:** test whether stronger non-maximum suppression (NMS) can reduce duplicate cassette detections while preserving object recall and the established two-corner quality measures. This is an inference-settings experiment on the frozen development reference, not a new trained model or an independent test.

The current M0/epoch173 result at box confidence 0.25 and NMS IoU 0.70 has 39 missed reference objects, 63 extra detections and 17 duplicate false positives across 664 objects in 20 images. The known duplicate category provides a concrete reason to test NMS before allocating another training run. Source: [the recorded warmup comparison](warmup-comparison-2026-11-07.json), M0 metrics.

## Fixed experiment

- Checkpoint: the original epoch173 bytes, SHA-256 `e0aab1b377491700318e0b14cd4c558ec3c1684087a13b7dde2298ebdbf2847e`.
- Reference: all 20 original validation images and frozen labels, including the three empty scenes. Existing manifest/hash and checkpoint checks apply.
- Box confidence: 0.15, 0.25 and 0.40.
- NMS IoU: 0.70, 0.60, 0.68 and 0.50; four values by three confidences, totaling 240 fresh image/setting predictions.
- Other inference settings: image size 1280, rectangular preprocessing, FP32, ordinary NMS, batch 1, `max_det=300`, one common CPU runtime and four threads.
- Scoring: unchanged confidence-ordered matcher, matching IoU 0.50, endpoint confidence 0.50. NMS IoU and matching IoU serve different purposes; the matching rule stays fixed.
- Retain full predictions and every image's scores, including all missed objects and empty scenes. Preserve the runtime, exact source files, source revision, input hashes and successful completion status.

The primary comparison is **NMS 0.60 / confidence 0.25 versus NMS 0.70 / confidence 0.25**. The NMS 0.68 and 0.50 settings at confidence 0.25 are separately reported secondary candidates. Confidence 0.15 and 0.40 are sensitivity checks, not replacements for the primary comparison.

The conservative 0.68 secondary was added during pre-inference review because an existing true-positive box in `new_000009` overlaps an earlier false-positive box by approximately 0.658 after clipping. Stronger NMS might suppress that correct match. This review uses existing outputs to formulate the hypothesis; it does not claim unseen-data selection. There will be no adaptive grid expansion after reading the fresh results.

## Acceptance and interpretation

For a setting to pass the existing five-measure no-regression guard, all of the following unrounded values must be finite and no larger than the current 0.70/0.25 control, with at least one strictly smaller:

1. Missed objects (FN).
2. Extra detections (FP).
3. Missing reference-visible endpoints.
4. Conditional median endpoint error in pixels.
5. Conditional p95 endpoint error in pixels.

Report precision, recall, F1 and FN+FP as descriptive object-detection summaries. The user has not prioritized missed over extra detections, but an equal-cost FN+FP summary does not retrospectively replace the existing five-measure guard. The 2/5/10-pixel corner radii remain illustrative diagnostics rather than user-approved tolerances. Per-image regressions and every lost/gained reference match must remain visible even if an aggregate guard passes.

Verify that fresh NMS-0.70 controls reproduce the previously saved raw predictions at all three confidences. This checks common-runtime consistency before interpreting a candidate difference. Replay saved experimental predictions through the same unchanged scorer and compare the complete numerical reports. Inspect relevant original-image regions for candidates that pass the numerical guard.

No default model or inference setting changes automatically. A same-data result can support a bounded candidate setting, but cannot establish its performance on future camera scenes. Unresolved supplementary annotations remain quarantined.

## Implementation rationale

Ultralytics documents that a lower NMS IoU suppresses more overlapping boxes and can reduce duplicates. Suppression can also remove distinct overlapping objects. The pinned framework performs NMS before the pose predictor scales/clips coordinates into the original image. Consequently, applying another NMS pass to already saved, clipped boxes is not an equivalent implementation of this experiment. Each cell must run through fresh framework inference.

Primary references:

- [Ultralytics prediction arguments](https://docs.ultralytics.com/modes/predict/#inference-arguments).
- [Pinned v8.4.182 NMS implementation](https://github.com/ultralytics/ultralytics/blob/v8.4.182/ultralytics/utils/nms.py).
- [Pinned v8.4.182 pose prediction conversion](https://github.com/ultralytics/ultralytics/blob/v8.4.182/ultralytics/models/yolo/pose/predict.py).
- [Existing model-quality definitions and limitations](model-quality.md).

## Execution and result

Pending the fixed experiment above. The completed report will preserve both the pre-inference source revision and the actual result, including a failed primary comparison if one occurs.
