# Predict cassette objects in a folder of images

Use `scripts/predict_pose_folder.py` to run the unchanged epoch173 model on a folder of original still images. It writes annotated PNGs, structured predictions and an image summary. No annotation TXT files, camera, desktop display or training run are needed.

The agreed scope is one `cassette_front` class and its two lower-front corner slots. Repository files are the available source material. Missed and extra detections have no prescribed relative priority, so the existing operating point remains box confidence **0.25** and endpoint display confidence **0.50**. These settings are not a claim of an optimal error tradeoff.

## Run

The current implementation, including the optional NMS setting, is on `Siuuuers/dl` branch `codex/baomai-postprocess-review-20261107`. Use this branch while its PR is open; the older evaluator checkout does not contain this runner. For a fresh Linux checkout with Git and Git LFS installed:

```bash
GIT_LFS_SKIP_SMUDGE=1 git clone --depth 1 --single-branch \
  --branch codex/baomai-postprocess-review-20261107 \
  https://github.com/Siuuuers/dl.git dl-folder
cd dl-folder
git lfs install --local
git lfs pull --include='weights/cassette_lower_epoch173.pt' --exclude=''
```

This downloads the original model once without fetching the image dataset. Create `.venv` using only [the isolated-runtime setup](evaluate-pose.md#2-use-an-isolated-runtime), keeping the folder-inference checkout above. The runner requires Ultralytics `8.4.182`; the verified CPU environment uses PyTorch `2.10.0+cpu`, torchvision `0.25.0+cpu`, NumPy `2.5.4`, Pillow `13.0.0` and OpenCV `5.0.0.93`. The live-camera environment's RealSense and display setup is unnecessary for this command.

```bash
.venv/bin/python scripts/predict_pose_folder.py \
  --input /path/to/original-images \
  --output /path/to/new-results \
  --device cpu
```

In an existing compatible CUDA environment, use that environment's Python in place of `.venv/bin/python` and choose `--device 0`. Choose a new output directory for each run. The input and output directories must not contain each other. Existing outputs are refused.

The command recursively selects `.png`, `.jpg` and `.jpeg` files, including uppercase extensions, in a stable relative-path order. Use original camera images; `data/annotated/` contains reference previews with a footer and is not the image input. Labels and other file types are ignored. Unreadable images are reported explicitly. Animated/multiple-frame files and non-8-bit image modes are rejected rather than silently converted into an unintended still image.

### Optional overlap suppression

The default NMS IoU remains **0.70**. A bounded [existing-data comparison](postprocess-experiment-2026-11-07.md) found that **0.68 at box confidence 0.25** reduces extra detections from 63 to 61 on the 20 development images, preserving all 625 matched objects and their exact predictions. This is a small settings result on familiar images, not a new trained model or an independent accuracy guarantee.

To try that candidate, add `--conf 0.25 --nms-iou 0.68` to the command above. The actual IoU is recorded in `run_status.json` under `inference.iou`. A lower value suppresses overlapping predictions more strongly; the tested 0.60 and 0.50 values also remove valid reference matches, so lower is not automatically better. NMS IoU is separate from the evaluator's fixed matching IoU of 0.50.

## Results

| Output | Contents |
| --- | --- |
| `overlays/` | One annotated PNG for each successfully processed image, retaining its subdirectory and complete original filename before `.prediction.png`. |
| `predictions.jsonl` | One record for every selected image, including zero-detection images and image errors. Successful records retain raw postprocessed boxes, corner coordinates, confidences and display flags. |
| `summary.csv` | Per-image status and predicted-object counts for quick inspection. |
| `run_status.json` | Completion state, selected/successful/failed counts, original model identity, input/settings information and the actual runtime. |

For example, `tray/frame.jpg` and `tray/frame.png` produce `overlays/tray/frame.jpg.prediction.png` and `overlays/tray/frame.png.prediction.png`.

The overlay uses cyan boxes, green left-slot points, pink right-slot points, and an orange line when both points pass the display checks. Small numbers identify each prediction's one-based position in that image's JSON list; they are not tracking IDs across images. Confidence values remain in JSON so dense scenes stay readable.

Require both process exit **0** and `run_status.json` with `status: complete` before treating a run as finished. A failed or interrupted run is not a complete result; create a new output directory after correcting its cause. A zero-object prediction is a successful processed image, not a read error or proof that the scene contains no cassette.

Coordinates are in the displayed image's pixels. EXIF orientation is applied once before inference and rendering; the record preserves the source dimensions/orientation and the decoded dimensions. There is no resize of the saved overlay. Framework preprocessing is reversed by the existing prediction implementation, so output coordinates are not scaled again.

The two keypoint slots remain in model order. Their confidence and coordinates are preserved even when a point is not drawn. Display requires confidence at least `--endpoint-conf` and finite coordinates inside `0 <= x < width`, `0 <= y < height`. The point flags describe these checks. They do not certify physical visibility, occlusion, annotation correctness or a pixel-error tolerance. No hidden point is extrapolated and no prediction is exported as an approved `v=2` label.

## What this result establishes

This is an inference deliverable. Unlabeled images cannot supply false-negative, false-positive, precision, recall or accuracy measurements. `ground_truth_available: false` distinguishes predictions from reference annotations. Keep the existing [evaluation command](evaluate-pose.md) for the frozen labeled development set.

The [additional-frame candidate](additional-annotation-pilot-2026-11-07.md) still has three quarantined images and zero eligible scoring/training images. Drawing predictions on an image does not resolve its reference boundaries. The unchanged epoch173 default and historical model comparison remain recorded in [the warmup report](warmup-comparison-2026-11-07.md).

## Verified example, 2026-11-07

Clean source revision [`df9254ac16f61eb6f5f858dfd4737459d4133a7e`](https://github.com/Siuuuers/dl/commit/df9254ac16f61eb6f5f858dfd4737459d4133a7e) passed all **11 focused checks** and a real CPU run on two hash-verified repository originals:

| Original image | Folder-runner predictions | Comparison with the existing evaluator |
| --- | ---: | --- |
| `new_000007.png`, dense scene | 53 | Raw boxes, keypoints and confidences exactly equal |
| `new_000146.png`, reference-empty scene | 0 | Raw predictions exactly equal |

Both runs completed with exit 0 at the same clean revision, using four CPU threads, ordinary NMS and FP32. Both 1280×720 overlays were inspected. The dense overlay uses compact numbers; the empty-image output preserves the image without invented annotations. Prediction counts alone are not missed-object or false-positive counts.

![Epoch173 folder-runner prediction on the original dense cassette scene](images/baomai-folder-dense-2026-11-07.png)

The [19-file demo ZIP](../results/prediction/baomai-folder-20261107/demo.zip) contains both original inputs, both rendered outputs, predictions, CSV, runtime/status records, comparison output and command logs. Fresh extraction passed every payload hash and the archive CRC check. The [verification record](../results/prediction/baomai-folder-20261107/verification.json) preserves that original runner revision, source hashes, commands, runtime and exact archive identity.

The optional-NMS extension was subsequently tested at clean revision `2c409f4fa7c16367b7895a8b021611a277c68a3c`: **14 focused folder checks** pass. A real 0.68 run on `new_000007`, `new_000008`, `new_000009` and `new_000090` exactly reproduces all four corresponding raw prediction lists from the fixed experiment. A separate default run on the original dense/empty pair still produces exactly 53/0 predictions. Both commands complete with exit 0. The [new evidence record](../results/evaluation/baomai-postprocess-20261107/verification.json) identifies these distinct tests; the older demo remains evidence for its original revision.

## Implementation and focused verification

The runner reuses `evaluate_pose.INFERENCE`, `_runtime_and_model` and `_prediction_records`, including the existing actual model-head check. Its model-only preflight verifies the original checkpoint's SHA-256, byte size and two-corner identity. The frozen evaluator's 20-image guard and scorer are unchanged.

It decodes each selected file explicitly to avoid a framework folder loader silently skipping an unreadable image. The same oriented RGB pixels are rendered and converted to BGR for inference. One blank warmup precedes predictions; the actual backend is checked for ordinary NMS and FP32, then the requested CPU thread count is restored.

Run the focused checks in an environment with Pillow and NumPy:

```bash
.venv/bin/python scripts/check_predict_pose_folder.py
```

The tests use temporary images and a substituted prediction adapter. A separately recorded real-model check uses a dense original and a reference-empty original, inspects both overlays, and compares their raw predictions with the existing evaluator in the same runtime. Neither check establishes independent-test accuracy.

Implementation research used the pinned official [pose-coordinate conversion](https://github.com/ultralytics/ultralytics/blob/v8.4.182/ultralytics/models/yolo/pose/predict.py), [image loaders](https://github.com/ultralytics/ultralytics/blob/v8.4.182/ultralytics/data/loaders.py) and [coordinate clipping](https://github.com/ultralytics/ultralytics/blob/v8.4.182/ultralytics/utils/ops.py), alongside the repository's existing evaluator.
