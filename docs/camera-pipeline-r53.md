# R53: camera, image processing and prediction filtering

27 November 2026. Source audit of commit
`9983d996e087aa6fd348a1861b61a1c99ec88a0d` (PR147).

**The application can apply an image filter before prediction, but the audited
live viewer currently passes the SDK color image directly to YOLO.** It requests
only the D405 color stream. It does not request depth or call the SDK's depth
post-processing filters. The D405 itself generates RGB through an onboard image
signal processor (ISP), so an overlay-free camera image is not an unprocessed
sensor readout. Actual exposure, gain, white balance and other supported camera
settings were not measured in this audit. [S1, S3, D1]

This is a source inspection, not a connected-camera experiment. No camera was
opened, no model was trained, and no live inference settings were changed. The
[machine-readable audit](../results/review/thesis-r53-20261127/camera-pipeline-audit.json)
records the source hashes and the boundary between observed code and unknown
runtime state.

## What happens to a live frame

| Stage | Audited behavior | What it changes |
| --- | --- | --- |
| Camera output | `D405Capture.start()` requests `rs.stream.color`, `rs.format.bgr8`, 1280 × 720 at 30 FPS by default. The active stream profile is reported after startup. | The requested stream profile; successful delivery and actual rate require a camera run. |
| Frame handoff | `_capture()` obtains `get_color_frame()`, copies its bytes into a NumPy array, and keeps one latest packet with frame number and timestamp. | Buffer ownership and frame selection; no repository contrast, sharpening or denoising step. |
| Model input | `main()` passes `packet['image']` directly to `model.predict(...)`. | Model/framework preprocessing is still part of prediction. The repository adds no separate RGB enhancement here. |
| Detection selection | The viewer explicitly sets `nms=True`, supplies confidence, NMS IoU, maximum detections and image size, and checks that the backend is not end-to-end. | Which predicted boxes survive; it does not alter camera pixels. |
| Endpoint display | `render()` checks endpoint confidence and image bounds. It draws the lower edge only when both endpoints pass. | The overlay; hiding an endpoint does not delete the detection or establish physical visibility. |
| Overlap diagnostics | With diagnostics enabled, `render()` counts post-NMS box pairs above the diagnostic IoU threshold. | Warning text and counts only; it does not suppress predictions. |
| Snapshot/recording | Snapshots save SDK-color PNG and a separate overlay PNG. Recording supports raw or annotated frames. | Saved representation; annotated pixels contain drawn predictions. |

All these stages are defined in [S1–S3]. The viewer records camera receive FPS,
completed inference FPS and display/update FPS separately. A request for 30 FPS
is not a measured guarantee of 30 completed predictions per second. It can skip
older captured frames while maintaining the image/prediction pairing for each
displayed frame. [S1]

### A deployment setting that must be checked

There are two distinct checkpoint selections in the repository:

| Entry or document | Checkpoint | Box confidence |
| --- | --- | --- |
| `start_live.sh` without overriding arguments | `weights/cassette_lower_epoch173.pt` | Inherited from `weights/base.json`: **0.25** |
| `python scripts/view_d405_yolo.py` without `--weights` | The same epoch173 checkpoint through `resolve_base()` | **0.25** |
| Retained research model in `docs/current-model.md` | **PR35 seed 46**, SHA-256 `420c00f73eeba24e587342400878b7d96371d3e745772113809aa705ec8163b1` | **0.13** |

The packaged epoch173 manifest declares SHA-256
`e0aab1b377491700318e0b14cd4c558ec3c1684087a13b7dde2298ebdbf2847e`.
Both configurations use image size 1280, NMS IoU 0.70 and endpoint threshold
0.50. These are source/manifest facts, not a claim that either checkpoint was
observed running on the user's camera. [S1, S2, S5, S6]

The viewer supports explicit `--weights`, `--conf`, `--iou` and `--imgsz`
arguments. An explicit checkpoint alone does **not** import that experiment's
threshold settings: `resolve_live_weights()` still starts from the inference
settings in `weights/base.json`. To reproduce a particular research result,
choose its verified checkpoint and pass its intended thresholds explicitly.
Check the resulting live report's `weights_sha256`,
`initial_inference_settings`, `current_conf`/`final_conf` and camera profile.
The report defaults to `reports/d405_live_latest.json`; keyboard controls can
change confidence and NMS while the application is running. [S1]

The shell launcher appends the user's arguments after its built-in arguments;
the argument parser uses the last occurrence of a repeated option. Thus
`./start_live.sh --weights ... --conf 0.13 --record-mode raw` overrides the
packaged weight and initial recording mode for that run. This precedence was
checked by calling the parser only, without opening the camera or loading a
checkpoint. The direct Python entry avoids the shell launcher's explicit
packaged-weight argument. [S1, S2]

This audit documents the distinction and leaves the startup defaults intact.
Changing the deployment checkpoint should be an explicit model-selection
decision, separate from a preprocessing experiment.

### Inspect the last live report

Run this from the repository root to display the saved report's model identity,
settings and timestamp. A missing or old report provides no evidence of the
current device state. The report does not record exposure/ISP option values.

```bash
python - <<'PY'
import json
from pathlib import Path

path = Path('reports/d405_live_latest.json')
report = json.loads(path.read_text())
keys = (
    'status', 'started_at', 'updated_at', 'finished_at',
    'weights', 'weights_sha256', 'weight_selection',
    'initial_inference_settings', 'initial_endpoint_conf',
    'current_conf', 'current_iou', 'current_endpoint_conf',
    'final_conf', 'final_iou', 'final_endpoint_conf',
    'camera_requested', 'camera', 'versions',
)
print(json.dumps({key: report[key] for key in keys if key in report}, indent=2))
PY
```

### A future explicit PR35 camera check

First obtain and verify the complete PR35 candidate bundle using the existing
[seed46 instructions](predict-pose-folder.md#use-the-seed46-candidate-explicitly).
When the camera and the existing compatible CUDA environment are available,
the following bounded run validates that bundle, enforces the retained PR35
hash, then passes the intended inference settings explicitly. Use the Python
interpreter from that environment. The sample assumes the documented bundle
location `../seed46-results/candidate/` and ends after 60 processed frames.
It writes a separate report and does not start recording automatically.

```bash
python - <<'PY'
import json
import sys
from pathlib import Path

root = Path.cwd().resolve()
sys.path.insert(0, str(root / 'scripts'))
from predict_pose_folder import verify_trained_checkpoint
from view_d405_yolo import main

selected = verify_trained_checkpoint(
    root, root.parent / 'seed46-results/candidate/checkpoint-manifest.json'
)
expected = '420c00f73eeba24e587342400878b7d96371d3e745772113809aa705ec8163b1'
if selected['model_sha256'] != expected:
    raise RuntimeError('The bundle is not the retained PR35 checkpoint.')
print(json.dumps({key: selected[key] for key in
                  ('weights', 'model_sha256', 'checkpoint_manifest')}, indent=2))
raise SystemExit(main([
    '--weights', selected['weights'],
    '--conf', '0.13', '--iou', '0.70', '--endpoint-conf', '0.50',
    '--imgsz', '1280', '--width', '1280', '--height', '720', '--fps', '30',
    '--display-fps', '30', '--device', '0', '--record-mode', 'raw',
    '--max-frames', '60', '--report', 'reports/pr35_live_r53_check.json',
]))
PY
```

This command is prepared for a future hardware check; it was not executed in
this audit. Add `--serial` with the intended camera's serial if multiple D405
devices are present. Inspect `reports/pr35_live_r53_check.json` afterward using
the read-only snippet above with its path changed. Live CUDA predictions need
their own comparison; sharing a checkpoint and settings does not prove they
are bit-identical to previous CPU results.

## Four meanings of “filter”

### 1. Camera image formation and controls

The manufacturer's D405 product documentation identifies its color technology
as the left imager with ISP. This supports the statement that color image
formation already involves camera processing. It does not identify the exposure
or gain used for any particular saved project photograph. [D1]

The audited capture code records the camera name, serial, firmware, USB type and
actual stream dimensions/rate/format. It does not call `get_option()`,
`set_option()` or `get_supported_options()`. Therefore the code does not tell us
whether auto exposure is currently active or what each available option reads.
Do not assume that every D405 exposes the same controls as a separate RGB
sensor, or that another application's settings are reproduced. [S1]

At the next real capture session, a read-only settings record would resolve
this gap: enumerate sensors, enumerate their supported options, read each
supported value and record unsupported/read-failure states. The official SDK
provides support checks, supported-option enumeration and value/range access.
Record the SDK version alongside firmware and stream profile. This is a
proposed capture-provenance improvement; no hardware values were collected
here. [D3]

### 2. RGB preprocessing before the neural network

The application can call OpenCV between frame acquisition and prediction. For
example, one controlled experiment can convert BGR to CIE Lab, apply CLAHE to
the L channel and convert back to BGR. OpenCV documents both the conversions
and CLAHE's contrast-limit/tile-grid controls. This changes intensities while
retaining the original pixel coordinate grid. [D4, D5]

The hypothesis is that local contrast adjustment might make a faint cassette
edge easier for the existing model to use. It could also amplify noise, alter
the expected appearance or increase false outputs on rack edges. OpenCV's
tutorial specifically discusses noise amplification by local histogram
equalization and contrast limiting. These are reasons to compare predictions,
not evidence that CLAHE improves this dataset. [D4]

An ordinary pose checkpoint does not automatically choose or execute an
arbitrary OpenCV filter. The application must explicitly call the preprocessing
function. Adding a learned filter selector would be a further model and
evaluation problem; it is unnecessary for the first comparison.

### 3. RealSense depth post-processing

The documented SDK pipeline contains decimation, depth/disparity conversion,
spatial smoothing, temporal filtering and hole filling for **depth data**.
Temporal filtering uses previous frames and can produce smearing; it needs a
continuous sequence to assess that behavior. These operations do not establish
that the BGR array supplied to this repository's pose model has been enhanced.
The live viewer does not enable a depth stream or call these processing
blocks. [D2, S1]

Depth may eventually support a different geometric or multimodal study, but
that would require captured depth, calibration, synchronization and a defined
way to use it. The current RGB-only path cannot be treated as such a system.

### 4. Filtering predictions after the neural network

The live viewer's confidence gate and ordinary NMS affect detections. Endpoint
confidence affects drawing. Neither can reconstruct a physical corner absent
from the observed image. [S1]

`scripts/predict_pose_folder.py` has a separate, optional `--postclip-nms`
switch. Its `postclip_nms()` function sorts final boxes by descending confidence
and suppresses same-class boxes when their final clipped IoU is **greater than
0.70**. It preserves original prediction identifiers and suppressed records;
it does not move boxes or average keypoints. The flag defaults to off and is
not connected to the live viewer. A successful historical folder experiment
does not imply this option is active during live use. [S4]

## Saved images and training provenance

The folder predictor decodes PNG/JPEG, applies EXIF orientation, converts to
RGB, then converts to contiguous BGR for prediction. It records source and
decoded dimensions and image SHA-256. It does not recover original camera
controls from the pixels. EXIF orientation handling also means source-file
dimensions and coordinates must not be silently mixed with decoded-image
coordinates. [S4]

The live snapshot function saves `packet['image']` separately from its overlay.
The recorder's `raw_pixels` field means overlay-free input to its encoder;
it does not mean Bayer sensor data or prove a byte-identical decoded video.
The recorder requests `mp4v` in an MP4 container. OpenCV documents that
`VideoWriter` applies the selected codec and distinguishes lossless/raw
recording from typical compressed codecs. Preserve the original PNG and
metadata when exact annotation pixels matter. [S1, S3, D1, D6]

`start_live.sh` starts in **annotated recording mode**, but recording begins
only after the user presses R. The direct Python entry starts in raw recording
mode by default. Annotated recordings are useful demonstrations; their drawn
endpoints and boxes must not become inputs for a clean recognition comparison.
The existence of this current viewer does not prove the older training images
were captured through it. Capture-session metadata is needed for that claim.
[S1–S3]

## Smallest useful preprocessing interface

Use one shared function for a controlled offline experiment and any later live
integration:

```python
processed_bgr = preprocess_bgr(original_bgr, preset)
result = model.predict(processed_bgr, **fixed_inference_settings)
```

The intended contract is:

1. **Identity is the default.** It must preserve the original pixel values.
2. Accept and return an `H × W × 3` BGR `uint8` array. Reject unsupported shapes
   and dtypes. The function does not crop, resize, rotate or mutate its input.
3. Start with one prespecified alternative, such as Lab-L CLAHE with clip limit
   2.0 and an 8 × 8 tile grid. Those are experimental settings, not a tuned or
   proven optimum.
4. Keep `original_bgr` for source-image review, annotation, snapshots and raw
   recording. Draw predictions in its unchanged coordinate system. Save an
   explicitly named processed preview if useful for diagnosis.
5. Record the preset, parameters, OpenCV version, source-image hash,
   checkpoint hash, all inference settings and preprocessing time separately
   from model time. Use the same implementation offline and online if the
   alternative later earns a deployment trial.

This interface is a proposal; this document does not install a live filter.
Changing image size is a separate experiment and should not be bundled with
CLAHE. Compare identical photographs and frozen annotations at identical
confidence and NMS settings. Report missed targets, unmatched outputs,
duplicates, endpoint errors, empty-scene outputs and latency together.

Keep annotation decisions anchored to the original image and physical
cassette-front policy. A brighter boundary after enhancement is a different
rendering of the same measurement, not independent proof of material ownership.
An inference-only comparison tests sensitivity of the existing checkpoint to
preprocessing. If a filter is later used for training, both training and
deployment pipelines must state that choice, and the resulting model requires
its own comparison.

## Source references

Repository sources below are pinned to the audited commit; the JSON companion
also records their Git object IDs and SHA-256 hashes.

- **S1.** [Live viewer: capture, controls, prediction, snapshots and reports](https://github.com/Siuuuers/dl/blob/9983d996e087aa6fd348a1861b61a1c99ec88a0d/scripts/view_d405_yolo.py).
- **S2.** [Live startup command](https://github.com/Siuuuers/dl/blob/9983d996e087aa6fd348a1861b61a1c99ec88a0d/start_live.sh).
- **S3.** [D405 recording implementation](https://github.com/Siuuuers/dl/blob/9983d996e087aa6fd348a1861b61a1c99ec88a0d/scripts/d405_recording.py).
- **S4.** [Folder prediction and optional postclip NMS](https://github.com/Siuuuers/dl/blob/9983d996e087aa6fd348a1861b61a1c99ec88a0d/scripts/predict_pose_folder.py).
- **S5.** [Packaged base manifest](https://github.com/Siuuuers/dl/blob/9983d996e087aa6fd348a1861b61a1c99ec88a0d/weights/base.json) and [checksum-enforcing resolver](https://github.com/Siuuuers/dl/blob/9983d996e087aa6fd348a1861b61a1c99ec88a0d/scripts/base_weights.py).
- **S6.** [Retained research checkpoint and settings](https://github.com/Siuuuers/dl/blob/9983d996e087aa6fd348a1861b61a1c99ec88a0d/docs/current-model.md).
- **D1.** RealSense, [D405 product documentation](https://www.realsenseai.com/products/stereo-depth-camera-d405/): color generation from the left imager with ISP.
- **D2.** RealSense, [Post-processing filters](https://dev.realsenseai.com/docs/post-processing-filters/): depth pipeline and temporal-filter limitations.
- **D3.** RealSense, [`pyrealsense2.options`](https://realsenseai.github.io/librealsense/python_docs/_generated/pyrealsense2.options.html): supported-option enumeration, value access and support checks.
- **D4.** OpenCV, [Histogram equalization and CLAHE](https://docs.opencv.org/4.13.0/d5/daf/tutorial_py_histogram_equalization.html) and [CLAHE API](https://docs.opencv.org/4.13.0/d6/db6/classcv_1_1CLAHE.html).
- **D5.** OpenCV, [Color conversions](https://docs.opencv.org/4.13.0/de/d25/imgproc_color_conversions.html): BGR/Lab conversions.
- **D6.** OpenCV, [`VideoWriter`](https://docs.opencv.org/4.13.0/dd/d9e/classcv_1_1VideoWriter.html): codec, frame rate and lossless/raw recording distinctions.

External documentation was consulted on 27 November 2026. Its current version
does not establish the versions installed on a particular capture machine. The
repository lock file at the audited commit declares `pyrealsense2==2.59.1.11053`,
`opencv-python==5.0.0.93` and `ultralytics==8.4.182`; a future capture record
should report the versions actually loaded.
