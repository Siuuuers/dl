# R53 inference comparisons

These two notebooks prepare evidence for the annotation thesis without another
training run. Neither comparison has produced new model results at publication.
The CPU tests and saved-result replays establish that the experiment code is
consistent; they do not establish that either treatment improves detections.

| Notebook | Control | One treatment | Main question |
|---|---|---|---|
| [Luminance CLAHE](../notebooks/baomai_clahe_inference_r53.ipynb) | Original BGR pixels, size 1280 | OpenCV Lab L-channel CLAHE, clip limit 2.0, tile grid 8×8; size 1280 | Does this fixed inference-only contrast transform recover partial fronts without increasing wrong outputs? |
| [Inference size](../notebooks/baomai_scale_inference_r53.ipynb) | Original BGR pixels, size 1280 | Same pixels, size 1600 | Does a larger network input improve small-front sampling enough to justify its measured cost? |

Here OpenCV Lab **L is perceptual lightness**, not linear physical luminance.
The transform changes appearance, not scene geometry or captured information.

The CLAHE settings are a prespecified hypothesis, not parameters selected by
searching these 20 photographs. The model has not been adapted to the filtered
distribution. CLAHE can amplify shadows, texture or noise; a negative result
would reject this particular inference-only pipeline, not every possible
photometric method. The size comparison changes model preprocessing only.
Interpolation adds no new camera detail. Read the hypotheses and primary-source
limitations in [the literature record](thesis-literature-r53.md), and the actual
application behavior in [the camera pipeline audit](camera-pipeline-r53.md).

## Fixed evidence and model contract

The input checkpoint is the **PR137 candidate**, SHA256
`222bf3c91a651183047a10eb1a50e16ae8fe999f15372afe005bfca2d0b157f9`.
Its source archive is 42,395,650 bytes with SHA256
`fc8c40357c83831d17f88b0bb5a5ecd2e0cb1a8128be697e37e80620e4981023`.
The PR146 reference's saved predictions equal this candidate's predictions on
the 20 development images. Its checkpoint bytes differ, so the notebook loads
the explicitly identified PR137 file. This is a research comparison; it does
not change the selected PR35 deployment checkpoint. See the
[PR146 audit](partial-classification-r51-results.md).

The model card identifies **YOLO26s-Pose**, one `cassette_front` class and two
three-value keypoint slots, ordered left then right. The YAML architecture uses
the native `Pose26` head with `reg_max=1`. Its original training architecture
has one-to-many and one-to-one branches; this experiment explicitly uses the
existing ordinary one-to-many FP32 inference path with NMS. It does not replace
the head, change the native RLE training objective, or load a generic public
detector. The reviewed training experiments initialize from the recorded
epoch 173 cassette checkpoint, SHA256
`e0aab1b377491700318e0b14cd4c558ec3c1684087a13b7dde2298ebdbf2847e`.
The earlier initialization hash is recorded in the model card, but that hash
alone does not identify a public pretraining dataset or prove random scratch
initialization. Sources: [model card](../metadata/model_card.json),
[architecture](../models/cassette_lower_yolo26s_pose.yaml),
[checkpoint contract](../scripts/pose_checkpoint.py), and
[PR137 audit](../scripts/audit_additional_front_pilot_r47_results.py).

The checkpoint requires **Ultralytics 8.4.182**. The tested package wheel is
`ultralytics-8.4.182-py3-none-any.whl`, SHA256
`11b90b26fa30f878807b0b8c4ed1870eb3c54e7e876a17a936bd92065a7e743b`.
Its four previously recorded native source hashes match the PR137 provenance.
R53 additionally enforces six inference-source hashes for the head, task loader,
predictor, pose predictor, NMS and coordinate operations. The setup pins
OpenCV 5.0.0.93, NumPy 2.4.6 and Pillow 13.0.0. It retains and records the notebook
runtime's compatible Torch/torchvision pair through an installation constraints
file; it does not silently install a different CUDA stack. An unsupported runtime
fails its checks. The original return used Torch 2.14.1+cu130 and Python 3.13.16;
runtime differences are recorded and must pass the fresh baseline gate.

## Baseline reproduction is required before treatment

Both notebooks first run unchanged 1280 inference on all 20 hash-verified original
1280×720 PNGs. The fresh control is compared with the PR137 saved raw and postclip
outputs. Exact JSON equality is reported separately from these tolerances:

| Quantity | Maximum permitted difference |
|---|---:|
| Each box or keypoint coordinate component |0.05 original-image pixel |
| Box confidence or keypoint confidence |0.00001 |
| Eligible endpoint distance from its reference |0.075 pixel |
| Per-image counts, match IDs, miss IDs, unmatched IDs, duplicate IDs and retained/suppressed raw IDs | Exactly unchanged |
| Each reference row's returned visible slots and 5/10-pixel success state | Exactly unchanged |

All four references and both output views must pass. Small floating-point noise
cannot justify a different discrete conclusion. Even a 0.02-pixel change that
crosses a 5-pixel success boundary fails this gate. A failed gate stops the
treatment and packages the control diagnostics with
`interpretation_allowed=false`; it is not interpreted as a treatment result.
All three repeated outputs are preserved so return verification can also
reconstruct the repeatability gate. A repeatability failure blocks interpretation.

The controls retain confidence 0.13, native NMS IoU 0.70, the existing postclip
NMS, maximum 300 outputs, batch 1, rectangular preprocessing, quantization 32 and
FP32. Matching remains IoU 0.50 with endpoint confidence 0.50. These are the
existing repository settings, not thresholds tuned for R53. Sources:
[inference](../scripts/evaluate_pose.py),
[postclip filtering](../scripts/predict_pose_folder.py), and
[scoring](../scripts/pose_scoring.py).

## Endpoints, references and interpretation

The frozen666 reference is the primary comparison. Historical663, R38 and the
reviewed R50 reference remain separately labeled diagnostic views. R50 uses the
currently reviewed target policy; differences between reference definitions can
change apparent gains. Report disagreement rather than choosing the reference
with the most favorable result. No reference row or original image changes in
either notebook.

Reports contain misses, unmatched outputs, formal duplicates and other unmatched
outputs; counts by 0/1/2 locatable corners; original-height strata `<8`, `8–<16`,
`16–<32` and `≥32` pixels; eligible endpoint errors; and both-corner success within
5 and 10 original pixels. Raw and postclip counts are retained separately. A
formal duplicate is defined by the existing matching protocol; it is not simply
a visually similar output. Other unmatched predictions require image review
before being assigned a visual cause.

Paired reference rows have stable row identities and explicit recovery/new-miss
and endpoint-success gain/loss flags. Predictions display both their one-based
postclip output position and their retained raw `P` number. These numbers are
local to one arm and image; they are not persistent cross-arm object IDs. An
example such as `out037(P041)` must not be compared to output 37 in another arm
as if the numbers identify the same cassette.

The conservative screen requires fewer missed targets on the frozen666 primary
reference, with no increase in total unmatched outputs or formal duplicates,
no newly missed two-locatable-corner target, no loss of all-eligible visible-corner
5/10-pixel success, and no new output on a reference-empty scene. Report the full trade-off if these
protections are not met. A benefit visible only on R50 is a diagnostic finding
under that target policy, not a primary-reference success. Runtime cost and all
per-image regressions remain part of the result. These are development screens,
not a statistical proof or an automatic deployment rule. The 20 images have
already guided repeated choices; future capture-session evaluation is required.

## Timing and resource cost

After the baseline gate, the runner warms each arm twice, then measures three
paired repeats over all 20 originals, alternating arm order by image and repeat.
It records photometric transformation time, prediction-call time, native internal
stage times, CPU prediction-record/postclip time, and the complete path from a
decoded original array to final outputs. CUDA synchronization uses the actual
backend device. Initialization, archive verification, file reading and camera
acquisition are excluded. These results must not be labeled camera FPS.

The return includes actual padded network tensor shapes, resize gain, resized
height/width, and left/top/right/bottom padding. Effective object height scales
with the resize gain, not with padded tensor height divided by 720. With the
verified aspect-preserving transform, the intended gains are 1.0 at 1280 and 1.25
at 1600. Keypoints are converted back to original coordinates before scoring.
CUDA allocated and reserved peaks are reported separately; allocator statistics
are not total device memory occupancy. Out-of-memory errors do not trigger an
automatic batch, precision, image-size or checkpoint change.

## Run and return

Use the two notebooks separately in a GPU runtime. Each has four code cells:
load the fixed source; audit inputs on CPU; install the compatible inference
packages and run; verify and download. The notebooks can share
`BAOMAI_R53_BASE=/content/baomai-r53`: source and cache are shared, while the
`clahe` and `scale` work directories are separate. Existing incompatible files
are preserved. Select a fresh base to start an independent run.

The return files are `baomai-clahe-inference-r53-results.zip` and
`baomai-scale-inference-r53-results.zip`. They include predictions, all repeated
predictions, baseline and repeatability checks, paired cases, timing/memory,
runtime and native-source receipts, the input audit, the exact driver and a member
hash inventory. They omit redundant model weights and photographs. The CPU
`verify` command reconstructs filtering, scores, paired cases and gates from the
returned predictions. It does not perform fresh inference.

The shared command is:

```bash
python -B scripts/run_inference_ablation_r53.py prepare \
  --experiment clahe --source-commit FULL_PUBLISHED_COMMIT \
  --script-sha256 PUBLISHED_DRIVER_SHA256 --root SOURCE --work WORK
```

Use `run` for inference and `verify` for replay. `--archive`, `--originals-dir`
and `--cache-dir` accept existing verified inputs. The runner never silently
trusts a cache because its filename looks correct. The publication builder is
`scripts/build_inference_ablation_r53_notebooks.py --source-commit FULL_COMMIT`;
the published notebooks contain complete immutable pins.

## Validation completed before publication

Nine focused CPU tests pass with the actual pinned OpenCV and NumPy packages.
They cover source-pixel preservation and same-geometry CLAHE, dtype/channel
rejection, original-pixel endpoint scoring, confidence/coordinate tolerances,
the strict threshold-crossing gate, distinct duplicate/missed identities, stable
paired recovery IDs, blocked-result packaging/replay, and both four-cell notebook
builders. The test file is
[test_inference_ablation_r53.py](../tests/test_inference_ablation_r53.py).
The package source inspection matched the pinned native hashes without loading
Torch or model weights. Fresh GPU inference and measured model improvements are
not claimed by these checks.


### Published-source notebook checks

Both notebooks pin source commit
`7b36e23dbbc24769f92a88cb93ae44acfc0406c5` and driver SHA256
`8c8a37c3dcaccde5c568e9df6f44651cdb05965ca5f5ed5b0f2978852387dcc1`.
The actual first two code cells passed against a fresh GitHub checkout, with
verified archive and original-image caches. The second notebook reused the
clean source/cache and created its own work directory. A resumed test omitted
the model-card path; another omitted the R38 row-map path. Rerunning the actual
cells restored each required path and preserved the corresponding preflight
hashes. Each audit checked 20 original images and 676 source/data files.

All four code cells compile. The inference/install cell and final result-download
cell were not executed in this environment: Torch/GPU inference remains for the
notebook runtime. Synthetic archive tests and replay of old predictions do not
replace that pending experiment.

[Published notebook check receipt](../results/review/thesis-r53-20261127/notebook-validation.json)
and [implementation check receipt](../results/review/thesis-r53-20261127/implementation-validation.json)
retain the exact completed checks and their limits.
