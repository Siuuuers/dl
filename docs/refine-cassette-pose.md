# Refine the baomai cassette lower-corner pose model

Use the supplied `baomai` baseline in `Siuuuers/dl`. First measure it reproducibly, inspect its errors, and train only when a specific data improvement is ready. This is an experiment plan; it records no new inference or training results.

## Baseline to preserve

Upstream reference: [`an94-doll/embedding_vision`, `baomai` at `351eb3cf960ccc34829e2bc72e1fa4645acc6295`](https://github.com/an94-doll/embedding_vision/tree/351eb3cf960ccc34829e2bc72e1fa4645acc6295). Workspace behavior checked at `b4df76935178de5a122119aea48e85d2d661effc`.

- Model: `weights/cassette_lower_epoch173.pt`, YOLO26s-Pose. Preserve its bytes and the SHA-256 in [`metadata/model_card.json`](../metadata/model_card.json): `e0aab1b377491700318e0b14cd4c558ec3c1684087a13b7dde2298ebdbf2847e`.
- Target: the white front's actual lower-left and lower-right corners, ordered by image X. Exclude upper-lip, tray, and basket corners. Keep `cassette_front`, `kpt_shape: [2, 3]`, and `flip_idx: [1, 0]` from [`data/data.yaml`](../data/data.yaml).
- Label only reliably visible endpoints as `v=2`; an unlocatable endpoint stays `0 0 0`. A visible endpoint is retained when its partner is missing. Do not infer hidden corners from an occlusion boundary.
- Frozen data: 137 train / 20 validation images, 3,772 / 664 objects, and 13 / 3 empty scenes. These labels are already human corrected. Keep [`data/labels/`](../data/labels/) and the original split unchanged; edit a separate working copy.
- Use the 1280×720 raw PNGs in `data/images/`. The 1280×750 previews in `data/annotated/` contain drawings and a footer; they are review aids, not model inputs. See [`data/README.md`](../data/README.md).

The supplied model was initialized from official COCO YOLO26s-Pose, trained to 200 epochs, and resumed to 300. Epoch 173 had the lowest summed validation loss, 1.79037. Its recorded box/pose mAP50–95 values are 0.58654 / 0.95963. These are historical validation measurements, not newly reproduced results or endpoint errors in pixels.

Pin **Ultralytics 8.4.182**, as specified by the model card. Record Python, PyTorch, CUDA, device, and precision too. The Colab notebook's floating `%pip install -U ultralytics` and three-epoch smoke test are not a controlled experiment configuration.

## 1. Measure before training

The next execution task is an offline evaluator on all 20 unchanged validation images. The current draft exporter does not compute the metrics below. This note specifies the evaluator contract; it does not claim that an evaluation command already exists.

Match [`weights/base.json`](../weights/base.json) and [`scripts/view_d405_yolo.py`](../scripts/view_d405_yolo.py). The live viewer explicitly requests `nms=True` and checks that the predictor leaves end-to-end mode. Apply the same check in the evaluator; a named IoU setting alone is insufficient.

```python
from ultralytics import YOLO

model = YOLO("weights/cassette_lower_epoch173.pt")
results = model.predict(
    source="data/images/val", imgsz=1280, rect=True,
    conf=0.25, iou=0.7, nms=True, max_det=300,
    half=False, save=False, verbose=False,
)
```

This is an inference configuration, not the scoring implementation. Select one device explicitly for the run and record it. Keep the same image order, device, precision, and settings across comparisons.

Start at box confidence **0.25**; then compare **0.15 / 0.25 / 0.40** on this same checkpoint. Keep endpoint confidence **0.50** and NMS IoU **0.70** fixed. These are reproducible diagnostic defaults, not supervisor acceptance criteria. Do not choose different thresholds for different candidate models.

Keep 0.25 as the first model-comparison setting. If the sweep motivates another setting, declare it before comparing models and reevaluate every arm at that setting. Test NMS IoU 0.50 versus 0.70 only as a separate follow-up if duplicates remain, checking that neighboring cassettes survive.

### Object matching and counts

Use a separate **matching IoU threshold of 0.50**; it is not the NMS IoU threshold.

1. For each image, sort accepted predictions by descending box confidence, breaking equal-confidence ties by their original prediction index.
2. For each prediction, find unmatched gold boxes of the same class with IoU ≥ 0.50. Match the one with greatest IoU; break ties by original gold-label row order. Each prediction and gold object can appear in only one pair.
3. Matched pairs are TP; unmatched gold objects are FN; unmatched predictions are FP. A duplicate FP overlaps an already matched gold box at IoU ≥ 0.50. All other unmatched predictions are other FP; visual review can distinguish background detections from localization errors.
4. Report these counts per image and summed across the dataset. Include empty scenes and edge slivers. Do not infer FP/FN from differences in annotation-row counts.

### Endpoints: coverage and conditional pixel error

- Denominator: every gold endpoint marked `v=2`, including endpoints on missed objects. Compare the corresponding left/right index within matched object pairs; do not swap endpoints to reduce error.
- A returned endpoint requires a matched object, endpoint confidence ≥ 0.50, finite coordinates, and an in-frame location (`0 ≤ x < 1280`, `0 ≤ y < 720`). Gold endpoints marked missing are excluded from endpoint error and coverage denominators.
- Report visible-gold endpoint count, returned count, missing count, and coverage (`returned / visible_gold`). Low-confidence/out-of-frame points and endpoints on unmatched gold objects contribute to missing count. Use `null` coverage if the denominator is zero.
- Convert normalized gold coordinates with the original width/height; measure Euclidean distance in original-image pixels. Report median and p95 over returned, matched, gold-visible endpoints only. Label these errors **conditional on returned endpoints**. If there are no such points, both values are `null`, never zero.
- Use the same percentile calculation throughout: sort distances and linearly interpolate at index `q × (n − 1)` for `q=0.50` and `0.95`. Keep counts beside the error statistics so suppressed endpoints cannot disappear from the assessment.

Save a compact table and failure overlays on raw images. Record preprocessing, prediction, and postprocessing timing definitions, warm-up exclusion, sample count, and device; do not equate offline throughput with camera/display FPS. No pixel or millimetre acceptance tolerance has been specified.

## 2. Review the error evidence

The existing command below creates review drafts, not an evaluation report:

```sh
python scripts/export_pose_drafts.py
```

It currently exports **validation only**, at `conf=0.25`, `imgsz=1280`, on CPU. It does not explicitly set `nms`, `iou`, or `rect`, so its predictions must not stand in for the fixed evaluation above. Its endpoint-label cutoff of 0.01 is also different from the 0.50 evaluation/display threshold.

The exporter **stops with `FileExistsError` when `data_pred/` exists**; it does not delete the folder. Retain existing corrections by moving that working folder before deliberately starting a fresh export. Keep `data_pred/` out of Git.

Open the working copy in [`tools/cassette_labeler`](../tools/cassette_labeler/README.md). Saving writes corrected `.txt` labels into the chosen folder. Validation drafts remain diagnostic material; do not promote their images or labels into training. If a genuine gold-validation error is discovered, version the corrected evaluation set and rerun every comparison rather than silently changing one arm's reference.

## 3. Optional controlled fine-tune

Proceed only after documenting verified **training-label corrections** or reviewed **new training data**. Record changed files and reasons. New images must be separate from validation/final-test scenes; preserve empty-scene negatives. If no training data changed, skip the label-improvement experiment.

| Arm | Initialization | Training data | Role |
| --- | --- | --- | --- |
| Baseline | Supplied epoch 173 | No new training | Reference to retain |
| Reference fine-tune | Fresh load of epoch 173 | Original 137 training images/labels | Measures extra-training effect |
| Candidate fine-tune | Fresh load of epoch 173 | Copy with verified train changes | Tests the planned data change |

Both fine-tune arms use byte-identical validation images/labels and shared settings. Train-label corrections should otherwise preserve the same images. If new images are added, record image counts and optimizer-step counts: equal epochs no longer mean equal computation, and this is a new-data experiment.

Proposed settings for each arm follow [`metadata/training_config.json`](../metadata/training_config.json), with a new 30-epoch schedule, batch 4, and `lr0=0.0001`. This is an optional training recipe, not a completed run:

```python
from ultralytics import YOLO

model = YOLO("weights/cassette_lower_epoch173.pt")
model.train(
    data="data/data.yaml", project="runs/refinement", name="reference",
    epochs=30, resume=False, imgsz=1280, batch=4, rect=True,
    optimizer="AdamW", lr0=0.0001, lrf=0.01, amp=False,
    seed=42, deterministic=True, patience=0, freeze=0,
    weight_decay=0.0005, momentum=0.937, nbs=64,
    warmup_epochs=3.0, warmup_momentum=0.8, warmup_bias_lr=0.1,
    mosaic=0.0, mixup=0.0, cutmix=0.0, copy_paste=0.0, close_mosaic=0,
    degrees=5.0, translate=0.05, scale=0.15, shear=0.0,
    perspective=0.0, flipud=0.0, fliplr=0.5,
    hsv_h=0.01, hsv_s=0.2, hsv_v=0.2,
    box=7.5, cls=0.5, dfl=1.5, pose=12.0, kobj=1.0,
    save_period=10,
)
```

Reload epoch 173 for the other arm and change only the dataset YAML and run name. Choose the same training device explicitly and record it. Keep architecture, resolution, augmentations, loss gains, and freezing unchanged in this first experiment.

## 4. Decide and preserve evidence

For the first paired comparison, evaluate the final epoch-30 checkpoint from each fine-tune arm with the fixed baseline evaluation. Do not silently substitute Ultralytics `best.pt`, which follows its fitness criterion, or select a different epoch rule for each arm. Intermediate checkpoints are diagnostic until a separate selection experiment is declared.

Use a conservative comparison rule while supervisor acceptance tolerances remain unspecified: a model can displace the baseline only if FN, total FP, visible-endpoint missing count, conditional median error, and conditional p95 error are all no higher, with at least one strictly lower. Compare unrounded statistics. Unavailable endpoint statistics leave the decision unresolved; retain the baseline. If alternatives trade improvements against regressions, report the tradeoff and keep the current default.

Assess the reviewed-data candidate against **both** baseline and reference fine-tune. Attribute a gain to the data change only if the candidate also meets that comparison rule against the reference. One seeded comparison is preliminary evidence, not proof of repeatable improvement or deployment readiness.

For each result retain: code revision; model SHA-256; image/label hashes and split; runtime/device; inference and matching settings; metric definitions; commands and exit status; result table and failure examples; and limits. Keep artifacts separately from the frozen baseline.

Validation already selected epoch 173 and will guide these experiments. It remains a development set. The five final-test and nineteen gap images mentioned in the data README are absent from this package; obtain genuinely untouched test data before making independent-test or generalization claims. Do not claim a new accuracy result from documentation, editor checks, or draft exports.
