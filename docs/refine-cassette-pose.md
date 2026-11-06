# Refine the cassette lower-corner pose model

Baseline to beat: `weights/cassette_lower_epoch173.pt` (YOLO26s-Pose, epoch 173). Target is the white cassette front’s actual lower-left and lower-right corners (`front_lower_left`, `front_lower_right` in [`data/data.yaml`](../data/data.yaml)), ordered by image X. Not the upper lip, tray, or basket. One class, `kpt_shape: [2, 3]`, `flip_idx: [1, 0]`. An endpoint that cannot be located stays `0 0 0` (`v=2` only when it is visible).

Do the inference sweep before any training. Change one thing at a time. Keep gold labels in `data/labels/` and this checkpoint byte-for-byte.

## What is already fixed

Recorded training ([`metadata/training_config.json`](../metadata/training_config.json)) is not the Colab smoke cell.

| | Recorded run | [`notebooks/cassette_colab.ipynb`](../notebooks/cassette_colab.ipynb) |
| --- | --- | --- |
| Init | Official COCO YOLO26s-Pose (not the old epoch-127 cassette weights) | Epoch 173 |
| Length | 200 epochs, then resumed to 300 | 3 epochs, pipeline check only |
| Input / batch | `imgsz=1280`, batch 8, `rect=True` | `imgsz=1280`, batch 4 |
| Optim | AdamW, `lr0=0.001`, `lrf=0.01`, `amp=False`, seed 42, `deterministic=True` | left to Ultralytics defaults |
| Augment | mosaic / mixup / cutmix / copy-paste off; `degrees=5`, `translate=0.05`, `scale=0.15`, `fliplr=0.5`; `hsv_h=0.01`, `hsv_s=0.2`, `hsv_v=0.2` | not set |

Epoch 173 won on the minimum sum of `val/*_loss` over all 300 epochs (**1.79037**). Training longer than 173 is not a new experiment. The notebook’s “drop `imgsz` to 640” line is a speed check, not a quality setting.

Package the comparison with the checkpoint’s Ultralytics, **8.4.182** (`environment.yml`, `metadata/model_card.json`). The notebook’s `%pip install -U ultralytics` floats the version; do not use that environment for a number you will keep.

Gold split, from [`data/README.md`](../data/README.md): **137 train / 20 val** images, **3772 / 664** objects, **13 / 3** empty scenes. Images are 1280×720. Val was used to pick the checkpoint, so it is not a held-out test set. Final-test and gap images are not in this package.

## 1. Inference confidence sweep

Same weights, same 20 val images, same device for every row. GPU is fine; the live viewer refuses CPU, while [`scripts/export_pose_drafts.py`](../scripts/export_pose_drafts.py) runs CPU. Do not mix the two in one table.

Match the live viewer ([`scripts/view_d405_yolo.py`](../scripts/view_d405_yolo.py), [`weights/base.json`](../weights/base.json)), not the exporter or the notebook. The architecture yaml sets `end2end: true`. The viewer forces `nms=True` and aborts if the predictor stays end-to-end, because the IoU control would do nothing. The exporter and the notebook pass neither `nms`, `iou`, nor `rect`.

```python
model.predict(
    source="data/images/val",
    imgsz=1280,
    rect=True,
    conf=0.25,       # sweep 0.15, 0.25, 0.40; nothing else changes
    iou=0.7,
    nms=True,
    max_det=300,
    half=False,      # FP32
    save=False,
    verbose=False,
)
```

Count, per threshold: missed cassettes (including thin edge slivers), duplicate boxes on one cassette, and background boxes. The viewer keys `1/2/3` are 0.10 / 0.25 / 0.50 for a live look; they are not this sweep. `--overlap-iou` (default 0.5) only paints a diagnostic; it does not filter.

Endpoint display confidence is separate and stays **0.50** until the box sweep is recorded. Score a corner only when its keypoint confidence is at least 0.50, in original 1280×720 pixels. The exporter’s draft writer uses a different cutoff (`0.01`, or out of frame → `0 0 0`). Do not treat that as the display threshold.

Only if duplicates remain at 0.25, rerun that one threshold at `iou=0.5` and at `iou=0.7`. Confirm neighboring cassettes are still both kept. IoU applies on the NMS path (`nms=True`); it does nothing useful in end-to-end mode.

## 2. Paired fine-tune, after labels are reviewed

Keep `data/` as the reproducible arm. Build a second tree for reviewed **train** labels only. [`tools/cassette_labeler`](../tools/cassette_labeler/README.md) overwrites whatever folder you open, so open the copy, not `data/`.

- Same 137/20 images, same empty-scene files, same `kpt_shape` / `flip_idx` / class name.
- `labels/val` byte-identical on both arms. Never train on val drafts.
- `labels/train` differs only where a review actually changed a file. An object-count change is an annotation edit, not a measured false positive or false negative. A file with the same count can still be a corner move.

[`scripts/export_pose_drafts.py`](../scripts/export_pose_drafts.py) deletes all of `data_pred/` (`shutil.rmtree`) and rewrites **val only** (`conf=0.25`, `imgsz=1280`, CPU). Copy reviewed drafts out before running it again. Do not commit `data_pred/`.

Two 30-epoch runs from epoch 173, `resume=False` (a new run, not a continuation of the 300-epoch schedule). Same arguments on both arms; only `data` changes. Start unfrozen.

```python
model = YOLO("weights/cassette_lower_epoch173.pt")
model.train(
    data="data/data.yaml",   # other arm: the candidate yaml
    epochs=30,
    resume=False,
    imgsz=1280,
    batch=4,                 # original run used 8; both arms stay at 4
    rect=True,
    optimizer="AdamW",
    lr0=0.0001,              # original lr0 was 0.001
    lrf=0.01,
    amp=False,
    seed=42,
    deterministic=True,
    patience=0,
    warmup_epochs=3.0,
    weight_decay=0.0005,
    momentum=0.937,
    nbs=64,
    mosaic=0.0,
    mixup=0.0,
    cutmix=0.0,
    copy_paste=0.0,
    close_mosaic=0,
    degrees=5.0,
    translate=0.05,
    scale=0.15,
    shear=0.0,
    perspective=0.0,
    flipud=0.0,
    fliplr=0.5,
    hsv_h=0.01,
    hsv_s=0.2,
    hsv_v=0.2,
    box=7.5,
    cls=0.5,
    dfl=1.5,
    pose=12.0,
    kobj=1.0,
    save_period=10,
)
```

Pass those explicitly. The smoke cell does not, and a newer Ultralytics default will not match this run. Also set `warmup_momentum=0.8` and `warmup_bias_lr=0.1` if you touch warmup at all; leave them at the recorded values on both arms (`warmup_epochs=3` is a tenth of this run, so do not “fix” it on one arm).

`freeze=10` is a follow-up, and only if validation gets worse while training loss drops. Ultralytics freezes the first N layers.

Keep a new checkpoint only when it beats epoch 173 on the **same** val evaluation (same `conf` / `iou` / `imgsz` / `nms`, same val labels). Ultralytics `best.pt` follows fitness, not the summed `val/*_loss` that selected epoch 173. Use `save_period` and pick the epoch yourself.

## Do not change first

- Gold labels, the epoch-173 file, the 137/20 split, or the empty-scene negatives.
- `imgsz`. Stay at 1280. Try 960 only if latency is the actual constraint, and report endpoint pixel error and latency together. 640 does not decide quality.
- Augmentation and loss gains (`box=7.5`, `pose=12.0`, `kobj=1.0`, `cls=0.5`, `dfl=1.5`). If leftovers are clearly lighting, the only aug follow-up is `hsv_v` 0.2 → 0.3, alone.
- NMS IoU, unless the 0.25 sweep still shows duplicates.
- Architecture (`models/cassette_lower_yolo26s_pose.yaml`) and the decision to train from epoch 173 rather than from COCO again.

## Metrics to write down

Epoch 173 validation, from the README (val, not a final test):

| | Box | Pose |
| --- | ---: | ---: |
| Precision | 0.96670 | 0.98297 |
| Recall | 0.91810 | 0.93675 |
| mAP50 | 0.93764 | 0.95993 |
| mAP50–95 | 0.58654 | 0.95963 |

Summed val loss **1.79037** (`box` 1.22175, `pose` 0.01252, `kobj` 0.22996, `cls` 0.3223, `l1` 0.00384, `rle` 0). Pose mAP50–95 is high and box mAP50–95 is not; neither number is corner error in pixels.

For every candidate also log, at 1280×720: missed cassettes, duplicate and background boxes, missed slivers, and the median and 95th percentile pixel error of endpoints that are visible on both the prediction and the gold label. Endpoint confidence is not that error.
