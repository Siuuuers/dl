# Handoff for the cassette YOLO-pose work

This document records the cassette pose work on Git commit `601bfc31cad10b4eb5898b850f965467563efda9`.
The sentences follow ASD-STE100 practice.
Each sentence has one idea.
The same word has the same meaning.
Verbs are active.
This document does not change code, weights, or labels.
This document does not add new measurements.

## 1. Words in this document

Use these words with one meaning:

- **Gold label.** A YOLO-pose `.txt` file under `data/labels/` in that commit.
- **Draft label.** A YOLO-pose `.txt` file that the exporter writes under `data_pred/labels/`.
- **Correction.** An edit of a draft label after export. A correction is not a model metric.
- **Candidate.** A saved copy of reviewed draft labels for a later training test.
- **Endpoint.** `front_lower_left` or `front_lower_right` on the white cassette front. The two endpoints are the lower corners. Order the endpoints by image X.
- **Sliver.** A cassette that covers a narrow region in the image.
- **Session record.** A count or a file list from the Codex summaries supplied for this handoff.

An endpoint that you cannot locate stays `0 0 0`.
Do not mark the upper lip, the tray, or the basket.

`data/data.yaml` names the endpoints `front_lower_left` and `front_lower_right`.
`weights/base.json` names them `lower_front_left` and `lower_front_right`.
Use the `data.yaml` names in labels and in the notebook.

## 2. Investigations

### 2.1 Draft labels

The exporter is `scripts/export_pose_drafts.py`.
The exporter on GitHub loads `weights/cassette_lower_epoch173.pt`.
It runs on CPU with `conf=0.25` and `imgsz=1280`.
It writes `data_pred/images/` and `data_pred/labels/` for the labeler.
The exporter on GitHub exports only the `val` split.
The exporter deletes `data_pred/` before it writes.
Image entries are symbolic links.
Each link target is an absolute path from `Path.resolve()`.

The session record says the first validation drafts had 20 images and 688 objects.
The session record says the first training drafts had 137 images and 3,989 objects.
This checkout does not contain `data_pred/`.
This document does not count those draft files again.
Do not use 688 or 3,989 as gold counts.

### 2.2 Corrections

The session record says three validation images were corrected:

- `new_000007`
- `new_000008`
- `new_000146`

Seventeen validation images are still pending review.
Each of those three stems is under `data/labels/val/` in this checkout.
The gold file `data/labels/val/new_000146.txt` is empty (0 objects).
The corrected row text is not on GitHub.

The session record says the training review finished 12 correction batches.
The session record says the corrected training drafts then had 3,765 objects.
The session record says 125 of 137 training label files have the same bytes as the gold files.
The session record names these 12 different files:

- `new_000013`
- `new_000015`
- `new_000019`
- `new_000023`
- `new_000075`
- `new_000083`
- `new_000094`
- `old_000003`
- `old_000004`
- `old_000007`
- `old_000010`
- `old_000013`

Each of those 12 stems is under `data/labels/train/` in this checkout.
The session record gives object-count changes for six `new_*` files only:

| File | Change in object count, candidate minus gold |
| --- | ---: |
| `new_000015` | +2 |
| `new_000083` | +2 |
| `new_000019` | -5 |
| `new_000023` | -1 |
| `new_000075` | -3 |
| `new_000094` | -2 |

The sum of those six changes is -7 objects.
Gold training labels in this checkout have 3,772 objects.
3,772 - 7 = 3,765.
That arithmetic agrees with the session total of 3,765 objects.
The session record does not give object-count changes for the six `old_*` files.
A file can keep the same object count and still change an endpoint.
These differences are label corrections.
They are not measured false-positive rates or false-negative rates.

The session record describes the review actions as removal of duplicate boxes, addition of slivers, and small moves of corners.
This checkout does not contain the review log.

### 2.3 HTML limits and egress limits

The labeler is `tools/cassette_labeler/index.html`.
The labeler is a local page.
The labeler does not need a network connection.
Open it in desktop Chrome or Edge.
If `file://` blocks folder access, serve the repository root:

```sh
python3 -m http.server 8000 --bind 127.0.0.1
```

Then open `http://localhost:8000/tools/cassette_labeler/`.

The labeler asks you to choose a data folder.
You edit boxes and endpoints on a canvas.
**Save label** replaces one `.txt` file.
The labeler does not run inference.
The labeler does not update `data/annotated/`, `data/manifest.json`, or `data/SHA256SUMS.txt`.

A Codex review of the labeler gave advice only.
That review did not change `index.html`.
That review did not make a commit.
The advice note names the log `labeler_upgrade_ideas.jsonl`.
That log file is not in this checkout.

The advice asks for these controls, in this order:

1. Add a collapsible label-text editor with Apply. Use the current parser. Draw the canvas again after Apply. Write the file only when Save succeeds.
2. Add one undo step. Restore the state before the last drag, deletion, or point change.
3. Add Previous, and Save and next. Open the next image only after Save succeeds.
4. Add a file-input fallback and a label download only if the folder picker still blocks the work. Download mode cannot replace the original file by itself.
5. Do not add inference, bulk edit, a backend, a framework, a theme, or Roboflow.

Do items 1 and 2 first.
The labeler already has these control ids: `open`, `files`, `save`, `objects`, `add`, `remove`, `mode`, `missingLeft`, `missingRight`, `canvas`.
The labeler does not have a label-text box, undo, or previous / next.

Use a text control to enter an exact label row.
Do not use a canvas drag for that entry.

A cloud session can have no desktop browser.
A cloud session can also block a host that is outside the egress allow-list.
This checkout has no egress error log.
If a session cannot open the labeler, record the exact error.
If a session cannot install a package, record the exact error.
Do not change gold labels to avoid that limit.

## 3. Current state

### 3.1 Repository

The remote is `https://github.com/Siuuuers/dl`.
Branch `main` in this checkout is commit `601bfc31cad10b4eb5898b850f965467563efda9` (`601bfc3`).
The work tree of that commit was clean in this checkout before this document.
Recent commits on that history:

- `601bfc3` adds the exporter.
- `4798748` adds the labeler.
- `2c18773` adds `notebooks/cassette_colab.ipynb`.
- `351eb3c` adds the epoch 173 training data and validation data.

### 3.2 Labeler

The labeler edits one box and two endpoints for each cassette.
Save writes this row:

```text
0 cx cy w h left_x left_y left_v right_x right_y right_v
```

The numbers are normalized to the image.
`v=2` means that the endpoint is visible.
**Left missing** and **Right missing** write `0 0 0`.
An empty cassette list writes an empty file.
Roboflow is not part of the labeler.

### 3.3 Colab notebook

`notebooks/cassette_colab.ipynb` can run GPU inference after a clone.
The inference cell uses `weights/cassette_lower_epoch173.pt`.
It sets `imgsz=1280`, `conf=0.25`, and `device=0`.
It sets `save=True`.
It shows an image preview under `runs/colab_predict`.
It does not set `save_txt`.
It does not write draft labels.
This checkout has no earlier Colab `.txt` artifact to correct.

The notebook also has a smoke training cell.
That cell calls `model.train` with `epochs=3`, `imgsz=1280`, `batch=4`, and `device=0`.
It writes under `runs/colab_train` with the name `smoke`.
The cell does not set the optimizer, the learning rate, `rect`, `amp`, or `seed`.

The install cell runs `%pip -q install -U ultralytics`.
That command installs the latest Ultralytics build on Colab.
`requirements.lock.txt` pins Ultralytics `8.4.182`.
`metadata/model_card.json` records the same pin: `ultralytics 8.4.182`.
The notebook tip says that you can set `imgsz` to 640 for a faster pipeline check.
Do not use 640 as the quality baseline.

### 3.4 Gold data in this checkout

`data/data.yaml` has one class, `cassette_front`.
It sets `kpt_shape: [2, 3]` and `flip_idx: [1, 0]`.

These gold counts were counted again from `data/labels/` in this checkout.
They match `data/README.md`.

| Split | Images | Label files | Objects | Empty label files |
| --- | ---: | ---: | ---: | ---: |
| train | 137 | 137 | 3772 | 13 |
| val | 20 | 20 | 664 | 3 |

Images and the weight are Git LFS objects.
In this checkout, `data/images/train/new_000020.png` is an LFS pointer (131 bytes).
It is not a PNG until you run `git lfs pull`.
`weights/cassette_lower_epoch173.pt` is also an LFS pointer in this checkout.
The pointer records SHA-256 `e0aab1b377491700318e0b14cd4c558ec3c1684087a13b7dde2298ebdbf2847e`.
The pointer records size 106,838,542 bytes.
Run `git lfs pull` before you open images in the labeler or load the weight.

Keep the gold labels and the weight as the baseline.

### 3.5 Exporter on GitHub, and the local edit

The exporter on GitHub has no command-line arguments.
The constants are `SPLITS = ('val',)`, `CONF = 0.25`, and `IMGSZ = 1280`.
The predict call sets `device='cpu'`.
For each endpoint, confidence below `0.01` becomes `0 0 0`.
A coordinate outside 0 to 1 also becomes `0 0 0`.
Other endpoints are written with visibility `2`.

The session record describes a later exporter edit.
That edit keeps labels in the splits that you did not request.
That edit still replaces labels in the split that you did request.
That edit is not in commit `601bfc3`.
This checkout had no uncommitted diff for `scripts/export_pose_drafts.py`.
Treat the split-aware exporter edit as local-only until it is in Git.

The exporter on GitHub deletes all of `data_pred/` when you run it.
Copy reviewed draft labels to another folder before you run it.

### 3.6 Recorded training baseline

The recorded training run is not the notebook smoke test.
The sources are `metadata/training_config.json`, `weights/base.json`, `metadata/model_card.json`, and `README.md`.

| Setting | Recorded training | Colab smoke cell |
| --- | --- | --- |
| Start weights | Official COCO YOLO26s-Pose | `cassette_lower_epoch173.pt` |
| Epochs | 200, then extended to 300 | 3 |
| `imgsz` / batch | 1280 / 8 | 1280 / 4 |
| Optimizer | AdamW, `lr0=0.001`, `lrf=0.01` | not set in the cell |
| Other | `rect=true`, `amp=false`, `seed=42` | not set in the cell |

Augmentation in `metadata/training_config.json`:

- `mosaic=0.0`, `mixup=0.0`, `cutmix=0.0`, `copy_paste=0.0`
- `degrees=5.0`, `translate=0.05`, `scale=0.15`, `fliplr=0.5`
- `hsv_h=0.01`, `hsv_s=0.2`, `hsv_v=0.2`

Epoch 173 has the minimum sum of `val/*_loss` over the 300 epochs.
The validation total loss is `1.79037`.
`README.md` gives these epoch 173 validation metrics:

| Metric | Box | Pose |
| --- | ---: | ---: |
| Precision | 0.96670 | 0.98297 |
| Recall | 0.91810 | 0.93675 |
| mAP50 | 0.93764 | 0.95993 |
| mAP50-95 | 0.58654 | 0.95963 |

These values are validation metrics for model selection.
They are not an independent test score.
A high pose mAP50-95 does not give the corner error in pixels.

The live viewer defaults are in `weights/base.json` and `scripts/view_d405_yolo.py`:

- `imgsz=1280`
- `rect=true`
- `conf=0.25`
- `iou=0.7`
- `endpoint_conf=0.5`
- `nms=true`

The exporter does not set `nms`, `iou`, or `rect`.
The notebook inference cell does not set `nms`, `iou`, or `rect`.
The live viewer sets them.

### 3.7 Pending validation review

The session record leaves 17 validation images pending.
Section 2.2 names the three corrected validation images.
`data_pred/` is not on GitHub.
A clean clone does not contain the pending drafts.

## 4. Next steps

Do these steps in order.
Keep the gold labels unchanged.
Keep `weights/cassette_lower_epoch173.pt` unchanged.

1. Finish the review of the 17 pending validation images.
   Review the 12 training files in section 2.2.
   Accept or reject each difference.
   Save the candidate in a named folder.
   Record the date and the source commit in a small text file in that folder.
   Use paths relative to the repository.
   Do not store absolute machine paths.
   Do not train on validation drafts.

2. Run inference with epoch 173 at `imgsz=1280`.
   Use the same validation images for three box confidence values: `0.15`, `0.25`, and `0.40`.
   Set `nms=true`, `iou=0.7`, and `rect=true`, as in the live viewer.
   Count missed cassettes.
   Count duplicate boxes and background boxes.
   Tune the endpoint confidence separately from `0.50`.
   If duplicate boxes remain, compare NMS IoU `0.5` with `0.7`.
   Check that adjacent cassettes remain.
   For visible endpoints, record the pixel error at the original image size.
   Record the median error and the 95th percentile error.

3. Run two fine-tunes of 30 epochs from epoch 173.
   Set `resume=false`.
   Use AdamW, `lr0=0.0001`, `imgsz=1280`, batch `4`, and seed `42`.
   Keep the augmentation values from `metadata/training_config.json`.
   Keep the loss values from `metadata/training_config.json` (`pose=12.0`, `kobj=1.0`, `box=7.5`, `cls=0.5`, `dfl=1.5`).
   Train one run on the gold labels.
   Train the other run on the candidate training labels.
   Keep the 137 / 20 split.
   Keep the empty scenes.
   Use the same validation labels for both runs.
   Compare both runs with the unchanged epoch 173 checkpoint.
   Start with all layers trainable.
   Try `freeze=10` only if the validation result gets worse while the training loss falls.
   Keep a new checkpoint only when the same validation check is better than epoch 173.

4. Keep `imgsz=1280` while corner error matters.
   Test `imgsz=960` only when you must measure runtime.
   Measure endpoint error and latency in the same test.
   Keep the augmentation unchanged in the first tests.
   If the remaining failures follow illumination changes, test `hsv_v=0.3` in place of `hsv_v=0.2`.
   Do not change another augmentation value in that test.

5. The labeler controls in section 2.3 are separate from the model test.
   Add the label-text editor and one undo step only when a person asks for that HTML change.
   Do not add the other labeler ideas in the same change.

For each new checkpoint, record these signals:

- Box precision, box recall, and box mAP50-95
- Pose precision, pose recall, and pose mAP50-95
- Validation loss
- Duplicate boxes
- Missed slivers
- Median endpoint pixel error and 95th percentile endpoint pixel error at the original resolution

Save periodic checkpoints and `best.pt`.
Epoch 173 already has the minimum summed validation loss in the completed 300-epoch run.
More epochs after 173, with no other change, are not a new test.

## 5. Handoff notes

### 5.1 Files on GitHub at `601bfc3`

- `tools/cassette_labeler/index.html` and `tools/cassette_labeler/README.md`
- `notebooks/cassette_colab.ipynb`
- `scripts/export_pose_drafts.py` (val only; it deletes `data_pred/`; it has no CLI)
- `scripts/view_d405_yolo.py` and `start_live.sh`
- Gold labels in `data/labels/`, plus `data/data.yaml`, `data/README.md`, and `data/manifest.json`
- `data/images/` and `weights/cassette_lower_epoch173.pt` as Git LFS objects
- `metadata/training_config.json`, `metadata/model_card.json`, and `weights/base.json`
- `models/cassette_lower_yolo26s_pose.yaml`

### 5.2 Local only, until a later commit

- The `data_pred/` tree, including reviewed corrections
- An uncommitted exporter CLI edit or split-aware edit, if it exists only on the review machine
- `labeler_upgrade_ideas.jsonl`
- A candidate folder that exists only on the review machine
- Colab output under `runs/`

`.gitignore` does not list `data_pred/`.
A later `git add` can stage `data_pred/` by mistake.
Do not commit large binary files from a review folder.
Do not commit image copies.
Do not commit weight files from a review folder.
Do not commit an LFS pointer as if the pointer were a PNG.

### 5.3 How to continue

1. Clone `https://github.com/Siuuuers/dl`.
2. Use commit `601bfc31cad10b4eb5898b850f965467563efda9`, or a later commit on `main` that still contains this handoff.
3. Run `git lfs install`.
4. Run `git lfs pull`, so each PNG and the `.pt` weight is a real file.
5. Copy the local `data_pred/` tree from the review machine if you must finish the 17 validation reviews.
6. Open the labeler on that copy.
7. Write corrected labels with **Save label**.
8. Copy the reviewed `.txt` files into the candidate folder before you run the exporter again.
9. Keep `data/labels/` as the gold baseline.
10. Do the confidence test and the two 30-epoch runs in section 4.
11. For that comparison, use Ultralytics `8.4.182`.
    Do not use an unpinned Colab install for the comparison.

The exporter on GitHub writes absolute image links under `data_pred/images/`.
Replace those links with relative links before you share the folder.
You can also omit the image links and open `data/images/` in the labeler.
