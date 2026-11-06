# Cassette pose handoff

Source snapshot: `main` at `b4df76935178de5a122119aea48e85d2d661effc` (6 November 2026).
This handoff separates committed files from earlier session reports.

## 1. Confirmed scope and reference data

Use the `baomai` task: one `cassette_front` class and two front-lower corners.
The corners belong to the white wax block's front face.
Order them by image X as `front_lower_left` and `front_lower_right`.
Do not use the upper lip, tray, or basket corners.
The older `yolo-pose` branch has a different annotation schema.

Each instance has one box and two points in an 11-number row:

```text
0 cx cy width height left_x left_y left_v right_x right_y right_v
```

Normalize coordinates against the original 1280 × 720 image.
Use `v=2` for a visible point; use `0 0 0` when a point cannot be located.
Keep the other visible point when only one end can be located.
An empty scene has an empty label file.
The 1280 × 750 annotated previews include a footer; use originals for training and editing.

The committed reference snapshot is documented in [data/README.md](../data/README.md):

| Split | Images / label files | Objects | Empty scenes |
| --- | ---: | ---: | ---: |
| train | 137 | 3,772 | 13 |
| val | 20 | 664 | 3 |

Keep `data/labels/` and `weights/cassette_lower_epoch173.pt` as the original baseline.
The reference `data/labels/val/new_000007.txt` has 54 instances; it is an existing corrected example.
This file does not establish that the later session's correction has been recovered.
Validation helped select epoch 173; its scores are not independent final-test results.
The five final-test images are excluded from this package.

## 2. Current editor and exporter

[PR #3](https://github.com/Siuuuers/dl/pull/3), [PR #4](https://github.com/Siuuuers/dl/pull/4), and [PR #5](https://github.com/Siuuuers/dl/pull/5) are merged.
The browser editor has Label text, Apply, Reset text, one-step Undo, and Save & Next.
It also has Edit selected row and Show annotations.
Hiding annotations makes the canvas view-only; it preserves labels and pending text.
Save & Next advances only after the label write closes successfully.
Use a working copy and follow the [editor guide](../tools/cassette_labeler/README.md).
Save changes `.txt` labels only; it does not rebuild previews or snapshot hashes.
The editor does not run inference or training.

`scripts/export_pose_drafts.py` exports validation drafts from epoch 173 on CPU.
It uses `imgsz=1280` and `conf=0.25`, with no command-line split selector.
It stops if `data_pred/` exists; the older description that it deletes this folder is obsolete.
Its image links use absolute paths and need replacement when moving to another machine.
Its inference call does not explicitly set the live viewer's `nms`, `iou`, and `rect` options.
Use the baseline evaluation settings consistently before comparing predictions.

## 3. Earlier session reports: files still to recover

The original PR #2 handoff described source commit `601bfc31cad10b4eb5898b850f965467563efda9`.
It recorded the following claims from a correction session:

- Initial drafts: 688 validation objects and 3,989 training objects.
- Validation corrections: `new_000007`, `new_000008`, and `new_000146`; 17 images pending.
- Training candidate: 3,765 objects after 12 correction batches.
- Training comparison: 125 files identical to the reference and 12 files different.

The 12 named training files comprise **seven `new_*` and five `old_*` files**:

```text
new_000013  new_000015  new_000019  new_000023
new_000075  new_000083  new_000094
old_000003  old_000004  old_000007  old_000010  old_000013
```

The reviewed source tree contains no `data_pred/`, candidate labels, or correction log.
Their location is unresolved; absence from this tree does not mean they are lost.
The reported 3,765 count and review completion have not been verified from those files.
Object-count differences alone do not measure missed or duplicate detections.
Treat the 17-image remainder as a historical status until the review records are recovered.

## 4. Recover a portable candidate

Recover the earlier `.txt` files and any review log before repeating corrections.
Inventory image stems, split names, file hashes, and review status against the original snapshot.
Keep the recovered labels in a separate candidate directory.
Copy small text labels directly; do not rely on absolute image symlinks surviving the move.
Use matching original PNGs on the destination machine and rebuild local links as needed.
Download actual Git LFS image bytes before opening them; a pointer file is not a PNG.
Record the source commit, reference label hashes, review date, and reasons for accepted changes.
Small reviewed label sets and their provenance can be versioned without duplicating images or weights.
If the earlier files cannot be located, record that result and create a new draft set in a fresh folder.

## 5. Next actions, in order

1. Run unchanged epoch 173 against the existing 20-image reference validation set first.
   Save the source revision, settings, predictions, matching rules, and error report.
   Complete this baseline before continuing the historically pending 17 validation reviews.
2. Use the baseline failures and recovered review records to prioritize label inspection.
   Review the 12 named training differences and accept or reject each change against the original image.
   Check missing objects, duplicate boxes, narrow visible faces, and actual lower corners.
3. Keep proposed corrections separate from the reference labels while they are reviewed.
   If a reference error is confirmed, publish a distinct, approved annotation version.
   Reevaluate epoch 173 and every model candidate on that same validation version.
4. Train only on the training split after the review supports a useful comparison.
   Never train on validation drafts or validation corrections.
   Preserve empty scenes and retain the unchanged epoch-173 checkpoint for comparison.

The evaluation and model-comparison plan belongs in [PR #1](https://github.com/Siuuuers/dl/pull/1), at `docs/refine-cassette-pose.md`.
Read that file on the PR branch until it is merged; it is not in the source snapshot above.

## 6. Verification and limits

The editor check was rerun on the source snapshot above: `node tools/cassette_labeler/check.mjs` exited 0 with 18/18 scenarios passed.
The exporter check, `python3 scripts/check_export_pose_drafts.py`, also exited 0; it uses a fake model.
The editor checks substitute DOM, canvas, and file handles; they do not establish native browser acceptance.
A scratch probe loaded the real 54-row example, targeted row 37, and round-tripped an unchanged save.
The probe also confirmed that typed reversed endpoints and nonzero coordinates on a missing endpoint can be saved.
Validate image-X point order and canonical `0 0 0` missing points before accepting corrections for training.
A real browser open → edit → Apply → Save → reopen walkthrough remains pending.
In this review environment, the browser could not reach the local server and its security policy rejected local-file URLs.
Those access failures do not establish an HTML defect; do not bypass the browser policy.
This handoff adds no inference result, corrected label set, training run, or physical-accuracy claim.
For the reference model and dataset provenance, see [model_card.json](../metadata/model_card.json) and [data/manifest.json](../data/manifest.json).
