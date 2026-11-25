# R34 annotation follow-up and next GPU notebook

R34 applies 16 accepted follow-up corrections across 14 photographs to the preserved R33 annotation snapshot: seven training rows across six photographs and nine development rows across eight photographs. One additional reviewed row is retained. The review covers the user-named rows and prediction cues, with original-image evidence and explicit before/after coordinates. It does not repeat or enlarge the previous whole-photo review claim.

## Source and review records

- Full review: `data/reviews/fronts-2026-11-25-r34/`.
- Eligible training overlay: `data/candidates/front-review-2026-11-25-r34/`.
- Exact operations, provenance and training delta: `results/review/front-review-20261125-r34/`.
- Fixed experiment: `experiments/front-review-2026-11-25-r34.json`.
- Notebook: `notebooks/baomai_front_review_r34.ipynb`.

The annotation box encloses owned visible frontal material, including a visible fragment of that same face seen through a gripper opening. Internal occluder pixels may therefore lie inside one rectangular box. Rack, gripper, neighboring fronts and background alone do not extend the bounds. Corner points still require the actual observed lower-front turn; a gripper opening edge is not a cassette corner.

Original image coordinates have positive Y downward. Physical top is stored `ymin`; physical bottom is stored `ymax`. Prediction identifiers such as `p54` refer to the specific preserved output stream in the cited review, not a stable label-row identifier.

## Controlled training change

The reference arm uses accepted eligible R33 labels; the candidate uses accepted eligible R34 labels. All 137 training photographs remain. Held photograph `new_000015` and 31 excluded old-series photographs retain exactly the same supervision in both arms. Original trainer validation remains 20 images with 664 objects; the reviewed development labels never replace it.

Only reviewed annotation differences enter the candidate. Class and object-row inventories remain fixed. The sole allowed visibility transition is an explicitly audited restoration of the actual left lower-front corner of `new_000066`, row 14, from unavailable to observed. It is not an inferred hidden point. The delta records whether that restoration is present and verifies the corresponding endpoint totals.

Both arms start independently from the original epoch-173 initializer and use the native Pose26 loss, seed 46, batch 4, image size 1280, FP32, AdamW and 30 epochs. The fixed budget remains 1,050 batches, 4,110 image presentations and 78 optimizer updates per arm. The four-batch resource trial is discarded. The final epoch supplies the checkpoint in each arm.

## Run and return

Open the published notebook in Colab, select an L4 or A100 GPU, and select Run all. The notebook restores and verifies the exact published source, original images and initializer. It preserves incompatible or incomplete folders; compatible completed results can be restored from Drive without repeating training.

Return `baomai-front-review-r34-results.zip`. The archive contains both final checkpoints, logs, receipts, protocol and exact training delta. Packaging verifies bytes and confirms both arms used the same runtime and budget.

The source is committed and published before the notebook is generated. Generate the notebook using:

```bash
python3 -B scripts/build_front_review_r34_notebook.py --source-revision FULL_PUBLISHED_COMMIT_SHA
```

The generator verifies that the chosen commit contains the current source, protocol, manifests and label bytes. The notebook fetches that immutable commit directly; it does not embed source files or reconstruct a historical local commit. PDFs, original photos and weights are not added to the review PR.

## Evaluation and decision

Keep box confidence 0.13, endpoint confidence 0.50, image size 1280, one-to-many output, NMS IoU 0.70 and matching IoU 0.50. Preserve frozen 666 and historical 663 as separate references. Keep preserved R33 and revised R34 development diagnostics separately identified; reference correction is not model improvement.

Prioritize missed visible fronts, especially the zero/one-locatable-corner groups. Report unmatched outputs and their duplicate subset, other unmatched outputs, empty-scene outputs and both-corner successes at 5/10 original pixels together. Keep eligible missed objects in success denominators. A poorly localized output can produce both an unmatched prediction and an unmatched reference, so inspect association and box overlap before calling it total detection failure.

PR35 remains selected. This preparation makes no R34 accuracy or promotion claim. The repeatedly reviewed development set cannot establish performance on a new camera session.
