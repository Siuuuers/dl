# R35 annotation follow-up and GPU comparison

R35 continues the accepted R34 annotation snapshot after the PR117 returned run. Nine rows are amended and two development-only rows are deleted. Five box-only training corrections across three photographs form the next controlled training change. The focused review records each decision against its preserved R34 row identity. The full review, eligible training overlay and development scoring references remain separate.

## Review and source

- Full review: `data/reviews/fronts-2026-11-25-r35/`.
- Eligible training overlay: `data/candidates/front-review-2026-11-25-r35/`.
- Accepted decisions, exact operations and row mapping: `results/review/front-review-20261125-r35/`.
- Fixed experiment: `experiments/front-review-2026-11-25-r35.json`.
- Notebook: `notebooks/baomai_front_review_r35.ipynb`.

The review retains original R34 row identities even when the development snapshot deletes `new_000009` rows 56 and 57. These are the final two rows of that image's R34 label file, so surviving TXT rows 1-55 retain their positions. The saved row map marks the two original identities as deleted; an old row number must not silently become a different physical object. The exact before/after lines and original-image hashes identify every operation.

Development-only boundary corrections and those two object removals change the revised review snapshot. They do not alter either training arm or the original trainer-validation labels. A lower miss or unmatched count caused by deleting a reference object is a reference change, not a model improvement.

## Controlled training question

Does the accepted R35 training-annotation delta improve visible-front detection and actual lower-front corner localization under the unchanged native loss?

The reference uses eligible R34 supervision; the candidate uses eligible R35 supervision. Only the verified training rows in `training-input-delta.json` differ. Training object, class and corner-availability inventories are unchanged. No hidden corner is introduced. Held photograph `new_000015` and the 31 excluded old-series photographs retain identical prior supervision in both arms.

Both arms keep all 137 training images, 3,784 objects and 13 empty scenes. They start independently from the original epoch-173 initializer and use seed 46, batch 4, input size 1280, FP32, AdamW and the unmodified native Pose26 loss for 30 epochs. Each arm receives 1,050 batches, 4,110 image presentations and 78 optimizer updates. The four-batch resource trial is discarded; the final epoch supplies the returned checkpoint. Original trainer validation remains 20 images with 664 objects.

## Run and return

Open the published notebook in Colab, choose an L4 or A100 GPU, and select Run all. It verifies the source/protocol pins, the original image bytes and initializer, runs the resource trial and matched pair, then produces `baomai-front-review-r35-results.zip`. Compatible completed runs can be recovered from Drive without retraining; incompatible or incomplete folders are preserved.

The ZIP contains both final checkpoints, logs, receipts, protocol and exact training delta. It does not contain the original photographs or initializer.

Publish the source and accepted annotation files before generating the notebook:

```bash
python3 -B scripts/build_front_review_r35_notebook.py --source-revision FULL_PUBLISHED_SOURCE_SHA
python3 -B scripts/check_front_review_r35_notebook.py
```

The notebook fetches the immutable published commit and contains no embedded repository payload. Its setup has one source-loader definition. The integration check executes both complete setup cells in order against a synthetic Git remote, including actual source checkout and original-input hash verification. It also checks clean reuse and rejection of dirty or incompatible source.

## Evaluation

Keep box confidence 0.13, endpoint confidence 0.50, input size 1280, one-to-many output, NMS IoU 0.70 and matching IoU 0.50. Frozen 666 and historical 663 remain unchanged. Report preserved R34 and revised R35 development diagnostics separately, with the deleted-object mapping explicit.

Prioritize missed visible fronts, including fronts with zero or one locatable corner. Report unmatched outputs, their duplicate subset, other unmatched outputs, empty-scene outputs and both-corner successes at 5/10 original pixels together. Missed eligible objects remain in the success denominators. Inspect box association before describing an IoU mismatch as a completely absent prediction.

PR35 remains selected. This notebook prepares a new annotation comparison; it does not claim an R35 training result or select a new model.
