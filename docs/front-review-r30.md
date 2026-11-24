# R30 accepted annotation review and partial-target loss experiment

Based on the accepted R28 review and the returned R29 sampling run in [PR #107](https://github.com/Siuuuers/dl/pull/107).

## What is published

- `data/reviews/fronts-2026-11-24-r30/`: the R30 157-image review label snapshot, preserving the 137/20 split.
- `data/candidates/front-review-2026-11-24-r30/`: the eligible 92-file training overlay, including the three amended training rows.
- `results/review/front-review-20261124-r30/row-review.json`: 29 amended and 12 retained review decisions, with exact lines and reasons.
- `experiments/partial-target-loss-2026-11-24-r30.json`: the paired, fixed R30 experiment.
- `scripts/partial_target_loss.py`, `scripts/train_pose_partial_loss.py`, and supporting trainer/preflight code.
- `notebooks/baomai_partial_target_loss_r30.ipynb`: the exact validated standalone notebook. It reconstructs its own pinned local R30 source from the published R28 base, so its input verification does not depend on the branch's merge status.

## Run in Colab

Open `notebooks/baomai_partial_target_loss_r30.ipynb` from the pinned GitHub commit, select an L4 or A100 GPU, then **Run all**. Return `baomai-partial-target-loss-r30-results.zip`.

Both roles use identical accepted eligible R30 labels, the original epoch-173 checkpoint, seed 46, 30 epochs, batch 4, image size 1280, FP32, AdamW and ordinary image sampling. The sole planned factor is assigned-partial-target detection-loss weight 1 (control) versus 2 (candidate). Assignment counters are recorded. The notebook includes a disposable resource trial and backup validation.

**No R30 GPU comparison has been run yet.** PR35 remains the selected model. Frozen 666 is the primary selection reference; frozen 663 and the current 661-front review must remain separate.

The two cropped R30 PDFs were delivered to the conversation; they are not required for Colab and are intentionally not added to the repository.
