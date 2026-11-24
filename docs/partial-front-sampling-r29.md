# R29: Partial-front sampling, fixed training budget

**Status:** Notebook and reproducible implementation ready; full GPU comparison not yet run. **Do not promote a model from this PR.**

## Scope

This follows [PR #105](https://github.com/Siuuuers/dl/pull/105). It preserves all accepted eligible R28 training labels, all 137 training images, the original trainer validation split (20 images, 664 annotated fronts), and the frozen 666-front primary / 663-front historical model-selection references. The accepted R28 661-front review is a separate exploratory reference. The retained PR35 model stays selected.

**Hypothesis:** More exposure to previously annotated, partial-front-rich training images may improve detection of cassette fronts with zero or one physically locatable lower-front corner, without extra training images or optimizer updates.

## Controlled comparison

| Setting | Control | Candidate |
|---|---|---|
| Labels | Accepted eligible R28 | The same accepted eligible R28 |
| Initial weights | Independent epoch-173 start | Independent epoch-173 start |
| Sampling | Each original training image once/epoch | Nine partial-rich photos twice/epoch, nine other presentations replaced |
| Image presentations | 137/epoch, 4,110 total | 137/epoch, 4,110 total |
| Batches | 1,050 | 1,050 |
| Planned optimizer updates | 78 | 78 |
| Epochs | 30 | 30 |

Both roles use seed 46, batch 4, image size 1280, FP32 (AMP off), AdamW with initial learning rate 0.0001, independent initialization, and predetermined final-epoch checkpoints. Training must occur under the same verified runtime fingerprint.

The nine selected images are `new_000012`, `new_000018`, `new_000023`, `new_000031`, `new_000044`, `new_000045`, `new_000094`, `new_000116`, and `new_000128`. Selection is from **training supervision only**: at least one zero-corner front, at least eight zero-/one-corner fronts, and at least one third of the image's fronts in that partial category. There are 83 eligible non-cohort omission donors and 45 protected photographs; protected images retain their ordinary exposure. The omission schedule rotates across the 83 images with a private seed-46 RNG. Empty training images are protected.

The selected images each appear 60 times in 30 epochs; protected images 30 times; eligible omitted-donor images 26 or 27 times. Whole-photo sampling also changes exposure to ordinary fronts and backgrounds: the comparison must inspect regressions, not merely wins on familiar misses.

## Files

- [Standalone Colab notebook](../notebooks/baomai_partial_front_sampling_r29.ipynb)
- [Fixed experiment protocol](../experiments/partial-front-sampling-2026-11-24-r29.json)
- `scripts/train_pose_sampling.py`: sampled-image dataset/schedule support
- `scripts/partial_front_sampling.py`: inventory and experiment verification
- `scripts/train_pose_comparison.py`: training path with sampling integration
- [Source-verified partial-front inventory](../results/review/partial-front-sampling-20261124/r28-training-partial-inventory.json)
- [Nine-photo cohort and crop geometry decisions](../results/review/partial-front-sampling-20261124/proposed-cohort.json)
- [PR105 failure interpretation](pr105-next-step.md)

The notebook is independently executable: it reconstructs its exact source tree from published R28 commit `c487e7abbb0aca19f59fb143ca2320d3adb01b3e`, verifies a deterministic **local source commit** `eea8e52a983f162512c4aa34ced73761b089609b`, and does not need this PR to be merged to run. That deterministic local assembly SHA is intentionally distinct from the GitHub publication commits in this PR.

## Running

Open `notebooks/baomai_partial_front_sampling_r29.ipynb` in Colab. Select an **L4 or A100 GPU**, authorize Drive for recovery, choose **Run all**, and retain `baomai-partial-front-sampling-r29-results.zip`. The resource trial checks four actual batches; each role then trains from the original initializer. Completed model bundles and the result ZIP are hash-checked before reuse. Do not infer GPU resource availability from the CPU preflight.

## Evaluation after GPU results return

Infer the returned pair together using the same evaluation runtime. For each unchanged reference, report zero-, one-, and two-corner target recall; gained/lost target identities; five- and ten-pixel paired corner localization; box-overlap behavior; missed fronts; unmatched outputs split by duplicates and other outputs; and empty-scene behavior.

Use detector confidence 0.13, endpoint confidence 0.50, matching IoU 0.50 and the fixed NMS setting in the experiment. Keep the accepted R28 reference, frozen 666 primary, and 663 historical reports **separate**. Reference rescoring is not model learning. This experiment does not establish new-capture-session reliability or robot-grasp accuracy.
