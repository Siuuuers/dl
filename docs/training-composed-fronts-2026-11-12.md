# One comparison for the composed front annotations

**The input data and workflow are prepared for one matched GPU comparison. Full training has not been executed here. Keep PR35 selected until the returned models are evaluated.**

This experiment holds the current ten endpoint-slot corrections fixed and tests the additional **17 box corrections and one duplicate removal**. The original training images have now been restored and verified. No new camera captures are required for this comparison.

## Run it

1. Open [baomai_composed_fronts.ipynb](../notebooks/baomai_composed_fronts.ipynb) in [Google Colab](https://colab.research.google.com/github/Siuuuers/dl/blob/codex/composed-training-20261112/notebooks/baomai_composed_fronts.ipynb).
2. Select **Runtime → Change runtime type → GPU**, then **Runtime → Run all**. Keep the same GPU/runtime for both arms. The notebook asks to mount Drive so it can preserve completed work.
3. Return **`baomai-composed-fronts-seed46-results.zip`**. Keep its verified Drive backup until evaluation is complete.

The final notebook pins implementation commit [`082ea773`](https://github.com/Siuuuers/dl/commit/082ea773426b6422f859c68b8b6ec8496ef07df0) and protocol SHA-256 `2e73d4d81585e8ed1759bdb02aea98d8c9db3a856f8887ef9f41069eb799fa3f`. It refuses unset pins. It restores the original images and initialization weights, verifies both label sets, performs a four-batch resource trial when a full run is still needed, then completes the two 30-epoch runs. Each full run starts independently from original epoch173 weights. The trial and PR35 weights never initialize either arm.

Use the delivered notebook rather than an earlier unpinned preparation snapshot. Its dedicated backup folder is `MyDrive/baomai-composed-fronts-seed46-20261112`. Compatible completed arms and their exact artifacts can be restored. Partial or incompatible runs are preserved and cannot count as complete; a retry needs a fresh `RUN_BASE`. If both compatible arms are already complete, the notebook skips GPU probing, the resource trial and training, then verifies the existing return ZIP or creates it from the preserved artifacts.

## What the two arms mean

Both runs use the same composed manifest and the same numerical recipe. Their actual runner arm determines which verified labels are used.

| Runner arm | Training supervision | Objects | Locatable endpoints | Missing slots |
| --- | --- | ---: | ---: | ---: |
| `reference` | Exact current ten-slot endpoint candidate | 3,771 | 6,842 | 700 |
| `candidate` | That control plus 17 boxes and one duplicate removal | 3,770 | 6,842 | 698 |

Both retain **137 training images and 13 empty training images**. In this experiment, `reference` means the endpoint-corrected control. It is not the original uncorrected annotation set. Completed receipts and checkpoint identities explicitly name `endpoint_control` or `composed_treatment` so that a generic arm name cannot conceal different supervision.

The [complete composed candidate](../data/candidates/composed-fronts-2026-11-12/README.md) contains eleven overlay files. Eight are the unchanged PR35 parent overrides. Its three composed files cover `new_000023`, `new_000031` and `new_000045`. The control has ten overlay files. Their effective label inventories differ only in those three images.

The removed object is **original `new_000023` row 22**, a recorded duplicate with both endpoints missing. Original row 36 is retained. Deletion shifts original row 33 to treatment row 32; its accepted right corner `(612,549)` remains intact. All ten endpoint-slot corrections are preserved by original object identity. No geometric estimate or new endpoint correction enters this comparison.

The PR49 box/deletion changes are disjoint from the fourteen-box PR47 experiment. PR52 tested endpoint supervision alone. Neither earlier result answers this incremental box/deletion question. The [frozen composition proposal](../results/review/next-training-20261112/combined-operations-spec.json) and [review record](../data/candidates/composed-fronts-2026-11-12/review.json) identify the exact source rows and evidence.

## Fixed training and evaluation

Recipe ID: `reviewed-composed-seed46-v1`. Its complete numerical settings equal the completed PR52 endpoint recipe: **seed 46, 30 epochs, batch 4, input 1280, rectangular batches, FP32, AdamW, learning rate 0.0001 and bias warmup 0.0001**, with the recorded augmentation and loss settings. Ultralytics remains pinned to **8.4.182**. Select finalized epoch 30 `last.pt`; there is no best-epoch or seed search.

The numerical recipe SHA-256 is `f18e0a813a1057e1b3ab3a44ea3ce9b2a3e7673cc2bf04ca09b394bfdd8c51b8`. The [prospective protocol](../experiments/composed-fronts-2026-11-12.json) fixes both roles, the candidate/control identities, initialization, operation evidence, evaluation settings and decision criteria before training.

The original in-training validation labels remain unchanged: **20 images, 664 objects, 1,227 locatable endpoints, 101 missing slots and three empty images**. The [adjudicated development reference](../data/references/visible-front-adjudication-2026-11-12/README.md) is separate and supplies post-training denominators of **663 objects, 1,227 locatable endpoints and 584 two-corner objects**. Do not copy the revised development labels into either training arm.

Evaluate both returned checkpoints and the exact retained PR35 checkpoint in one inference environment at **box confidence 0.13, NMS IoU 0.70, matching IoU 0.50, endpoint confidence 0.50 and input size 1280**. Report all-object misses/extras, recall by zero/one/two-locatable-corner groups, full-denominator 5/10 px pair success, locatable coverage, miss-free scenes, empty-scene outputs and changed object/success identities. Report the preceding reference separately as the declared sensitivity comparison; do not silently overwrite older scores.

The primary question is whether the new box/deletion supervision produces supported improvement in completeness or fewer duplicate/false detections while preserving existing supported successes. Inspect originals to distinguish actual recovery from an existing proposal crossing an IoU threshold or being assigned differently. Compare the effect against the matched endpoint control first, then assess replacement of PR35 separately. No automatic promotion, threshold sweep, seed sweep or additional unlabeled-data claim is part of this experiment.

## What is verified already

The [preparation evidence](../results/verification/baomai-composed-preparation-20261112/README.md) distinguishes actual-data checks from synthetic execution tests:

- All **157 original PNGs** are restored, hash-verified and checked as 1280×720 images. Original epoch173 bytes match the model card. These files total **247,969,310 bytes**, including the checkpoint.
- The composed verifier reconstructs the exact control and treatment from immutable source labels, historical accepted edits and the frozen operation evidence. It checks row identity, complete file inventories, counts, geometry, missing-point representation and unchanged original validation.
- Both complete datasets were actually copied and independently reverified. All ten control overlay files match the existing endpoint candidate byte-for-byte.
- Focused tests check altered/rehashed labels and evidence, wrong row maps, arm swaps, control identity, completed-receipt identity, notebook recovery and returned archive integrity. They do not simulate a successful physical model.
- The notebook's exact embedded input preflight ran against the restored originals and the actual protocol, including the separate adjudicated evaluation reference.

No GPU resource trial, full training, new inference or measured model improvement is claimed by these preparation checks. This environment has no available `nvidia-smi`; the notebook performs the real CUDA/device checks in the selected GPU runtime before starting missing work.

## Historical rules remain strict

The current annotation policy appends a section separating context-derived proposals from observed labels. Its older core is byte-identical to the policy pinned by the PR35 and endpoint candidates. The new verifier keeps both identities explicit: it validates the old candidates inside a temporary snapshot containing their exact historical policy, then validates the composed candidate under the live current policy.

The live policy, historical manifests, accepted endpoint labels and existing strict validators are unchanged. The historical policy snapshot is [recorded separately](policies/annotation-policy-pr35-2026-11-10.md). The ordinary validator still rejects undeclared changes to original visible endpoint tokens. The composed kind adds explicit permission for this exact recorded composition; it does not relax that ordinary contract.

## Command-line use

After checking out the implementation revision pinned by the notebook and restoring original LFS assets, verify and stage into a fresh directory:

```bash
python -B scripts/verify_pose_composed_candidate.py
python -B scripts/stage_pose_training.py \
  --candidate data/candidates/composed-fronts-2026-11-12/manifest.json \
  --output /path/to/fresh-composed-staging
python -B scripts/stage_pose_training.py \
  --candidate data/candidates/composed-fronts-2026-11-12/manifest.json \
  --verify /path/to/fresh-composed-staging
```

The notebook supplies resource limits, runtime matching, recovery and the complete training commands. After both full runs finish, its return step runs:

```bash
python -B scripts/package_composed_results.py \
  --protocol experiments/composed-fronts-2026-11-12.json \
  --reference /path/to/reference \
  --candidate /path/to/candidate \
  --output /path/to/baomai-composed-fronts-seed46-results.zip
```

Add `--verify-existing` to recheck an existing ZIP without rewriting it. The archive contains the two complete six-file checkpoint bundles, the frozen protocol and an index: **14 files**. It verifies actual arm/label identities, identical numerical settings, source and runtime, original initialization, the full staged file inventories and every returned payload hash. Packaging establishes eligible completed artifacts, not accuracy or independent proof that a resource trial ran.
