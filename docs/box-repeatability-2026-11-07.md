# Do the reviewed-box gains repeat across training seeds?

**Keep epoch173 as the working model. This is the next prepared experiment, not a new accuracy result.** Run two additional matched pairs, at fixed seeds **43 and 44**, using the same original and corrected labels. The completed seed-42 experiment remains unchanged and is reported separately as historical context.

**Next action:** open the [repeatability notebook](https://colab.research.google.com/github/Siuuuers/dl/blob/codex/baomai-box-repeatability-20261107/notebooks/baomai_box_repeatability.ipynb), select a GPU runtime, and run all cells. It downloads the existing inputs, performs a bounded resource trial, trains four runs sequentially, and returns one `baomai-box-repeatability-results.zip`. Return that ZIP before closing the runtime. The immutable training source is `f087c35f74bf304eb61998edcaa2172fe1cba49c`.

## Why this is the next question

In the [completed seed-42 comparison](https://github.com/Siuuuers/dl/blob/5f2adb2a2d4aeecf87c29545406b50645e91ab5d/docs/box-supervision-comparison-2026-11-07.md), corrected boxes reduced misses from 37 to 36 and extras from 60 to 59 against the matched original-label reference. Conditional p95 corner error increased from 5.004297 to 5.096371 pixels. The increase persisted on the same returned endpoint identities, and the candidate did not pass its predeclared adoption rule.

The next question is whether the paired detection/corner changes have the same direction in other randomized training runs. This question can be tested with the existing images. It does not require a new architecture or a new collection of unreviewed labels.

The seeds are selected **before observing their results**, not searched for a favorable model. The [pinned Ultralytics trainer](https://github.com/ultralytics/ultralytics/blob/v8.4.182/ultralytics/engine/trainer.py) initializes training randomness from its seed setting. A common seed and runtime help control a paired comparison; they do not guarantee identical results across software releases or hardware. [PyTorch's reproducibility guidance](https://github.com/pytorch/pytorch/blob/v2.10.0/docs/source/notes/randomness.rst) documents that limitation.

## Fixed study contract

Study ID: `box-supervision-repeatability-v1`. The prospective seed set is exactly `[43, 44]`.

| Order | Seed | Arm | Labels | Fixed recipe ID |
| --- | ---: | --- | --- | --- |
| 1 | 43 | Reference | Original frozen labels | `box-supervision-warmup-seed43-v1` |
| 2 | 43 | Candidate | Same originals plus the 16 reviewed box edges | Same seed-43 recipe |
| 3 | 44 | Reference | Original frozen labels | `box-supervision-warmup-seed44-v1` |
| 4 | 44 | Candidate | Same reviewed box overlay | Same seed-44 recipe |

All four runs start independently from the exact original epoch173 weights. Each uses 30 epochs, batch 4, image size 1280, AdamW, nominal learning rate 0.0001, bias warmup learning rate 0.0001, AMP disabled and the existing augmentation/loss settings. The new numeric recipes differ from the old box recipe **only in seed**. The final epoch-30 `last.pt` is selected; neither the resource trial, another arm's checkpoint nor the trainer's `best.pt` initializes or selects a study run.

The same GPU session, source revision and environment must be used for all four runs. Within a seed, the reviewed label overlay is the only intended scientific difference. Across seeds, the only additional recipe difference is the seed itself. Output and data paths differ as necessary.

Each arm retains 137 training images, 3,772 objects and 6,839 visible endpoints. The overlay is the same `box-refinement-2026-11-07` proposal, SHA-256 `b61d012052001377089ad048ba13da14b8cd19d665ac8f0ef7828e45c530ec3e`. All object rows, classes, endpoint tokens and the 20-image validation reference remain fixed. The original review's unresolved annotation fields remain explicitly unresolved; the seed study does not certify them.

### Why seed 42 stays separate

The original seed-42 run is already complete in [PR #23](https://github.com/Siuuuers/dl/pull/23), evaluated in [PR #24](https://github.com/Siuuuers/dl/pull/24). Its training code predates the explicit seed recipes and its runtime is recorded separately. Do not rerun it merely to fill a new table, relabel it as seed 43 or 44, or silently pool it with the new matched runs.

The new summary covers seeds 43 and 44 with equal weight. The prior seed-42 result can appear in a clearly separate historical table. Numerical agreement with history is useful context, not evidence that the source/runtime was identical across studies.

## Run and recover simply

The notebook's resource trial uses the seed-43 candidate and stops after four real batches while preserving the full 30-epoch schedule. It must report optimizer updates and a parameter change. It does not produce an eligible checkpoint or guarantee full-run validation/finalization resource sufficiency.

The notebook preserves the installed CUDA Torch and torchvision versions while installing the pinned trainer, `ultralytics==8.4.182`. It prints the device, runtime and host-memory stop budget. The trial has a 600-second stop budget; each full run has a 21,600-second stop budget. These are limits, not duration promises. Do not silently change resolution, batch size, epochs or recipes to fit a resource failure.

Existing workspace, run and result paths are refused. There is no automatic checkpoint resume or overwrite. If one arm fails in an otherwise unchanged session, preserve its receipt/log and use the notebook's explicit fresh-folder retry instruction for that arm. Other completed runs may be retained only if the packager verifies the entire four-run set. If source, device, runtime or scientific settings change, start a fresh study. A missing or invalid run cannot be omitted to produce a successful partial study.

The return packager accepts explicit run paths, which also supports a fresh-folder retry:

```bash
python scripts/package_box_repeatability.py \
  --seed-43 /path/to/seed-43-reference /path/to/seed-43-candidate \
  --seed-44 /path/to/seed-44-reference /path/to/seed-44-candidate \
  --trial /path/to/resource-trial \
  --output /path/to/baomai-box-repeatability-results.zip
```

The ZIP preserves four finalized checkpoints and each run's original receipt, manifest, staging inventory, epoch CSV and log. An included resource trial remains supplemental, never an eligible model. Original images, initial weights and intermediate checkpoints do not need to be returned. The packager validates complete bundles, exact seed/recipe roles, the fixed candidate and shared source/runtime/non-seed settings before publishing a fresh archive, then reads its content back against the inventory.

## What the returned models will be compared on

Evaluate M0 once and all four returned models in one common evaluator runtime. Retain the same original images, labels, scorer and settings: image size 1280, rectangular preprocessing, FP32 inference, ordinary NMS IoU 0.70, batch 1, `max_det=300`, matching IoU 0.50, endpoint confidence 0.50, and box confidences 0.15/0.25/0.40. **The primary setting remains 0.25**; the others are diagnostic. The earlier optional NMS 0.68 experiment remains separate.

For each seed, report all five candidate-minus-reference changes:

1. Missed objects.
2. Total extra detections.
3. Missing visible endpoints.
4. Conditional median endpoint error.
5. Conditional p95 endpoint error.

Negative differences are improvements for these measures. Preserve per-seed comparisons against M0 as well, the matched endpoint diagnostics, illustrative two-corner radius counts and empty-image results. Missing endpoints are reported separately from conditional coordinate errors. Do not infer coordinates for reference-unlocatable corners.

Across the two new seeds, report each metric's **arithmetic mean paired change, minimum, maximum, and number improved/tied/worsened**. Retain every per-seed value so an average cannot conceal opposite effects. A mean of two seed-level p95 values is a summary of those two percentiles, **not a pooled endpoint p95**. Undefined statistics cannot silently become successful comparisons.

The summarizer expects `seed-43/` and `seed-44/` beneath an evidence directory, each with `m0`, `m1`, `m2`, their matching `-quality` directories, and `checkpoints/{reference,candidate}`. The same complete M0 evidence may be copied into both seed folders; it is one baseline inference, not two independent observations. Its prediction payloads and evaluation provenance must agree across the two folders.

```bash
python scripts/summarize_box_repeatability.py \
  --root /path/to/hydrated-dl \
  --evidence /path/to/repeatability-evidence \
  --output /path/to/new-repeatability-summary.json
```

The saved-prediction summary requires original validation PNGs/labels and the small checkpoint evidence bundles. It does not train, rerun inference or deserialize weights. Historical seed-42 evidence retains its own pinned replay source; do not rewrite its receipts to pass the newer source checks.

## How this study will affect the project decision

This is a **repeatability study**, not a replacement model-selection rule. Each seed's original five-measure comparison against both M0 and its reference can be reported descriptively, but the overall study does not automatically promote a model or choose a winning seed. The working decision remains `retain-M0` pending a separately justified decision after reviewing the complete evidence.

If both new pairs reproduce the same detection gain and corner tradeoff, the tradeoff deserves further targeted investigation. If the directions differ, the single-seed improvement is unstable under these tested conditions. If the corrections consistently improve every measure, that supports further candidate review; it still does not create independent-test evidence or reverse the historical seed-42 decision.

Two new seeds are a bounded check, not a precise estimate of the full training distribution. No significance claim, confidence interval or deployment guarantee is planned. The development split has already been used for selection, related observations are not independent deployment trials, and the annotation coordinates are not certified physical measurements. Human-reviewed still-image localization remains the supported workflow.

## Preparation and execution status

The [preparation verification record](../results/verification/baomai-box-repeatability-20261107/verification.json) preserves exact commands, exit codes, raw test output, source hashes, the delivered notebook hash and the real-input preflight. Immutable source [`f087c35`](https://github.com/Siuuuers/dl/commit/f087c35f74bf304eb61998edcaa2172fe1cba49c) contains the executed implementation. The delivered notebook differs from its reviewed template only by replacing the source placeholder with that commit.

| Preparation check | Observed result |
| --- | --- |
| Checkpoint identity and fixed recipes | 26 checks passed |
| Runner, seed selection and existing execution guards | 11 checks passed |
| Existing pair and warmup packaging regressions | 19 checks passed |
| Four-run study packaging | 7 checks passed |
| Cross-seed summary contract | 14 checks passed against final shared dependencies |
| Delivered notebook setup and compilation | 5 checks passed; all 10 code cells compile |
| Actual embedded input preflight | All 157 original PNGs and epoch173 bytes verified; exact 8/6/2 edge edits and both fixed recipe identities confirmed |
| Independent code review | No material findings after corrections |

There are **82 distinct passing focused checks**. Repeated checks after final source pinning are not counted twice. The real-input preflight ran at the clean published source and preserved its exact Git tree. Earlier checks were run before publication; their recorded file hashes match the published implementation.

Synthetic completed-bundle fixtures establish validation/packaging behavior; they are not trained models. The summary tests stub the existing replay auditor's boundary to exercise the new study contract. Notebook setup probes use safe substitutes for installation and GPU access. Full training and returned-model measurements for seeds 43 and 44 remain pending the actual GPU run.
