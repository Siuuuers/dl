# Training record: known corners under pixel corruption

## Completed result — 2026-11-14

**The PR62 return has been evaluated; retain PR35. No repeat GPU run is needed.** The candidate improves fixed covered-side ten-pixel successes from 54/60 to 58/60, but fails the clean preservation gates: unmatched proposals 85→86, returned locatable endpoints 1,207→1,205, and empty-scene outputs 0→1. Visual review explains the endpoint-count loss as a box-match crossing and confirms a genuine background false output plus clean-corner displacement.

Read the [completed result and decision](occlusion-results-2026-11-14.md), including raw evidence, exact PR35/control reproduction and offline replay. The frozen protocol below remains the historical experiment design. Operational settings remain PR35 / 1280 / 0.13 / NMS 0.70 / endpoint 0.50.

## Original hypothesis

The completed [PR60 diagnostic](pixel-occlusion-2026-11-13.md) found 28/30 left-covered and 26/30 right-covered endpoints within 10 px of their original references. All covered predictions remained above the display threshold, including six observations over 10 px error. Its largest covered-point error was 18.11 px at confidence 0.927. These findings justified testing training-only corruption of known corners while preserving clean-image behavior. They did not establish accurate naturally hidden coordinates or an irreducible error floor.

## Reproduce the completed experiment (historical instructions)

1. Open [the occlusion-training notebook in Colab](https://colab.research.google.com/github/Siuuuers/dl/blob/codex/occlusion-training-20261113/notebooks/baomai_occlusion_training.ipynb).
2. Select **Runtime → Change runtime type → GPU**, then **Runtime → Run all**. Authorize its Drive mount to preserve completed work. Keep the same runtime for both runs.
3. Return **`baomai-occlusion-seed46-results.zip`**. Keep the verified Drive backup until the comparison is evaluated.

The notebook checks out implementation commit [`b457a4a69751`](https://github.com/Siuuuers/dl/commit/b457a4a6975104700d7342fa4b997ea6da7625ad) and verifies the frozen protocol SHA-256 `c70563034d3ac969ff7837a594d53bc6316fda3897e1557b5d7828be52e94a62`.

No manual label edits or separate image upload are needed. The notebook retrieves the exact 157 original PNGs and original epoch173 initialization weights with selective Git LFS, checks their identities, and verifies the chosen label composition. It installs Ultralytics 8.4.182 while preserving the runtime's Torch/torchvision versions.

The notebook first performs four candidate training batches at the actual batch size and resolution. That trial must observe an optimizer update, parameter change, and applied masks. Its weights never initialize either full run. Each full arm starts independently from epoch173 and completes the fixed 30 epochs. The selected checkpoint is finalized epoch-30 `last.pt`; there is no best-epoch search.

Compatible completed work is restored from `MyDrive/baomai-occlusion-seed46-20261113`. Completed arms are checked and skipped. Partial or incompatible local runs are preserved and refused; choose a fresh `RUN_BASE` for a retry. A changed source or runtime needs a compatible matched pair. When both arms are already complete, the notebook verifies or creates the return ZIP without training them again.

## What changes between the two runs

| Item | Control | Candidate |
| --- | --- | --- |
| Training originals | Same 137 images | Same 137 images |
| Training labels | Exact PR35 composition | Exact PR35 composition |
| Initialization | Fresh original epoch173 | Fresh original epoch173 |
| Seed / epochs / input / batch | 46 / 30 / 1280 / 4 | Identical |
| Base optimizer and augmentation | Existing PR35 numerical recipe | Identical |
| Corruption planning | Runs with a private RNG; changes no pixels | Same planner; applies valid masks |
| Validation inputs | Original, uncorrupted images | Original, uncorrupted images |

The effective training labels contain **3,771 objects, 6,841 locatable endpoints and 701 missing slots**. Both runs select the existing `annotations-2026-11-10` overlay, including its exact eight label overrides. Pending border-front, endpoint and composed label candidates are not mixed into this comparison. The original 20-image in-training validation remains unchanged, with 664 objects and 1,227 locatable endpoints. Post-training evaluation uses the separately versioned current 663-object reference.

The base numerical recipe remains `reviewed-annotations-seed46-v1`: FP32, rectangular batches, AdamW, learning rate 0.0001, bias warmup 0.0001, four CPU threads, zero loader workers, and the established spatial/color augmentation. Its existing recipe ID/hash identifies those numerical settings. The additional protocol, role, corruption parameters and source hashes identify the effective experiment; the candidate is not represented as having an identical effective training recipe.

For compatibility, both runs use the historical **label arm** named `candidate`. The new **experimental roles** are `control` and `candidate`. Receipts and returned bundles preserve this distinction explicitly. An old PR35 checkpoint with the same base recipe cannot masquerade as either completed new role.

## The added transform

The [small transform](../scripts/pose_input_corruption.py) runs after ordinary training preprocessing, when the image tensor and transformed targets share the current pixel coordinate system. It runs only in training preprocessing. The existing validation path receives no corruption hook.

In the pinned Ultralytics implementation, the ordinary `erasing` option is used by the classification dataset; the pose pipeline does not use it. Setting that option alone would therefore not implement this experiment. See the pinned [dataset implementation](https://github.com/ultralytics/ultralytics/blob/v8.4.182/ultralytics/data/dataset.py) and [augmentation implementation](https://github.com/ultralytics/ultralytics/blob/v8.4.182/ultralytics/data/augment.py).

For an eligible training image, the new transform has probability **0.50** of attempting one mask. It chooses one object with two valid known corners, then chooses its left or right side. The mask is derived from the transformed box, with:

- inward extent sampled uniformly from 15–35% of box width;
- outward extent of 5% of box width;
- vertical padding of the larger of 3 pixels and 15% of box height;
- a uniform gray value sampled from integers 64–192;
- at most one mask per image.

The intended corner's radius-3-pixel neighborhood must be covered, and the opposite corner's neighborhood must remain untouched. A failed geometry attempt is skipped and counted; it is not repeatedly resampled. Empty and ineligible images remain unchanged. The varying width, padding and gray value avoid training exclusively on PR60's exact fixed mask.

The transform uses a separate `random.Random(46013)` generator. It draws nothing from the existing Python, NumPy or Torch global random streams. Control and candidate plan the same masks when their pre-corruption targets and ordering match; only the candidate applies them. Their planning digests must agree at packaging. That digest binds the recorded planning decisions, not every pre-corruption RGB byte.

Receipts report planned/applied masks, skipped attempts, known endpoint coverage and collateral known endpoints. A registered hook with no actual treatment exposure cannot qualify as a completed experiment.

## What the training targets mean after masking

This is **reconstruction of known source geometry from a corrupted input**. The target coordinate was observed before artificial corruption. Its value is not inferred from a naturally hidden corner.

Keep every transformed box, keypoint coordinate, class, batch association and visibility flag unchanged. Source `v=2` describes the coordinate known before corruption; it does not claim that the modified input still shows that point. Source `v=0` remains missing. No model guess or geometric extrapolation fills a missing slot.

The retained box is the transformed pre-corruption visible-front extent. It is neither a newly measured visible extent after masking nor a full amodal box for an entirely hidden object. Do not shrink boxes to the gray rectangle or move corners to mask boundaries.

The endpoint score remains a learned score under this supervision. It is not an independent physical-visibility probability. In the pinned [pose loss](https://github.com/ultralytics/ultralytics/blob/v8.4.182/ultralytics/utils/loss.py), nonzero visibility activates coordinate supervision and a positive keypoint-objectness target; changing to `v=1` would not automatically downweight the point.

This contract changes only the experimental corrupted-input supervision. Canonical source files and the existing observed-label policy remain intact. Historical PR35 label verification runs under its exact historical policy in an isolated temporary snapshot; the live policy and old validators are not rewritten.

## Decide whether it improves the working model

The [protocol](../experiments/occlusion-training-2026-11-13.json) was fixed before training. Assess the candidate against its matched control first, then against the exact operational PR35 checkpoint. Beating a weaker newly trained control alone does not justify replacing PR35.

Use identical inference settings for all three checkpoints: input 1280, box confidence 0.13, NMS 0.70, matching IoU 0.50, and endpoint confidence 0.50. Keep all target identities and missing/suppressed outputs in their original denominators.

### Clean-image preservation

On all 20 original development images, require no deterioration in misses, unmatched proposals, duplicate proposals, locatable-endpoint coverage, or 5/10 px complete-pair successes against either comparator. Require no added outputs on the three empty images. Review every changed match and success identity, preserving supported existing successes and distinguishing actual recovered fronts from small IoU-threshold or assignment changes.

### Benefit under synthetic occlusion

On the exact frozen PR60 cases, require:

1. At least **two additional covered-side 10 px successes out of 60 observations**, against both the matched control and PR35.
2. Supporting gains on at least **two distinct target objects**, each with at least **2 px reduction** in the covered-point error. Tiny threshold crossings alone are insufficient.
3. No decrease in either side's 10 px success count, covered-side 5 px counts, target match/return coverage, opposite-side 10 px counts, or center-control 5/10 px pair success.
4. Complete reporting of changed identities and full-image effects on other objects.

The two-observation/two-object/2-pixel criterion is a prospectively chosen practical minimum. It is not a statistical-significance claim or a user-specified physical tolerance. If the gate fails, retain PR35 and record the result. Do not search seeds, epochs, thresholds, radii or subsets until an apparent gain appears.

The four PR60 scenes and their failures already informed this hypothesis. They are a reused development regression probe, not an independent test. Their altered copies add no new captured scenes. A pass would support a better development candidate; naturally hidden-point accuracy and deployment readiness would still require appropriate independent evidence and an application acceptance contract.

## Preparation and return verification

The [preparation record](../results/verification/baomai-occlusion-preparation-20261113/README.md) preserves the local checks performed before GPU training. The subsequent complete GPU return and its evaluation are recorded in the [result report](occlusion-results-2026-11-14.md). Preparation checks alone do not establish training completion or accuracy.

The [wrapper](../scripts/train_pose_occlusion.py) reuses the existing resource monitor, recorded trainer and checkpoint writer through a small optional preprocessing hook. The [experiment verifier and packager](../scripts/occlusion_experiment.py) add role/protocol/transform binding without changing the PR60-pinned checkpoint or scoring helpers. Completed runs must include all **4,110 training-image presentations and 1,050 batches per role**, the actual mask exposure, and compatible source/runtime/settings.

The returned ZIP contains the two complete six-file checkpoint bundles, the frozen protocol and an index: **14 files**. Its checks establish completed-artifact eligibility and paired-experiment identity. Model accuracy is evaluated separately after the result is returned.

For command-line use, the notebook provides the fixed source pin and environment. The metadata-only preflight is:

```bash
python -B scripts/occlusion_experiment.py verify --root .
```

The final package command is:

```bash
python -B scripts/occlusion_experiment.py package \
  --root . \
  --control /path/to/control \
  --candidate /path/to/candidate \
  --output /path/to/baomai-occlusion-seed46-results.zip
```

Add `--verify-existing` to check an existing return ZIP without replacing it.
