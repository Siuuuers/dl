# R56 explicit horizontal flip experiment

## Purpose

Test whether the same cassette pose model produces useful complementary detections when the input image is horizontally reversed. The two separate views are evaluated in the original image coordinate system. This experiment does not combine their predictions, edit labels, train a model, search thresholds, or promote a model.

This is a proposed inference experiment. Implementation and CPU verification do not establish that horizontal flipping improves cassette recognition. A returned GPU run is required for that conclusion.

## Fixed comparison

| Item | Control | Treatment |
| --- | --- | --- |
| Checkpoint | PR137 candidate | Same bytes |
| Input pixels | Original 1280×720 BGR array | Explicit horizontal reversal of the same array |
| Network input size | 1280 | 1280 |
| Ordinary native NMS | IoU 0.70 | IoU 0.70 |
| Box confidence and cap | 0.13 and 300 | 0.13 and 300 |
| Numeric precision | FP32 | FP32 |
| Additional coordinate operation | None | Inverse horizontal mapping and keypoint slot swap |
| Postclip NMS | Original image frame, IoU 0.70 | Original image frame, IoU 0.70 |
| Evaluation | All four fixed references; matching IoU 0.50; endpoint score 0.50 | Same |

The checkpoint SHA-256 is `222bf3c91a651183047a10eb1a50e16ae8fe999f15372afe005bfca2d0b157f9`. It is the R53 research checkpoint, whose saved outputs equal the earlier PR146 reference under the documented comparison. It is not selected PR35. Preserve that distinction when reporting results.

## Coordinate convention

Pixel-array reversal changes discrete column order. Predicted geometric coordinates use the native continuous image-coordinate convention. For image width W, the inverse map is:

\[
x_{\mathrm{original}}=W-x_{\mathrm{flipped}}.
\]

A predicted box `[xmin, ymin, xmax, ymax]` maps to `[W-xmax, ymin, W-xmin, ymax]`. The keypoint slots swap `[1, 0]` after reflection because the task orders the actual lower corners by image X. Each point's confidence follows that point. The implementation does not sort keypoints by their predicted coordinates, because doing so would conceal a model's mistaken point order.

Use W−x, not W−1−x. The latter describes integer pixel-index reversal, which is a different convention. Native `Instances.fliplr` uses W−x and native `RandomFlip` applies the configured keypoint slot permutation. These are inspected in the pinned Ultralytics 8.4.182 package:

| Native file | SHA-256 |
| --- | --- |
| `utils/instance.py` | `ebc8b6337f4636c1457473c079d007693b3d8766c4891e2b5dd521c22c2e5527` |
| `data/augment.py` | `802333ea359def55e88f61d6726e35c29d28a6491110840a5e912ec1fafd1de3` |

The runner verifies these hashes together with the existing inference-source hashes. No generic `augment=True` path is used.

The order is native output scaling/clipping, inverse horizontal mapping, then the existing postclip NMS in the original frame. No extra clipping is introduced. A point mapped to coordinate W remains at W and therefore fails the existing `x < W` endpoint-bounds check; it is not silently moved to W−1. Unavailable annotation points remain unchanged in the fixed references.

## Baseline and repeatability protection

The unchanged control runs first on all 20 hash-verified originals. It must reproduce the PR137 raw and postclip records within the inherited R53 tolerances: 0.05 original pixel per coordinate component, 1e−5 per confidence, and 0.075 pixel per eligible endpoint error. Discrete match, miss, duplicate, and retained-output identities must remain exact, as must each row's returned endpoint slots and 5/10-pixel success state. A small floating-point change does not excuse crossing an endpoint success threshold.

A failed control blocks treatment and still produces a replayable diagnostic archive. Successful runs use two warmup calls per arm followed by three paired repeats, alternating arm order. Repeatability must pass before treatment interpretation is allowed.

Frozen666 remains the primary reference; historical663, R38, and R50 are separately named diagnostic views of the same photographs. The development screen requires fewer primary misses without increasing unmatched or duplicate outputs, newly missing a two-corner front, reducing all-eligible visible-corner success, or adding outputs to an empty scene. Every gained and lost case remains visible even when an aggregate count is unchanged. No automatic promotion occurs.

## Returned evidence

The archive `baomai-flip-inference-r56-results.zip` contains:

- Original-frame raw and postclip predictions for both arms.
- Native-view predictions before inverse mapping, including all repeated predictions.
- All four reference reports, corner-availability and original-height strata, paired stable row identities, and unmatched/duplicate records.
- `complementarity.json`, with control-only detections, flip-only detections, targets missed by both views, corner-success gains/losses, and reference-empty scene outputs.
- Baseline and repeatability checks, runtime/package/source records, timing and memory records, source/data preflight, both the R56 driver and the reused R53 helper, and a complete member hash inventory.

The “matched by either” quantity in the complementarity report uses reference associations. It is a diagnostic bound on possible recovery, not an implementable ground-truth-free selector and not a fused model result. Unmatched outputs remain separate; combining views could create extra duplicate or unsupported outputs.

CPU replay reconstructs native-to-original inverse mapping, postclip suppression, scores, paired cases, complementarity, repeated-output comparisons and timing summaries. It does not rerun the neural network.

## Implementation and validation

The implementation reuses the R53 source/data closure, scorer, baseline comparison, case inventory and repeatability logic. The new driver is independently hash-pinned and included in the returned inventory. The original R53 code and existing data files remain unchanged.

Eight focused tests cover pixel-array reversal and immutability, continuous inverse geometry, double reflection, slot/confidence retention, missing-coordinate retention, inverse mapping before postclip suppression, strict baseline threshold crossings, complete and blocked offline return replay, rejection of an incorrectly inverted native output, case identity in complementarity, and the four notebook cells.

These CPU checks use synthetic predictions where a model result is needed. They are implementation checks and must not appear as experimental recognition results. Final notebook source pins and actual bootstrap receipts are recorded at publication.

## Run

Use `notebooks/baomai_flip_inference_r56.ipynb` after its immutable source pin has been published. Select a GPU runtime and run all four code cells. The notebook checks inputs before installing the fixed compatible inference packages; an incompatible source or cache is preserved rather than overwritten. The default working base is `/content/baomai-r56`.

The command-line equivalent accepts `prepare`, `run`, and `verify` actions:

```bash
python -B scripts/run_flip_inference_r56.py prepare \
  --source-commit FULL_PUBLISHED_COMMIT \
  --script-sha256 PUBLISHED_DRIVER_SHA256 \
  --root SOURCE --work WORK --cache-dir CACHE
```

For `run`, add the required `--device` selection. Existing verified archive and image inputs may be supplied with `--archive` and `--originals-dir`. Source revisions and driver hashes must be complete immutable values.

Build the notebook after committing the source:

```bash
python -B scripts/build_flip_inference_r56_notebook.py \
  --source-commit FULL_PUBLISHED_COMMIT
```

## Interpretation limits

Horizontal reversal creates no new scene evidence. It probes the learned model's response to orientation and its transformation consistency. Improvements on these repeatedly examined development images remain exploratory and require later independent capture evaluation. A failed flip comparison does not establish that all test-time augmentation methods are ineffective, and a successful separate view does not establish that merging views will help.
