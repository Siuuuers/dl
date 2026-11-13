# PR35 input-pixel occlusion diagnostic

**Completed: keep PR35.** This is a development experiment on four existing images, not training or a model promotion. Read the [result and decision](../../../docs/pixel-occlusion-2026-11-13.md) first.

The covered endpoint remains within 10 px of its original known coordinate in 28/30 left-mask cases and 26/30 right-mask cases. Every designated point is still returned above the 0.50 display threshold, including six covered endpoint errors over 10 px. Confidence therefore does not certify hidden-point accuracy.

## Contents

| File | Purpose |
| --- | --- |
| [evaluation.zip](evaluation.zip) | Nine-file replay capsule with frozen protocol, preparation, cases, original references, expected intact arrays, all 94 prediction records, scores, summary, and complete status |
| [summary.json](summary.json) | Readable projection with result tables, conditional confidence changes, and named paired gains/losses; omits only repeated individual-error arrays retained in the complete archive summary |
| [verification.json](verification.json) | Publication, archive, model/run bindings and replay scope |
| [independent-review.json](independent-review.json) | Independent reconstruction of 94 input hashes, matching/counts, 120 target-condition observations and 240 endpoint checks |

Archive: **737,981 bytes**, SHA-256 **`f2dfbc58479ec48172137ceec5b07c5d74f5fb60428dd0c42bd12590d43a780e`**. It expands into `evaluation/`; there are no image or weight bytes in this capsule. The older 2.1 MB PR58 replay archive remains a dependency for checking the published baseline during replay.

The [protocol and inference runner](https://github.com/Siuuuers/dl/commit/3647d0937ab036974cda4867815a8fe20aef07d7) were published and fetched back exactly before masked inference. Protocol SHA-256 is `a8723e77d0f23aa22b62c5788967c0a7593b453996c4e1956bd714c93a606305`; runner SHA-256 is `35385aa16dfd4aeba76738fd766d92761a451a885fa584be75c7b17a5653d361`.

## Replay without model inference

From a checkout of this PR containing the normal reference manifests, source helpers, and the prior PR58 archive, choose fresh extraction/output directories outside the checkout. For example:

```bash
python -m zipfile -e results/verification/baomai-pixel-occlusion-20261113/evaluation.zip /tmp/pixel-occlusion-capsule
python scripts/probe_pixel_occlusion.py \
  --repo-root . \
  --protocol experiments/pixel-occlusion-2026-11-13.json \
  --replay /tmp/pixel-occlusion-capsule/evaluation \
  --output /tmp/pixel-occlusion-replayed
```

The runner refuses an existing output directory or an output overlapping the checkout or source capsule. Replay validates source/protocol/preparation bindings, the exact run inventory and hashes, raw record identities, all four published intact arrays, and the recomputed scores/summary. Successful output ends with `"status": "complete"` and `"replayed": true`. It does not rerun the model, verify absent image/weight bytes, or establish natural hidden-corner accuracy.

The actual inference run did verify the exact original PNGs, current label files, PR35 checkpoint, prepared pixel arrays, and source helper hashes. Each of its four intact arrays matched the published PR35 predictions before the first masked input was evaluated. The independent review separately rebuilt the masks and recomputed association/endpoint arithmetic without importing the runner's implementation.

## Recreate preparation and inference

Use the compatible CPU environment and verified PR35 bundle from the [folder-inference guide](../../../docs/predict-pose-folder.md). Set `--image-root` to a directory containing the exact original `new_000007.png` through `new_000010.png`, and `--checkpoint-manifest` to the completed PR35 candidate manifest. The protocol pins all image, label, source, checkpoint, and baseline hashes.

```bash
python scripts/probe_pixel_occlusion.py \
  --repo-root . \
  --protocol experiments/pixel-occlusion-2026-11-13.json \
  --image-root /path/to/original/images/val \
  --checkpoint-manifest /path/to/pr35/candidate/checkpoint-manifest.json \
  --prepare-only \
  --output /tmp/pixel-occlusion-prepared

python scripts/probe_pixel_occlusion.py \
  --repo-root . \
  --protocol experiments/pixel-occlusion-2026-11-13.json \
  --image-root /path/to/original/images/val \
  --checkpoint-manifest /path/to/pr35/candidate/checkpoint-manifest.json \
  --run-prepared /tmp/pixel-occlusion-prepared \
  --output /tmp/pixel-occlusion-fresh-run
```

Preparation performs no inference. The run makes 94 predictions plus one excluded blank warmup. It stops if any intact prediction array differs from the archived baseline. Runtime differences must be resolved before interpreting masked results; the comparison is intentionally exact.

## Reproduce the illustration

With Pillow, DejaVu Sans, the exact original images, and the complete capsule or fresh run:

```bash
python scripts/plot_pixel_occlusion.py \
  --run-dir /tmp/pixel-occlusion-capsule/evaluation \
  --image-root /path/to/original/images/val \
  --output /tmp/pixel-occlusion-figure.png \
  --index /tmp/pixel-occlusion-figure-index.json
```

The [published figure](../../../docs/images/pixel-occlusion-20261113.png) shows the first and last frozen target, plus the explicitly post hoc largest-error example. Its [index](../../../docs/images/pixel-occlusion-20261113-index.json) records source hashes, selection, crop rectangles, exact input/case/prediction identities and rendering details. The crops were not model inputs.

## Scope of the result

Original boxes provide frozen association geometry after masking, and original coordinates remain known even where masked pixels make them unobserved. Every condition retains all 30 designated targets. Full-image results also retain collateral effects: 51/90 masks cover another reference endpoint center. The center-face condition removes meaningful pixels and is not a sham.

The four scenes were already used for development and may be correlated. No new captured data, accepted hidden labels, training updates, default changes, calibrated physical-visibility classifier, or deployment acceptance result is claimed.
