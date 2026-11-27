# Original supplied model comparison evidence

The primary comparison is the original supplied epoch-173 detector versus the trained detector with fixed equal corner averaging. Both receive the same processing and are scored against the same reviewed annotations. The earlier fixed annotations remain a sensitivity check.

## Archived comparison

`archived/archived-summary.json` contains the same-threshold replay of historical predictions. This is not a fresh common-runtime inference result: the original detector used CPU Torch 2.10, while the trained detector's two views used A100 Torch 2.14. The fresh comparison notebook reruns both in one recorded environment.

The archive was not trusted by its summary alone. The audit checks its exact bytes, every inventory entry and image/label identity, then reproduces the original model's 60 per-image rows and all three original aggregate scores. The predictions are then rescored under the common protocol. Coordinate averaging is recomputed after the common confidence filter and suppression.

The six compressed JSON files preserve raw outputs, processed outputs, suppression decisions, view associations, reference-level cases and per-image metrics. `manifest.json` binds their exact bytes. `scripts/run_original_comparison_r59.py archive` reproduces them from immutable source archives.

```sh
python -B scripts/run_original_comparison_r59.py archive \
  --cache-dir /tmp/baomai-comparison-cache \
  --work /tmp/baomai-comparison-archive
```

The original model SHA-256 is `e0aab1b377491700318e0b14cd4c558ec3c1684087a13b7dde2298ebdbf2847e`. The trained model SHA-256 is `222bf3c91a651183047a10eb1a50e16ae8fe999f15372afe005bfca2d0b157f9`. They are distinct checkpoints; the trained model is not relabelled as the original baseline.

The primary reviewed reference has 654 fronts and 583 complete corner pairs. The fixed sensitivity reference has 666 fronts and 584 complete pairs. All 20 photographs are development data. Their two annotation references are not independent test sets.
