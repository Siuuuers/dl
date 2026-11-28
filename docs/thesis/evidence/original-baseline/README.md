# Original supplied model comparison evidence

The primary comparison is the original supplied epoch-173 detector versus the trained detector with fixed equal corner averaging. Both receive the same processing and are scored against the same reviewed annotations. The earlier fixed annotations remain a sensitivity check.

## Fresh common-runtime comparison

`fresh/summary.json` is the primary measurement. Both original and trained checkpoints were run in the same CPU environment; the trained checkpoint was evaluated on original and reflected pixels. All 60 required model/image passes completed. The full result ZIP contains source and input identities, three raw prediction views, processed predictions, suppression and association decisions, per-reference cases, all measurements and replay receipts.

- Source commit: `b4e8420ef6014daae9693c18ed564c224067cda3`.
- Driver SHA-256: `279e2c0bd701299ea2a6f30403e97da68cc44a9c4ff874c5ee5a0d155a6e6ea3`.
- Result ZIP: 1,133,014 bytes; SHA-256 `aae05d14298afad0263197fe4ab7e0e753f63445d9d3503dc1b006030e16c516`.
- Runtime: Python 3.12.14; Torch 2.9.0+cpu; TorchVision 0.24.0+cpu; Ultralytics 8.4.182; NumPy 2.4.6; OpenCV 5.0.0.93; Pillow 13.0.0; FP32.

`fresh/notebook-execution.json`, its four cell logs, and `fresh/executed-notebook.ipynb` record execution of all four unchanged notebook code cells in an in-process IPython shell. A fresh immutable source checkout and hash-verified cached inputs were used. A separate Jupyter kernel could not start because local sockets were unavailable, so this checks actual code-cell execution and full CPU inference, not the Colab browser interface. The notebook's CPU override was explicit; its default remains GPU.

Replay verifies the inverse reflection, common suppression, view association, equal averaging, reference matches, individual outcomes and final scores from saved raw predictions. Historical continuity also passes for all three views; the largest coordinate-component difference is 0.000184 pixels. All headline counts and case outcomes are unchanged. `fresh/manifest.json` binds every saved evidence file.

The reviewed main result is original/proposed 555/569 successful pairs within 5 pixels, from 583 eligible objects, with 34/18 missed fronts and 61/51 extras. The earlier fixed reference gives 534/525 of 584 within 5 pixels, with 26/34 misses and 41/55 extras. Both results remain part of the evidence.

## Archived comparison

`archived/archived-summary.json` contains the same-threshold replay of historical predictions. This is not a fresh common-runtime inference result: the original detector used CPU Torch 2.10, while the trained detector's two views used A100 Torch 2.14. The completed fresh comparison above reruns both in one recorded environment.

The archive was not trusted by its summary alone. The audit checks its exact bytes, every inventory entry and image/label identity, then reproduces the original model's 60 per-image rows and all three original aggregate scores. The predictions are then rescored under the common protocol. Coordinate averaging is recomputed after the common confidence filter and suppression.

The six compressed JSON files preserve raw outputs, processed outputs, suppression decisions, view associations, reference-level cases and per-image metrics. `manifest.json` binds their exact bytes. `scripts/run_original_comparison_r59.py archive` reproduces them from immutable source archives.

```sh
python -B scripts/run_original_comparison_r59.py archive \
  --cache-dir /tmp/baomai-comparison-cache \
  --work /tmp/baomai-comparison-archive
```

The original model SHA-256 is `e0aab1b377491700318e0b14cd4c558ec3c1684087a13b7dde2298ebdbf2847e`. The trained model SHA-256 is `222bf3c91a651183047a10eb1a50e16ae8fe999f15372afe005bfca2d0b157f9`. They are distinct checkpoints; the trained model is not relabelled as the original baseline.

The primary reviewed reference has 654 fronts and 583 complete corner pairs. The fixed sensitivity reference has 666 fronts and 584 complete pairs. All 20 photographs are development data. Their two annotation references are not independent test sets.
