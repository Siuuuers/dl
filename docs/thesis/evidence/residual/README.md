# Residual correction implementation evidence

These checks verify an implementation and its proposed training targets. **No full residual training run or resulting development accuracy is reported here.**

- `pretrained-cpu-smoke.json`: actual hash-verified pretrained ResNet50 on two real training photographs, with disposable perturbed annotation-box fixtures. It records exactly zero initial offsets, exact preservation of original coordinate records, finite nonzero SmoothL1 loss, an output-layer update and zero gradient for an unavailable endpoint. The experiment network is reinitialized after this check. Runtime: Torch 2.9.0 CPU and TorchVision 0.24.0 CPU.
- `training-target-audit.json`: reconstruction of eligible residual targets from the verified returned frozen training predictions and staged labels. It records 3,809 crops and 7,204 endpoint targets, all exclusion counts and correction-size distributions. Fresh notebook inference must generate and bind its own training predictions; this file does not claim it has done so.
- `implementation-check.json`: source hashes and the result of 11 focused tests plus an independent implementation review. The tests check identity, actual resize scales, endpoint masks, unavailable/invalid outputs, unchanged hints, strict confidence filtering, complete saved-coordinate replay, notebook compilation and failed-baseline packaging/recovery.
- `published-bootstrap.json`: successful execution of the actual notebook source and data-preparation cells, followed by the actual pretrained CPU mechanics action, from published source `06e55d9c1b2557014aff7ff0d68b5096890fe546`. All 697 source/data files passed verification, the source remained clean, and the model's initial state matched the earlier mechanics check. Neither full detector inference nor experiment training ran in this bootstrap.

Run the focused checks with the repository's native package versions and a compatible Torch/TorchVision pair:

```bash
python -B -m unittest discover -s tests -p test_residual_corner_r59.py -v
```

After the notebook has fetched its pinned source, the actual model mechanics can also be checked independently of experiment training:

```bash
python -B scripts/run_residual_corner_r59.py smoke \
  --root /path/to/immutable-source \
  --source-commit FULL_PUBLISHED_COMMIT \
  --script-sha256 FULL_DRIVER_SHA256 \
  --work /path/to/separate-work \
  --cache-dir /path/to/cache
```

Use the actual values embedded in the published notebook. The driver refuses a dirty or differently pinned source checkout and preserves incompatible files. The [experiment guide](../../residual-correction-experiment.md) describes the hypothesis, recipe, accounting and return format.
