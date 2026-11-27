# Local refinement evidence

This folder records the audits of the returned image-only and initial-corner-hint training experiments. The [result note](../../local-refinement-results.md) explains the outcomes. The controls are the expanded-data research detector, not the supplied original baomai model.

| File | Contents |
| --- | --- |
| `rgb-audit.json` | Exact archive, source/data, training-supervision/order, model-parts, baseline and score replay for the image-only run |
| `hints-audit.json` | The same complete audit for the initial-corner-hint run |
| `paired-analysis.json` | Independently calculated pair errors, gains/losses, cross-arm identity checks, image/size groups and common-confidence results |
| `paired-case-changes.json.gz` | Every complete target's three coordinate states and synthetic ideal-decoder error |
| `ideal-heatmap-diagnostic.json` | Synthetic Gaussian target/decoder capacity experiment, explicitly separate from trained accuracy |
| `rgb-model-package.json` | Independent verification of all nine RGB checkpoint wrappers and concatenated exact model bytes |
| `rgb-selected-cpu-forward.json` | Actual saved RGB model inference on two selected adverse cases |
| `hints-cpu-forward.json` | Actual saved hint model inference on the same cases |
| `manifest.json` | Evidence member byte lengths and hashes |

## Immutable inputs

The image-only result is from commit `bf718680bc5f2888e51ac084610e678fc761244a`, directory `results/t4/baomai-corner-refinement-r58-rgb-t4-001/`. Its compact result ZIP has 3,117,954 bytes and SHA-256 `1f66eae2410c9e9d866f924956e1dc94916c86dd115ec2e43a0c27f5c3fa0b60`. Its full model SHA-256 is `6dcc57f1d9b807e472c62557e8a85d0dfd456e418957737323fea52b0c00fbb6`.

The hints result is from commit `fc0ec1a92e5d9357808e3c9f3bcf87d87450cdd3`, directory `results/t4/baomai-corner-refinement-r58-hints-t4-001/`. Its compact result ZIP has 3,118,004 bytes and SHA-256 `f57e0a53a24a6159e4cb9b91613099146f05e01a483b02e304419a520e69202b`. Its full model SHA-256 is `a389d44a04366529ebf8ec35b76f23467d02ee0cbe515f375b64fc3eb51df69c`.

Both use implementation commit `5b1650b8fa3959a0917680305d7b499af783c5aa`. Each archive contains 21 members with a hash inventory. Each exact FP32 model has 136,332,779 bytes and nine wrapped checkpoint parts. Those bytes remain in their original results branches.

## Reproduce

The [generic audit](../../../../scripts/audit_corner_refinement_returns.py) accepts either complete returned pack. It checks the immutable original training/input archive as well as all returned files. Use a checkout of the pinned implementation as `--root`, or supply independently retrieved pinned files with `--pinned-sources` when an active source file has since changed.

```bash
python -B scripts/audit_corner_refinement_returns.py \
  --root /path/to/pinned-source \
  --directory /path/to/rgb-return-pack --mode none \
  --original-archive /path/to/baomai-additional-front-pilot-r47-results.zip \
  --output /path/to/rgb-audit.json
```

For the hints pack, use `--mode initial`. The script replays 16 aggregate and 320 per-image tables per run, every raw-output decoder trace, training masks and 30 sample-order hashes. Its Torch runtime is recorded; the audit does not retrain the model.

Extract both compact evidence ZIPs into separate directories, then run the [independent paired analysis](../../../../scripts/analyze_corner_refinement_pair_r59.py):

```bash
python -B scripts/analyze_corner_refinement_pair_r59.py \
  --root /path/to/source \
  --rgb-return /path/to/extracted-rgb-evidence \
  --hints-return /path/to/extracted-hints-evidence \
  --output-dir /path/to/paired-audit
```

This second script checks the exact known inventory hashes, independently reconstructs crop/decoder arithmetic, calculates complete-pair errors directly, and verifies the two runs' shared conditions. It also repeats scoring at strict box confidence above 0.25 and runs the ideal reference-Gaussian diagnostic. The diagnostic uses reference coordinates to construct synthetic targets and must not be reported as achieved recognition accuracy.

The selected-case CPU forward checks use the exact saved models and checked original PNGs. They match the returned corner coordinates on two cases, with Torch 2.9 CPU rather than the recorded Torch 2.14.1 T4 runtime. This is a limited model check, not a complete fresh reproduction of all GPU heatmaps.
