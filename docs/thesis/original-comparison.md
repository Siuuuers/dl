# Original supplied model versus the proposed method

The report's main comparison uses two clearly defined methods: the original supplied cassette detector and the trained detector with equal corner averaging. The original checkpoint is the supplied epoch-173 model, not a later research control. Both use the same photographs, annotations, confidence thresholds, box matching and additional suppression.

## Current archived measurement

These numbers are recomputed from verified historical predictions. Both models have been rescored consistently, but their original inference environments differ. The new notebook runs the complete comparison again in one environment before a common-runtime result is claimed.

| Reviewed reference: 654 fronts; 583 complete corner pairs | Supplied detector | Proposed method |
|---|---:|---:|
| Missed fronts | 34 | 18 |
| Extra detections | 61 | 51 |
| Duplicate detections, included among extras | 5 | 5 |
| Both corners within 5 pixels | 555 / 583 | 569 / 583 |
| Both corners within 10 pixels | 577 / 583 | 583 / 583 |

At 5 pixels, 18 reviewed objects change from failure to success and four change from success to failure. The net gain is 14 objects, or 2.40 percentage points. The denominator includes missed objects; it is not restricted to successful detections. The proposed method still has 14 reviewed complete pairs above the 5-pixel tolerance and 18 missed fronts, so this is not a perfect automatic recognizer.

### Annotation sensitivity

| Earlier fixed reference: 666 fronts; 584 complete pairs | Supplied detector | Proposed method |
|---|---:|---:|
| Missed fronts | 26 | 34 |
| Extra detections | 41 | 55 |
| Duplicate detections, included among extras | 11 | 9 |
| Both corners within 5 pixels | 534 / 584 | 525 / 584 |
| Both corners within 10 pixels | 580 / 584 | 574 / 584 |

The direction changes under the earlier annotations. The main result is therefore an improvement against the reviewed reference, not an improvement independent of annotation choice. The reviewed reference is the selected measurement standard because the annotations were reviewed, not because the result happens to favor a method. Both tables must remain available. Both references annotate the same 20 photographs and do not form separate validation sets.

## Shared comparison procedure

1. Verify the two full checkpoint hashes and the exact 20 original PNG files. The images are 1280 by 720 pixels.
2. Run each model in FP32 at image size 1280 with ordinary class-aware non-maximum suppression at intersection-over-union 0.70. Retain boxes with confidence strictly above 0.25. The trained model also runs once on horizontally reflected pixels.
3. Map reflected coordinates back to the original image and swap the two corner slots. Apply the same additional suppression to every view: suppress a lower-scored same-class box when its clipped-box intersection-over-union with a retained box is strictly above 0.70.
4. Associate trained original-view and reflected-view detections using mutual best same-class box overlap of at least 0.50. Equal overlaps choose the lower record index. If both corners in both records are valid and have confidence at least 0.50, average corresponding coordinates equally. Otherwise retain the trained original-view record. Preserve original-view boxes, identities, class scores and corner scores.
5. Match each method to the same reference with box intersection-over-union at least 0.50. Score endpoints with confidence at least 0.50. A complete pair succeeds only when both Euclidean corner errors are at most the stated tolerance. All complete reference pairs remain in the denominator, including missed detections and unavailable predicted pairs.

The extra suppression is common processing applied to supplied weights. This comparison does not claim to recreate every setting of an upstream live-camera program. The box threshold 0.25 follows the supplied detector's default, and the same threshold is applied before association for both trained views. No confidence, averaging weight, model epoch or annotation is selected by this comparison run.

The archived proposed output averages 648 of 687 retained records. Another 29 have no mutual partner, five have an unavailable original-view pair and five have an unavailable reflected-view pair. These cases retain the original trained prediction. The averaging step preserves detection boxes; reductions in missed and extra detections relative to the supplied model therefore arise from the trained detector. The averaging step changes corner positions.

## Model and data identities

| Item | SHA-256 |
|---|---|
| Supplied epoch-173 detector | `e0aab1b377491700318e0b14cd4c558ec3c1684087a13b7dde2298ebdbf2847e` |
| Trained detector | `222bf3c91a651183047a10eb1a50e16ae8fe999f15372afe005bfca2d0b157f9` |
| Original saved evaluation archive | `3dfe3415841b47a605e2fd4b9e490b2c12b348b0014c6c2a8c78c06f323ffc1a` |
| Trained original/reflected prediction archive | `88689ce9f3d03178683ea98ce98d50f5ddc4e07ba1fb24262e7820e9488bd114` |

The original archive contains predictions at confidence settings 0.15, 0.25 and 0.40. The audit reproduces all 60 image/threshold score records and all three original summaries before selecting the prespecified 0.25 output for common scoring. The original archive used CPU Torch 2.10.0; the trained views used A100 Torch 2.14.1. Ultralytics 8.4.182 is common to both. The confirmatory run removes this runtime difference by rerunning both models together.

The supplied model was selected using the 20 validation photographs. These images have also guided subsequent development. Their repeated use permits a controlled development comparison, but cannot measure generalization to an independent capture session. No physical length calibration is available; pixel errors are not presented as millimetre errors.

## Reproducible artifacts

- [Archived measurement and individual cases](evidence/original-baseline/README.md)
- [Comparison driver](../../scripts/run_original_comparison_r59.py)
- [Notebook builder](../../scripts/build_original_comparison_r59_notebook.py)
- [Four-cell comparison notebook](../../notebooks/baomai_original_comparison_r59.ipynb)

The notebook checks immutable source and all model/image/annotation identities, runs three model passes per photograph, reconstructs every score from saved predictions, and exports one compact ZIP. GPU is the default; explicitly setting `BAOMAI_ORIGINAL_COMPARISON_DEVICE=cpu` runs the same protocol on CPU. It does not retrain either detector and does not automatically replace an application model.
