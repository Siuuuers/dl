# R39: trace the remaining R38 misses before another training run

This is a fresh CPU inference diagnosis of the returned **PR123 candidate**, using the accepted **R38 development reference**. It changes no annotations, model weights, loss, matching rule or selected deployment model. The eleven R38 misses are traced through the decoded one-to-many proposals, confidence filtering and ordinary NMS.

## Results

All twenty instrumented primary prediction arrays reproduce the accepted PR123 cache **exactly**, including coordinates, scores, keypoints, ordering and the three empty scenes. Reconstructing NMS with original proposal indices also reproduces those arrays exactly. Each image produces 19,320 decoded proposals. The primary settings remain confidence 0.13, NMS IoU 0.70, size 1280, rectangular preprocessing, FP32 CPU, batch one and maximum 300 outputs. Matching remains IoU 0.50.

| Cause at the primary operating point | Misses | Meaning |
| --- | ---: | --- |
| No decoded box reaches matching IoU 0.50 | 4 | Even before confidence/NMS, no decoded rectangle has sufficient overlap with that accepted reference box. |
| Confidence gate | 6 | Geometrically qualifying proposals exist, but every such proposal scores at or below 0.13. |
| NMS suppression | 1 | Geometrically qualifying proposals score above 0.13, then are suppressed by a higher-scored retained rectangle. |
| Output limit or unresolved suppression | 0 | Neither the 30,000 pre-NMS cap nor 300-output cap explains a missed target. |
| Retained proposal lost through assignment | 0 | No remaining miss has a qualifying retained box. |

These are stages of this inference pipeline, not diagnoses of why training learned those scores or rectangles. A low-confidence raw box with convenient geometry is not itself a recovered detection.

### Every missed identity

Rows below use R38 identities, with the original R37 row retained where it shifted.

| Image | R38 row | R37 row | Mechanism | Highest score among IoU-eligible raw boxes | Recovered at confidence 0.05? |
| --- | ---: | ---: | --- | ---: | --- |
| new_000008 | 33 | 33 | Confidence | 0.00908 | No |
| new_000008 | 34 | 34 | Confidence | 0.09797 | Yes |
| new_000009 | 54 | 55 | Confidence | 0.02762 | No |
| new_000010 | 30 | 31 | Confidence | 0.09515 | Yes |
| new_000010 | 33 | 34 | Confidence | 0.10344 | Yes |
| new_000011 | 55 | 55 | Geometry | None | No |
| new_000011 | 56 | 56 | Geometry | None | No |
| new_000011 | 57 | 57 | Geometry | None | No |
| new_000011 | 60 | 60 | Geometry | None | No |
| new_000088 | 51 | 51 | Confidence | 0.09677 | Yes |
| new_000090 | 39 | 39 | NMS | 0.67983 | Yes, through a different lower-scored proposal |

All four geometry failures are in `new_000011`. Their best raw IoUs are respectively 0.49935, 0.40651, 0.33883 and 0.41476. The first is close to the established 0.50 cutoff; retain that fact without changing the cutoff or accepted label to manufacture a match.

### The concrete NMS case

For `new_000090:39`, decoded proposal **9354** has confidence **0.67983** and reference IoU **0.53746**. Primary output **p48**, decoded proposal **9355**, has the higher confidence **0.77317**, but reference IoU only **0.48971**. Their input-space IoU is **0.89866**, so ordinary NMS at 0.70 retains p48 and suppresses proposal 9354. The trace records the same relationship for the other above-threshold qualifying proposals. Neither output cap is reached.

At confidence 0.05, a different decoded proposal, **9352**, survives: confidence **0.10359**, reference IoU **0.89972**, probe output **p65**. Thus the lower-confidence probe recovers this target without proving that a universal NMS setting should change. Raw proposal indices are zero-based; displayed prediction IDs are one-based within their own setting.

## One bounded confidence counterfactual

The same decoded tensors were passed through the same pinned NMS at confidence **0.05**. This is a postprocessing counterfactual, not another forward pass, training run, threshold sweep or selected operating point. Every primary output remains present with identical fields.

| R38 measure | Primary 0.13 | Probe 0.05 | Change |
| --- | ---: | ---: | ---: |
| Reference fronts | 654 | 654 | 0 |
| Returned outputs | 737 | 810 | +73 |
| Matched fronts | 643 | 648 | +5 |
| Missed fronts | 11 | 6 | -5 |
| Unmatched outputs | 94 | 162 | +68 |
| Duplicate subset | 29 | 48 | +19 |
| Other unmatched | 65 | 114 | +49 |
| Both corners within 5 px | 570/583 | 570/583 | 0 |
| Both corners within 10 px | 582/583 | 582/583 | 0 |
| Outputs on three empty scenes | 0 | 0 | 0 |

There are no newly missed identities. The five recovered fronts all belong to the partial-visibility population. Primary detection is 583/583 for two-corner fronts and 60/71 for fronts with fewer than two locatable corners; the probe raises the latter to 65/71. The additional 68 unmatched outputs are the measured cost of the five recoveries. The duplicates are already included in unmatched outputs. Not every unmatched output has been independently adjudicated as background.

This supports targeted learning evidence for partial-front confidence and localization, with duplicate behavior tracked alongside recall. It does not support adopting a global threshold reduction without an application cost criterion. The development images have been reviewed repeatedly and do not establish unseen-session performance.

## Evidence and replay

- [diagnosis.json](diagnosis.json): input/runtime identity, exact baseline checks, per-image decoded-tensor fingerprints, original input-space boxes of all retained primary outputs, every missed case and its exhaustive list of IoU-eligible raw proposals, and concrete NMS suppressors.
- [counterfactual-predictions.jsonl](counterfactual-predictions.jsonl): all twenty confidence-0.05 prediction arrays and their original decoded proposal indices.
- [summary.json](summary.json): primary/probe metrics, recovered identities and mechanism totals.
- [duplicate-overlap-audit.json](duplicate-overlap-audit.json): companion audit of the accepted primary duplicate outputs.
- [diagnose_r39_proposals.py](../../../scripts/diagnose_r39_proposals.py): capture and standard-library replay entry points.

The fresh inference used Ultralytics 8.4.182, PyTorch 2.10.0+cpu and four CPU threads. Checkpoint, image, labels, accepted predictions, runtime NMS implementation and diagnostic source identities are recorded. The fresh instrumented baseline matched all accepted outputs. A subsequent `python -S` replay independently rescored the saved primary and probe arrays against the hashed R38 labels and reproduced the saved summary exactly.

Replay example from the repository root:

```bash
python -S scripts/diagnose_r39_proposals.py \
  --replay results/evaluation/pr123-r39-diagnosis-20261126/diagnosis.json
```

The compact replay rechecks the retained eligible-proposal geometry and output identities. It does not regenerate the complete raw tensors. Their hashes record identity; independently establishing that a list is exhaustive or that no qualifying geometry exists requires the original images, checkpoint and fresh inference. No GPU is needed for the compact replay, and the full diagnostic also ran on CPU.

## Fixed extra NMS after image-boundary clipping

The duplicate audit identified a specific stage difference: ordinary NMS compares decoded input-space rectangles before they are clipped to the original image boundary, whereas evaluation compares the final clipped rectangles. Clipping can increase their mutual IoU. A second bounded probe therefore applies **one extra same-class NMS pass at the existing 0.70 threshold** to the accepted confidence-0.13 output rectangles. The confidence, model, initial NMS and labels stay fixed. Stable primary prediction IDs are retained.

| Measure | Primary R38 | Post-clip probe R38 | Primary frozen666 | Post-clip probe frozen666 |
| --- | ---: | ---: | ---: | ---: |
| Outputs | 737 | 728 | 737 | 728 |
| Matches | 643 | 643 | 631 | 631 |
| Misses | 11 | 11 | 35 | 35 |
| Unmatched | 94 | 85 | 106 | 97 |
| Duplicate subset | 29 | 21 | 29 | 22 |
| Other unmatched | 65 | 64 | 77 | 75 |

The nine removed primary outputs are `new_000007` p50/p51/p54/p55/p58, `new_000008` p39, `new_000011` p55, `new_000055` p27 and `new_000090` p53. On both references, every matched reference retains **the exact same original prediction identity**, so detection matches, missed identities and corner outcomes are unchanged. The three empty scenes remain empty.

This is a promising bounded output-filter candidate. Matching preservation and overlap do not independently establish that every pair depicts the same physical object, and an extra suppression step may affect neighboring cassettes in other scenes. It remains exploratory: no deployed setting or selected model changes here.

[clip-nms-probe.json](clip-nms-probe.json) records each suppression pair, original/clipped rectangles, preclip and postclip IoUs, retained primary IDs, source and reference hashes, both reference scores and exact identity comparisons. [clipped-nms-predictions.jsonl](clipped-nms-predictions.jsonl) preserves all twenty filtered output arrays. The pure-Python [probe script](../../../scripts/probe_r39_clipped_nms.py) independently regenerates the suppression pairs, prediction bytes and both score reports:

```bash
python -S scripts/probe_r39_clipped_nms.py \
  --replay results/evaluation/pr123-r39-diagnosis-20261126/clip-nms-probe.json
```

That complete replay passed. It requires no image, checkpoint, ML package or fresh inference.
