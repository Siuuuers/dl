# Completed residual corner correction results

Updated 28 November 2026, Asia/Hong_Kong.

The returned single-step correction model gives **564 successful complete corner pairs out of 583 at five pixels**, compared with 566 for its unrefined research detector and 569 for the existing two-view coordinate average. The original supplied model gives 555. The correction model therefore does not replace the current two-view method.

Its effect is mixed: mean maximum-corner error improves slightly relative to the unrefined control, while the five-pixel success count falls. All values below are measured on development photographs, with no independent capture evaluation.

## Common comparison

The table uses the reviewed annotations, the same 20 photographs, box confidence strictly above 0.25, endpoint confidence at least 0.50, and the same box assignment and suppression rules. A pair succeeds only when both annotated corners are within the stated distance. The denominator includes every complete reference pair, including missed detections or unavailable predictions.

| Method | Missed fronts | Extra outputs | Duplicates included in extras | Both within 5 pixels | Both within 10 pixels |
| --- | ---: | ---: | ---: | ---: | ---: |
| Original supplied model | 34 | 61 | 5 | 555/583 | 577/583 |
| Adapted research detector | 18 | 51 | 5 | 566/583 | 582/583 |
| Existing two-view average | 18 | 51 | 5 | 569/583 | 583/583 |
| Residual correction | 18 | 51 | 5 | 564/583 | 582/583 |

The research detector is the control for the correction experiment. It is distinct from the untouched supplied model used as the report's main baseline. The correction and averaging methods preserve detector boxes and scores, so they cannot recover missing detections or remove duplicate boxes.

The detector, correction and averaging methods each have 583 matched complete pairs with available outputs. Their mean maximum-corner errors are respectively **2.316739, 2.266216 and 2.163264 pixels**. Thus the correction improves mean error relative to its control, but the two-view average retains the better mean and threshold counts. The supplied-model conditional mean uses only 580 matched available complete pairs and is not compared as if it had the same denominator.

## Individual gains and losses

| Residual correction compared with | Five-pixel gains | Five-pixel losses | Ten-pixel gains | Ten-pixel losses |
| --- | ---: | ---: | ---: | ---: |
| Unrefined research control | 4 | 6 | 0 | 0 |
| Existing two-view average | 3 | 8 | 0 | 1 |
| Original supplied model | 18 | 9 | 5 | 0 |

The largest reviewed increase in maximum corner error occurs at `new_000010/R018`, changing from approximately 3.18 to 7.65 pixels relative to the unrefined control. A useful improvement occurs at `new_000008/R029`, changing from approximately 6.73 to 4.67 pixels. These examples were selected after scoring to inspect the implemented model; they are not a separate test set.

At the common threshold, the correction model offers the same 678 available pairs as its control. They comprise 564 confirmed five-pixel pairs, 19 known complete-pair errors, 48 pairs on incompletely annotated matched fronts, and 47 unmatched pairs. The last two categories are unconfirmed, not proof of correct automatic positioning. A success fraction over reference targets must not be interpreted as the probability that an arbitrary offered output is correct.

## Sensitivity to the earlier annotations

| Method | Missed fronts | Extra outputs | Duplicates | Both within 5 pixels | Both within 10 pixels |
| --- | ---: | ---: | ---: | ---: | ---: |
| Original supplied model | 26 | 41 | 11 | 534/584 | 580/584 |
| Adapted research detector | 34 | 55 | 9 | 511/584 | 575/584 |
| Existing two-view average | 34 | 55 | 9 | 525/584 | 574/584 |
| Residual correction | 34 | 55 | 9 | 522/584 | 573/584 |

The residual model gains 17 and loses six five-pixel successes relative to its research control under this reference. It gains one and loses three ten-pixel successes. Both complete-pair counts remain below the existing two-view method and supplied model. The earlier and reviewed annotations describe the same photographs; they are alternative measurement references, not independent datasets.

Changing evaluation labels measures annotation sensitivity. Establishing the effect of **training** annotation changes requires a controlled training comparison with the photograph set, initialization, training settings and evaluation reference held fixed. The original-versus-proposed system comparison also changes training and inference, so its complete gain cannot be assigned to annotation review alone.

## Training and audit

The return is PR155 at immutable head `244e553b2ea0ae6fe35d518c33da8998c2a31e5e`, using implementation `06e55d9c1b2557014aff7ff0d68b5096890fe546`. The fixed recipe completed 30 epochs on 3,809 eligible crops, with 114,270 crop presentations and 3,600 parameter updates. The recorded runtime is an NVIDIA L4, Python 3.13.16, Torch 2.14.1+cu130 and TorchVision 0.29.1+cu130. The final epoch was used.

Training loss falls from 1.560202 to 0.065675. This records optimization of the training objective; it does not establish improvement in the development five-pixel criterion. The training proposals came from a detector already trained on those photographs, which may make their errors easier than errors on unseen photographs.

The independent audit passes 697 source/data identities, all 24 compact archive members, all six wrapped checkpoint parts, 32 aggregate and 640 per-image score tables, 732 raw correction records, the training target masks, and all 30 sample-order hashes. The zero-output rule preserves every original raw detector record exactly. The recorded initial model state matches the earlier published bootstrap.

The final checkpoint was also run again on a CPU for all 71 raw proposals in the two diagnostic images. The maximum coordinate-component difference from the returned L4 predictions is **0.000024514 pixels**. All matching identities and five-/ten-pixel outcomes remain unchanged for both thresholds and all four technical annotation references. This is a selected inference replay using Torch 2.9.0+cpu; it does not independently repeat the training run.

## Evidence and decision

- [Audit and replay evidence](evidence/residual-return/README.md)
- [Fixed residual experiment protocol](residual-correction-experiment.md)
- [Original supplied-model comparison](original-comparison.md)
- [Earlier heatmap-refinement results](local-refinement-results.md)

Keep the completed residual result as a useful negative result for the selected five-pixel objective. The simpler two-view average remains the preferred reviewed-reference method among these comparisons. No deployment model or live setting is changed by this analysis.
