# Current model and training entry

22 November 2026, Asia/Hong_Kong.

**PR35 remains selected. Run the r23 notebook for the next comparison.** The completed [PR96 audit](../results/evaluation/pr96-r21-20261122/README.md) evaluates the returned r21 models. [R22](front-review-r22.md) preserves the completed annotation review and full PDF. [R23](front-review-r23.md) adds four right-point corrections at the actual lower front-face corners and prepares the next controlled run.

## Run inference

Use branch `codex/front-review-r23-20261122`. Follow the [folder guide](predict-pose-folder.md#use-the-seed46-candidate-explicitly) to obtain the verified PR35 bundle, retaining its manifest and companion files.

| Setting | Value |
| --- | --- |
| Checkpoint | PR35 seed 46 candidate |
| Checkpoint SHA-256 | `420c00f73eeba24e587342400878b7d96371d3e745772113809aa705ec8163b1` |
| Input size | 1280 |
| Box confidence | 0.13 |
| NMS IoU | 0.70 |
| Endpoint confidence | 0.50 |

## Latest completed comparison

On the frozen 666-front primary reference, PR35 / fresh r21 control / r21 candidate have **18 / 20 / 35 unmatched targets** and **80 / 83 / 100 unmatched outputs**. Both-corner successes within five pixels are **533 / 531 / 518**, each out of 584 eligible fronts. All three return zero outputs on the three empty scenes.

The candidate's ranking changes on its separately pinned r21 review. The completed audit documents how altered reference boxes change IoU matching, as well as residual rack/occluder errors. The fixed-reference selection remains PR35. No r22 or r23 training result is claimed.

## Run r23

[Open the notebook in Colab](https://colab.research.google.com/github/Siuuuers/dl/blob/codex/front-review-r23-20261122/notebooks/baomai_front_review_r23.ipynb), select GPU, choose **Run all**, and return **`baomai-front-review-r23-results.zip`**.

Both arms start independently from original epoch 173: seed 46, 30 epochs, batch 4, input 1280, FP32, AdamW, learning rate 0.0001 and final-epoch checkpoint selection. Both keep 137 training images including 13 empty scenes. Photo 015 and the 31 old-series treatment exclusions retain identical PR35 supervision in both arms. The [handoff](front-review-r23.md) describes the four new points, exact source pins, staged-input verification and fixed scoring references.

## Next evidence after the run

Use the completed controlled comparison to decide whether to retain or replace PR35. Repeated development images do not establish reliability in a new capture session; a separate independently labeled session is the next useful generalization check. The present task remains RGB detection and two 2-D corners. Pixel error alone does not establish robot placement accuracy in millimetres.
