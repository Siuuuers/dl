# Current model and training entry

23 November 2026, Asia/Hong_Kong.

**PR35 remains selected. Run the r24 notebook for the next comparison.** The
completed [PR97 audit](../results/evaluation/pr97-r23-20261123/README.md)
evaluates the returned r23 models. [R24](front-review-r24.md) corrects 26
front-face corner points across the 21 named rows plus adjacent `012:5`,
preserves both accepted standards and prepares the next controlled run.

## Run inference

Use branch `codex/front-review-r24-20261123`. Follow the
[folder guide](predict-pose-folder.md#use-the-seed46-candidate-explicitly) to
obtain the verified PR35 bundle, retaining its manifest and companion files.

| Setting | Value |
| --- | --- |
| Checkpoint | PR35 seed 46 candidate |
| Checkpoint SHA-256 | `420c00f73eeba24e587342400878b7d96371d3e745772113809aa705ec8163b1` |
| Input size | 1280 |
| Box confidence | 0.13 |
| NMS IoU | 0.70 |
| Endpoint confidence | 0.50 |

## Latest completed comparison

On the frozen 666-front primary reference, PR35 / fresh r23 control / r23
candidate have **18 / 20 / 36 unmatched targets** and **80 / 83 / 110
unmatched outputs**. Both-corner successes within five pixels are
**533 / 531 / 521**, each out of 584 eligible fronts. All three return zero
outputs on the three empty scenes.

The candidate's ranking reverses on the separately pinned pre-return r23
exploratory review: PR35 / candidate have 625 / 645 matched targets and
545 / 556 both-corner successes within five pixels, out of 584. The audit
traces reference-box changes that alter IoU matching while the predictions
remain fixed. An unmatched target is a scoring outcome, not proof that a
physical cassette has no prediction. The conflicting rankings do not
establish a replacement. PR35 remains selected; no r24 training result is
claimed.

## Run r24

[Open the notebook in Colab](https://colab.research.google.com/github/Siuuuers/dl/blob/codex/front-review-r24-20261123/notebooks/baomai_front_review_r24.ipynb),
select GPU, choose **Run all**, and return **`baomai-front-review-r24-results.zip`**.

Both arms start independently from original epoch 173: seed 46, 30 epochs,
batch 4, input 1280, FP32, AdamW, learning rate 0.0001 and final-epoch checkpoint
selection. Both keep 137 training images including 13 empty scenes. Photo 015
and the 31 old-series treatment exclusions retain identical PR35 supervision
in both arms. The [handoff](front-review-r24.md) describes the new points,
source pins, staged-input verification and scoring references. `new_000010`
remains development-only; its amended review points never enter training.

## Next evidence after the run

The notebook evaluates accumulated eligible supervision. It does not isolate
the latest 26 point edits. Repeated development images do not establish
reliability in a new capture session; a separate independently labeled session
is the next useful generalization check. The present task remains RGB
detection and two 2-D corners. Pixel error alone does not establish robot
placement accuracy in millimetres.
