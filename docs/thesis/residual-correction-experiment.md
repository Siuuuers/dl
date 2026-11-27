# One-step residual corner correction

## Purpose

The two returned local heatmap models reduced five-pixel corner-pair success relative to their fixed research detector. On the reviewed reference, the detector achieved 566 successful pairs out of 583; the image-only and image-plus-hint models achieved 547 and 558. Their boxes and detection scores were unchanged. The [paired audit](local-refinement-results.md) explains the measurements and their limits.

The next experiment asks one question: **can a network improve available detector points by learning their correction while starting from exactly the detector's output?** It predicts an offset instead of finding a new heatmap maximum. A benefit remains unmeasured until the notebook completes.

This control is the project's fixed research detector, not the untouched original supplied baomai model. Use [the direct original-model comparison](../../notebooks/baomai_original_comparison_r59.ipynb) for the report's main comparison. The residual notebook also exports scores at the same strict box-confidence threshold greater than 0.25, but those scores alone do not run the original model.

## Model and coordinates

The input is the same aspect-preserving 256 × 256 colour crop around a predicted cassette box, expanded by a factor of 1.5. Two additional channels contain Gaussian maps at the available original corner estimates, with standard deviation eight canvas pixels. RGB normalization and crop geometry use the established implementation.

A ResNet50 initialized from the hash-verified ImageNet1K V2 checkpoint feeds global average pooling and a four-output linear layer. Both weights and biases of this last layer start at zero. The new input-channel weights also start at zero; the RGB weights retain their pretrained values. The model has 23,522,500 trainable parameters.

Let \(p_i\) be an original point in image coordinates, \(g_i\) its known reference, and \(s=(s_x,s_y)\) the actual resize scales after integer dimension rounding. The target and corrected estimate are

\[
\Delta_i=s\odot(g_i-p_i),\qquad
\widehat p_i=p_i+\widehat\Delta_i\oslash s.
\]

The multiplication and division act separately on the horizontal and vertical coordinates. The correction is applied directly to the original point. It does not map the original point into the canvas and back. Therefore exactly zero model output preserves the original coordinates, scores and records exactly. The four outputs are unconstrained real values; no correction bound is chosen from development errors.

The image-plus-estimate correction idea is motivated by Carreira *et al.* [1]. Their method iterates pose updates and uses a different training procedure. This experiment is a single-step adaptation with a zero-initialized output, ResNet50 and SmoothL1 loss. It does not reproduce their full method or inherit their reported performance.

## Supervision and fixed training recipe

Training uses the existing 142 photographs and 3,999 annotated cassette fronts. The frozen research detector supplies proposals. One-to-one box matches at intersection-over-union of at least 0.50 identify reference correspondences.

A corner contributes coordinate supervision only if its reference visibility is 2 and its original point is finite, lies inside the image, and has score at least 0.50. An unavailable reference gives no coordinate target. An unavailable original estimate gives no correction target and is preserved unchanged at inference. This experiment does not learn to recover unavailable points.

The loss is SmoothL1 with beta equal to one canvas pixel, averaged over the two coordinate components of each eligible endpoint. A known reference outside the predicted crop remains a valid unconstrained regression target and is counted. Unmatched training proposals have no coordinate target. Every raw development proposal is still processed, including unmatched and subsequently suppressed proposals.

Hint jitter is disabled in this first pilot. The hint centre, correction origin and target then refer to the same original point. This differs from the earlier hinted heatmap run, which perturbed hint locations during training. The residual pilot also changes the output head, loss and endpoint eligibility. It is a targeted alternative, not an isolated one-variable comparison with the heatmap model.

| Setting | Fixed value |
|---|---|
| Seed | 46 |
| Epochs and selection | 30; final epoch, no development-based selection |
| Batch size | 32, retaining the final partial batch |
| Arithmetic | FP32 |
| Optimizer | AdamW, learning rate 0.0001, weight decay 0.0001 |
| Schedule | Cosine decay to 0.000001 |
| Gradient clipping | Norm 10 |
| RGB augmentation and hint noise | None |
| Training ordering | Deterministic epoch order, saved as a hash |
| Detector boxes, scores and suppression | Preserved |

A nonfinite correction or a corrected point outside the 1280 × 720 image becomes unavailable. There is no clamping or fallback selected using the reference. Original point scores are retained for the comparison; they are not recalibrated confidence estimates for the new coordinates.

## Training target audit before a new run

The already returned, verified frozen training predictions give the following inventory. The new notebook regenerates these predictions after reproducing the development baseline and records its actual inventory again.

| Quantity | Count |
|---|---:|
| Training photos | 142 |
| Reference fronts | 3,999 |
| Detector proposals after suppression | 4,652 |
| Matched fronts | 3,883 |
| Missed reference fronts | 116 |
| Unmatched proposals | 769 |
| Eligible endpoint targets | 7,204 |
| Unavailable reference endpoints among matches | 554 |
| Known endpoints with unavailable original estimate | 8 |
| Matched crops without an eligible endpoint | 74 |
| Usable crops | 3,809 |
| Eligible reference endpoints outside their predicted crop | 1 |

The required image-coordinate correction has median magnitude 1.768 pixels, 95th percentile 5.473 pixels and maximum 26.350 pixels. In the resized canvas, these are 2.631, 7.556 and 44.679 pixels. These are training-target measurements, not predictions from a trained residual model. The corresponding fixed budget would be 114,270 crop presentations and 3,600 optimizer updates.

The detector has previously trained on these photographs. Their proposal errors may therefore be easier than errors on new photographs. This limits what a successful development result would establish. The pilot records this fact rather than treating the training residuals as independent data.

## Run and return

Open [the residual correction notebook](../../notebooks/baomai_residual_corner_r59.ipynb), select a GPU runtime and run all cells. Its implementation is pinned to an immutable published source commit and driver hash.

Published source: `06e55d9c1b2557014aff7ff0d68b5096890fe546`. Driver SHA256: `54cdf49281abdc5409c9a8828d1a3bb00c5f9da8610e8767645b69315a78e5d3`.

Before any experiment training, the notebook runs an actual pretrained CPU forward/backward check on two real training images. It verifies exactly zero initial outputs, exact original-coordinate preservation, a finite nonzero loss, an updated output layer and zero gradient for an unavailable endpoint. These disposable mechanics fixtures are discarded. The experiment model is reinitialized and must match the recorded initial state exactly.

The detector must then reproduce its archived development output within the established tolerances, including matching identities and five-/ten-pixel success states. A failed check produces a diagnostic return and blocks training. A passing run trains once, evaluates the final epoch, and packages predictions, coordinate traces, score tables, training-target distributions, epoch exposure/order, runtime identities and exact checkpoint parts.

The download is `baomai-residual-corner-r59-upload.zip`. For GitHub return, extract its outer layer and upload the compact result ZIP, `return-manifest.json` and all wrapped checkpoint parts together. Keep the part ZIPs wrapped. The verifier can recreate an interrupted outer download from verified members.

## Evaluation and decision

The primary success count uses every eligible reference front, including missed detections, and requires both known lower corners within five image pixels. Ten-pixel success, missed fronts, unmatched and duplicate outputs, and offered-pair accounting remain visible. Box matching is fixed before offered pairs are classified. Both the established confidence threshold and the common strict threshold greater than 0.25 are exported.

Because boxes and suppression are fixed, this model cannot recover a missed front or remove a duplicate detection. The experiment addresses coordinate errors on available original points. Examine both improvements and regressions for the same reference rows, not only the total success count. A decrease rejects this configuration; a development gain motivates a separate capture-session evaluation. No checkpoint is adopted automatically.

Exceeding the unrefined control's 566/583 reviewed five-pixel successes is insufficient to prefer the added model: it should also improve on the existing coordinate average's 569/583 at the same common confidence threshold, with individual regressions, ten-pixel success and offered-pair outcomes examined.

Implementation checks are in [the evidence folder](evidence/residual/README.md). The actual notebook source and data-preparation cells, followed by the pretrained CPU mechanics check, passed from the published source with all 697 source/data files verified. No full residual training run is claimed by those checks.

## Reference

[1] J. Carreira, P. Agrawal, K. Fragkiadaki and J. Malik, “Human Pose Estimation with Iterative Error Feedback,” in *Proc. IEEE Conf. Computer Vision and Pattern Recognition*, 2016, pp. 4733–4742. [Official paper](https://www.cv-foundation.org/openaccess/content_cvpr_2016/html/Carreira_Human_Pose_Estimation_CVPR_2016_paper.html).
