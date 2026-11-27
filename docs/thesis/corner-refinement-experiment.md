# Local corner refinement: fixed pilot and hint ablation

## 1. Question and current status

Can a learned model improve the two lower-corner coordinates when the cassette detector's complete set of boxes is held fixed? A second comparison asks whether the detector's initial corner estimates help that local model.

The runnable experiment consists of two notebooks:

- `notebooks/baomai_corner_refinement_rgb_r58.ipynb`: the local RGB crop with two zero hint channels.
- `notebooks/baomai_corner_refinement_hints_r58.ipynb`: the same RGB crop and network, with two Gaussian maps describing available initial corners.

The implementation has passed an actual pretrained CPU forward/backward check on two verified training photographs, an optimizer update, an unavailable-corner gradient check, and focused geometry/audit tests. **Full GPU training and recognition results are pending.** These checks establish that the implemented mechanism runs; they do not establish an accuracy improvement.

This experiment preserves detector boxes, box confidences, original corner scores and postclip suppression identities. It can improve or worsen coordinate accuracy. It cannot recover a cassette for which the fixed detector produced no box, and it does not remove duplicate boxes. The unchanged detector provides the control inside each notebook.

## 2. Research basis and deliberate adaptation

Xiao, Wu and Wei describe a simple pose-estimation architecture that adds three deconvolution stages to a residual backbone to predict keypoint heatmaps [1]. Moon, Chang and Lee show how an image and an initial pose can jointly inform a refinement network [2]. These papers motivate a local heatmap model and the initial-estimate ablation; they do not establish that either will improve cassette recognition.

The implementation uses TorchVision's ResNet50 V1.5 architecture and its explicit ImageNet1K V2 initializer [3]. It adapts the heatmap architecture to two cassette corners. It does not reproduce PoseFix's complete architecture, synthetic error distribution, supervision or human-pose benchmark recipe. The two input-information conditions provide a simpler controlled first experiment.

| Stage | Output for one crop |
|---|---|
| RGB image and two hint channels | 5 × 256 × 256 |
| ResNet50 backbone | 2048 × 8 × 8 |
| Three learned upsampling stages | 256 × 16 × 16, then 256 × 32 × 32, then 256 × 64 × 64 |
| Final 1 × 1 convolution | 2 × 64 × 64 heatmaps |

Each upsampling stage uses a 4 × 4 transposed convolution, stride 2, padding 1, batch normalization and a rectified linear activation. The first convolution copies the pretrained RGB weights and initializes both hint-channel weights to zero. All 34,002,114 model parameters are trainable. The two arms therefore start from exactly the same function and parameter state. Their initial-state tensor hash and full tensor-shape inventory are recorded.

The external initializer has 102,540,417 bytes and SHA256:

```
11ad3fa62ca79e40addfd354a8ec4b7c75143b3038b8d2a807fbc68deab379ca
```

Its official URL is `https://download.pytorch.org/models/resnet50-11ad3fa6.pth`. The complete hash and length are checked before loading. RGB is normalized using the initializer's documented ImageNet mean and standard deviation [3].

## 3. Fixed data and proposal construction

The existing audited annotation release supplies 142 training photographs, 20 development photographs and 3,999 training cassette fronts. Whole photographs retain their existing membership. A crop is never randomly moved between the training and development sets. Independently verified capture-group labels are unavailable; the 20 photographs remain a development evaluation rather than an independent capture test.

The unchanged detector first runs on all 20 development photographs. Its outputs must reproduce the cached control within 0.05 original-image pixel in coordinates and 0.00001 in scores. Matched/missed/unmatched/duplicate identities and every 5- and 10-pixel success state must also remain unchanged. A tolerance does not permit a changed conclusion. Failure produces a diagnostic download and prevents experiment training.

After a successful baseline gate, the same frozen detector runs on all training photographs. Predictions after the existing postclip suppression are matched one-to-one to annotated boxes at intersection-over-union of at least 0.50. Matched proposals provide coordinate targets only for reference corners marked available and lying inside the fixed predicted crop. A known corner outside the crop is counted separately; its coordinate is not forced into the crop. Unavailable corners are masked, not replaced with a coordinate target at zero. Proposals without a matched annotated object do not supply coordinate supervision.

The crop manifest records matched, missed and unmatched training proposals, each omitted endpoint, every usable crop, and the number of usable crops per training photograph. All raw detector records, including records subsequently suppressed, are passed through the same trained refiner during evaluation. Their boxes and original scores are preserved exactly.

## 4. Coordinates, hints and padding

A predicted box is expanded by 25% of its width and height on each side, making the intended crop 1.5 times the predicted dimensions. Bounds are rounded outward to integer image edges and clipped to the original image. The crop is resized with bilinear interpolation while preserving its aspect ratio, then padded into a 256 × 256 canvas. The actual rounded resized width and height determine separate horizontal and vertical scale factors.

For an original-image coordinate `(x, y)`, the forward mapping is:

```
u = (x − crop_left) × scale_x + pad_left
v = (y − crop_top)  × scale_y + pad_top
```

The inverse uses the same actual scale factors, so rounding the resized dimensions does not introduce a second coordinate convention. A heatmap cell with zero-based column `j` and row `i` represents canvas position `(4(j + 0.5), 4(i + 0.5))`. Gaussian targets have a standard deviation of two heatmap cells and are centred at `(u/4 − 0.5, v/4 − 0.5)`.

Only heatmap-cell centres inside resized image content contribute to loss or decoding. Padded cells cannot win the decoder even if their model output is large. The decoder selects the largest positive valid-content response, then moves by one quarter of a heatmap cell toward the larger neighbouring response on each eligible axis, following the simple offset approach associated with the heatmap baseline [1]. It then applies the declared inverse mapping. A nonfinite heatmap, no valid content, a nonpositive maximum or an invalid decoded coordinate produces an unavailable coordinate. No ground-truth fallback is used.

The hint arm uses initial detector corners only when their original score is at least 0.50 and the coordinate is finite and inside the original image. Each available corner supplies a Gaussian map with an eight-canvas-pixel standard deviation. During training, its centre receives independent Gaussian perturbation with a four-canvas-pixel standard deviation. A local generator keyed by the fixed seed, epoch and crop identity supplies this noise in both arms; the RGB-only arm still draws the same noise but supplies zero hint channels. Development inference adds no perturbation. The RGB arrays and sample order remain identical between arms.

## 5. Fixed training recipe

| Item | Prespecified setting |
|---|---|
| Random seed | 46 |
| Epochs and checkpoint | 30; use the final epoch |
| Batch size | 32; retain the final smaller batch |
| Arithmetic | FP32 |
| Optimizer | AdamW; learning rate 0.0001; weight decay 0.0001 |
| Learning-rate schedule | Cosine decay over 30 epochs to 0.000001 |
| Loss | Squared heatmap error, averaged over supervised endpoint/content cells |
| Gradient clipping | Norm 10 |
| RGB augmentation | None |
| Sample order | New deterministic order from seed `46 + epoch` |
| Selection from development results | None |

There is no early stopping, best-epoch selection, threshold sweep, crop-size sweep or architecture selection inside these notebooks. The epoch logs bind the exact crop order, samples and optimizer updates. An unavailable endpoint contributes exactly zero coordinate-loss gradient; padded heatmap cells also contribute zero gradient.

The actual-model CPU check uses annotation-box crops from two training images solely to exercise model mechanics. Its temporary optimization step is discarded. Training resets the seed, reloads the exact pretrained initializer, recreates the model, and verifies that its initial-state hash equals the pre-update smoke-test hash.

## 6. What the scores mean

The primary coordinate measure is the number of reference objects with two known corners for which both predicted corners fall within five original-image pixels. Every eligible reference object remains in the denominator, including missed detections and unavailable predicted points. The exported reports also include the 10-pixel criterion, one-known-corner objects, detection counts, image-size strata, and per-image outcomes on all four existing references. Those references are alternative annotations of the same photographs, not four independent test datasets.

The original point scores are retained as a controlled threshold input. **They are not calibrated probabilities of correctness for the refined coordinates.** Refinement can make a previously unavailable coordinate valid, or make a valid coordinate unavailable; the accepted-pair identities can therefore change even though scores and boxes remain fixed.

Offered-pair accounting first matches the full fixed set of boxes, then classifies available pairs under those same assignments as within five pixels, known coordinate failures, matched objects with incomplete reference corners, or unmatched proposals. It does not rematch only the accepted subset. This convention is stated in the artifact because an accepted-subset metric answers a different question.

The two trained arms can be compared as a hint ablation only after their source, training data, frozen proposals, initial-state hashes, runtime versions, GPU type, sample-order hashes and epoch exposure agree. A single seeded development run does not establish statistical robustness. A promising result should be repeated with prespecified additional seeds and evaluated on independent capture groups before automatic use.

One relevant limitation is that the frozen detector's training proposals are in-sample predictions on its own training images. Their error distribution may be easier than proposals on unseen photographs. Fixed hint noise is a simple first perturbation, not an empirical cassette-error model. If the pilot is promising, generating training proposals with detectors that excluded each training fold is a useful controlled follow-up that can use the existing photographs.

## 7. Running and returning the result

Choose a GPU Colab runtime and run all four code cells. The notebooks keep the runtime's compatible Torch/TorchVision pair and pin Ultralytics 8.4.182, OpenCV 5.0.0.93, NumPy 2.4.6 and Pillow 13.0.0. Native inference sources are checked by hash. Use the same Torch/TorchVision versions and GPU type for the two arms. Hardware or memory failures do not automatically change the recipe.

Each run downloads one `baomai-corner-refinement-r58-<hint-mode>-upload.zip`. Keep and return that complete file. It contains:

- A compact evidence ZIP with inputs, source identities, frozen predictions, refined predictions, crop/decoder traces, training logs, scores, case transitions and an inventory.
- `return-manifest.json`, describing the exact final model and its parts.
- Small wrapped parts of the exact FP32 final checkpoint. The manifest specifies their order, individual hashes and the complete reconstructed model hash.

The checkpoint is split for practical repository storage; no precision reduction or model conversion is applied. Reassembly concatenates the single member from each wrapper in manifest order, checks the complete SHA256, then loads the recorded state dictionary into `corner_refinement_r58.build_model()`.

CPU artifact verification reconstructs postclip suppression, training supervision, per-epoch sample order, exposure and update counts, all score tables and case identities. For each endpoint it replays the recorded local-neighbour offset, heatmap-to-canvas mapping and inverse crop mapping. This arithmetic check does not recompute every heatmap or prove its global maximum without rerunning the saved model. Interrupted outer packaging is rebuilt from verified inner files. Resume requires matching package sources, Python/Torch/TorchVision versions, device and GPU identity.

The generator is `scripts/build_corner_refinement_r58_notebooks.py`. It requires the actual published implementation commit; placeholder source pins are not runnable releases. The complete implementation is in `scripts/corner_refinement_r58.py` and `scripts/run_corner_refinement_r58.py`. Focused checks are in `tests/test_corner_refinement_r58.py`.

## 8. References

[1] B. Xiao, H. Wu, and Y. Wei, “Simple baselines for human pose estimation and tracking,” in *Proc. European Conf. Computer Vision*, 2018, pp. 466–481. [Author manuscript](https://arxiv.org/abs/1804.06208).

[2] G. Moon, J. Y. Chang, and K. M. Lee, “PoseFix: Model-agnostic general human pose refinement network,” in *Proc. IEEE/CVF Conf. Computer Vision and Pattern Recognition*, 2019, pp. 7773–7781. [Publisher paper](https://openaccess.thecvf.com/content_CVPR_2019/html/Moon_PoseFix_Model-Agnostic_General_Human_Pose_Refinement_Network_CVPR_2019_paper.html).

[3] TorchVision, “resnet50,” official model documentation. [Documentation](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet50.html).
