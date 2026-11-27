# Local corner refinement results

28 November 2026, Asia/Hong_Kong.

Both returned training runs completed and passed the source, data, training and result audits. **Neither improves corner accuracy over the frozen research detector.** Initial-corner hints help relative to the image-only refiner, but do not recover the original detector's five-pixel result. The supplied original baomai model is a separate baseline; the control in these training notebooks is the later expanded-data research detector.

## Main result

The primary notebook setting is box confidence strictly above 0.13, box matching at intersection-over-union 0.50, and usable predicted corners at score 0.50. All complete reference targets stay in the denominator, including any missed front or unavailable returned pair.

| Reference | Coordinates | Both corners within 5 px | Both within 10 px | Mean maximum corner error among matched usable pairs |
| --- | --- | ---: | ---: | ---: |
| Fixed, 584 complete targets | Detector control | 511 | 575 | 2.788 px |
| Fixed | Image-only refiner | 501 | 571 | 3.117 px |
| Fixed | Image and initial-corner hints | 505 | 570 | 3.071 px |
| Reviewed, 583 complete targets | Detector control | 566 | 582 | 2.317 px |
| Reviewed | Image-only refiner | 547 | 582 | 2.622 px |
| Reviewed | Image and initial-corner hints | 558 | 581 | 2.590 px |

The fixed reference has 577 matched usable pairs; the reviewed reference has 583. Error means use those matched pairs, whereas the success denominators retain all complete targets. The mean error also worsens, so the conclusion does not arise solely from crossing one threshold.

Under the reviewed annotations, the image-only refiner gains 10 five-pixel successes and loses 29. The hint refiner gains 12 and loses 20. Under the fixed annotations, the corresponding gains/losses are 22/32 and 21/27. Every original detection box and confidence remains unchanged. Fixed-reference misses/extras/duplicates stay 34/91/17; reviewed-reference counts stay 15/84/13.

## Common comparison setting

For comparison with archived supplied-model predictions, the returned records were also rescored at box confidence **strictly above 0.25**, with the same annotations and additional suppression. Filtering the saved outputs agrees exactly with applying that suppression to the filtered raw outputs, apart from regenerated record numbering.

At this setting, all three research arms retain 687 detections. Fixed-reference counts are 34 misses, 55 extra outputs and nine duplicates. Reviewed-reference counts are 18 misses, 51 extras and five duplicates. All five- and ten-pixel pair-success counts in the table above are unchanged. The [paired evidence](evidence/refinement/paired-analysis.json) records these common-setting results separately. They can be placed beside the supplied original model only with the same image set, annotations and scoring convention.

## The two runs form a valid hint comparison

The runs have byte-identical staged data, frozen training predictions, 3,815 usable training crops, development control predictions and runtime records. Their initial model states and all 30 sample-order hashes agree. Each run processes 114,450 crop examples and performs 3,600 optimizer updates. Both use a Tesla T4, Torch 2.14.1 with CUDA 13.0, TorchVision 0.29.1, and the same native inference packages.

The frozen detector matches 3,883 of 3,999 training fronts. There are 116 unmatched reference fronts, 769 unmatched predictions, and 68 matched crops without usable corner supervision. The heatmap training supplies 7,211 supervised endpoints; one otherwise known endpoint lies outside its predicted crop. These proposal measurements explain the actual examples used by the refiner, rather than assuming that all annotated fronts enter training.

Training loss decreases substantially in both runs. That observation establishes optimization progress on the training crops, not an improvement on the development photographs. The detector generated those training proposals on its own training photographs, so their error distribution may differ from that of later images.

## Decoder capacity check

To test whether the heatmap grid itself prevents five-pixel accuracy, ideal Gaussian heatmaps were constructed at the annotated corner positions and passed through the exact crop transform and decoder. This is a controlled numerical simulation, not a model prediction.

The simulation uses 256 by 256 input canvases, 64 by 64 response grids, Gaussian standard deviation of two grid cells, the fixed quarter-cell neighbour offset, and the actual rounded resize scales. All 583 reviewed matched pairs and all 577 fixed-reference matched pairs decode within five pixels. The largest endpoint error is **1.5004 px** under either reference. Reviewed mean endpoint error is **0.4710 px**; mean maximum pair error is **0.5753 px**.

Thus the representation and decoder can express sufficiently accurate positions for these matched targets. This does not prove that a different decoder would never help. It does rule out the grid spacing alone as a necessary explanation for the observed failures above 20 pixels. Complete simulation data are in [ideal-heatmap-diagnostic.json](evidence/refinement/ideal-heatmap-diagnostic.json).

## Selected adverse cases

For reviewed case `new_000008/R028`, the detector's maximum corner error is 4.530 px. It becomes 21.591 px with the image-only refiner and 23.433 px with hints. The image-only right-corner estimate moves about 22 pixels left of the annotated outer corner, close to a junction with the adjacent cassette. This is a visual interpretation of this selected case, not a measured explanation for every failure.

For reviewed case `new_000053/R001`, the detector's maximum corner error is 2.324 px, increasing to 9.038 px with the image-only refiner and 10.309 px with hints. The learned right-corner response lies below the reference corner.

The exact saved models were reconstructed from all nine wrapped parts. CPU inference with those models on the two verified original images reproduces the returned T4 corner coordinates exactly in both arms. This small diagnostic supports the reality of these learned spatial errors. It does not substitute for full inference reproduction in the original T4 runtime.

## Consequence and next hypothesis

Keep the detector's original coordinates as the control. A focused next experiment can learn a correction added to each existing coordinate, with a zero-initialized output layer that initially preserves the detector exactly. This tests whether learning residual corrections is more useful than estimating fresh heatmap maxima. It remains a hypothesis until the next controlled run returns.

The full [audit evidence index](evidence/refinement/README.md) contains exact source, archive, checkpoint, training and replay records. All annotation references concern the same 20 development photographs.
