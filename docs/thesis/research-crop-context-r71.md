# Crop context and scale for the remaining partial fronts

## Completed finding and decision

Keep full-image inference. The fixed crop comparison recovered few additional fronts and worsened corner outcomes. Enlargement also produced many extra outputs. All four actual notebook cells completed locally on CPU, and an independent implementation reconstructed the recorded outcomes.

The [fixed protocol](evidence/crop-context-r71/protocol.json) and source were published at `5da7aa4f903dceb1e8300234ff98aebbdff06c17` before crop inference. The [four-cell notebook](../../notebooks/baomai_crop_context_r71.ipynb) uses that immutable source. The [complete evidence](evidence/crop-context-r71/README.md) includes the predictions and decision records.

| Method on the 658-front reference | Misses | Extras | Duplicates | Both corners within 5 px | Both within 10 px |
|---|---:|---:|---:|---:|---:|
| Fresh full-image two-view control | 18 | 50 | 14 | 572/583 | 583/583 |
| Add crops at original scale | 17 | 51 | 11 | 572/583 | 581/583 |
| Add enlarged crops | 17 | 106 | 44 | 566/583 | 581/583 |

Original-scale crops recover one one-corner front and lose no box matches. One five-pixel pair gain and one loss cancel, while two pairs lose ten-pixel success. Enlarged crops recover two one-corner fronts and newly miss one. Four five-pixel gains and ten losses produce a net loss of six. All arms return zero outputs on the three empty photographs.

The recovered fronts were previously lost at the confidence filter. Neither crop arm recovers any of the five original-view geometry failures or the one original-view NMS loss. The three restored focus misses also remain. This links the recovery to the earlier stage ledger. The retained-output archive does not identify where unsuccessful crop candidates were lost before the final outputs.

### Why the corner losses matter

At `new_000008/R010`, a crop record directly suppresses the baseline record. Its detection score increases while its worst corner error grows:

| Selected record | Detection score | Maximum corner error |
|---|---:|---:|
| Full-image control | 0.8288 | 8.877 px |
| Crop at original scale | 0.8396 | 12.542 px |
| Enlarged crop | 0.9065 | 13.422 px |

The four ten-pixel loss records across the two arms all have valid coordinates before clipping and accepted coordinates after mapping. Their losses therefore arise from the selected predictions, not from invalidating crop points. This is a concrete ordering failure, not a general measurement of score calibration.

The recovered partial front `new_000008/R007` has a known-corner error of 8.402 pixels with original-scale crops and 19.882 pixels with enlarged crops. Enlarged crops also recover `new_000008/R034` with a 2.233-pixel known-corner error, but lose `new_000089/R048`, whose earlier known corner was within 1.997 pixels. Detection recovery and usable corner recovery need to be judged together.

The [full four-reference matrix](evidence/crop-context-r71/results.md) also retains the adverse outcomes under earlier annotation versions. They describe sensitivity on the same photographs. The fresh CPU baseline has 14 duplicates, compared with 13 in the earlier L4 archive, and exactly reproduces the earlier CPU records. All treatment comparisons use the same fresh CPU baseline.

## What the current evidence makes worth testing

The [current-checkpoint proposal trace](evidence/current-proposals-r69/README.md) assigns the reviewed-label detector's 18 original-view misses to 12 confidence failures, five box-geometry failures and one native suppression loss. All five geometry failures are small, with reference boxes 15–31 pixels wide and 12–18 pixels tall. Cropping and enlargement therefore address a concrete observed group.

Size cannot explain every miss. Two confidence failures have reference boxes approximately 149 × 33 and 128 × 36 pixels. A model can also respond to surrounding material, local texture or a fragment's appearance. The crop comparison measures its response without claiming that any one of these factors caused the existing errors.

The [training-size audit](evidence/crop-context-r71/training-size-audit.json) independently checks all 137 reviewed training-label files against the PR158 archive before counting:

| Training selection | Photographs | Fronts | Two known corners | One known corner | No known corners |
|---|---:|---:|---:|---:|---:|
| Reviewed PR158 training set | 137 | 3,784 | 3,190 | 457 | 137 |
| Five supplementary photographs | 5 | 215 | 207 | 8 | 0 |
| Combined set | 142 | 3,999 | 3,397 | 465 | 137 |

The supplementary photographs mainly add complete fronts. They are a weakly targeted response to the remaining partial-front misses. Exact size counts are also descriptive. Although no one-corner training box is no larger in both dimensions than the 15 × 14 focus target, 36 fall within a factor of two of its width and height. There are 43 such neighbors for the 47 × 10 target. These counts do not establish appearance coverage or the distribution actually presented after augmentation.

The earlier assignment replay over 3,999 fronts belongs to the PR157 native/QFL models. It does not measure assignments for this PR158 checkpoint, which was trained on 137 photographs.

## Research that informs the design

### Overlapping inference crops

Akyon, Altinuc and Temizel's [SAHI paper](https://arxiv.org/abs/2202.06934v5) combines detections from overlapping, resized image slices. Its immediate value here is a simple inference comparison. The cassette implementation keeps the native pose records so each box retains both corner coordinates and scores. It uses fixed full-height windows and the project's existing merge rule. It does not reproduce SAHI's complete training and inference system.

### Rejecting fragments made by a crop

Najibi, Singh and Davis explain in [AutoFocus, Section 4.3](https://openaccess.thecvf.com/content_ICCV_2019/papers/Najibi_AutoFocus_Efficient_Multi-Scale_Inference_ICCV_2019_paper.pdf) that cutting a large object at a chip boundary can create a false small-object detection. Their stacking procedure rejects detections at artificial boundaries while allowing genuine image boundaries. R71 applies that distinction with a fixed one-pixel margin. AutoFocus also learns where to crop and constructs chips around dilated regions. Those components and their coverage properties are not part of this fixed-window experiment.

### Background examples matter if crop training follows

In [SNIPER, Section 4.2 and Table 2](https://proceedings.neurips.cc/paper_files/paper/2018/file/166cee72e93a992007a89b39eb29628b-Paper.pdf), adding mined negative chips raises COCO AP from 43.4 to 46.1. Its proposal recall comparison does not reveal that false-positive cost. For this project, the lesson is to retain difficult background if crop training is later justified. A training set made only from successful cassette crops could weaken rejection of rack or gripper material. SNIPER's reported numbers concern its own detector and datasets.

### Compare distributions before prescribing a size change

Yu et al.'s [Scale Match](https://arxiv.org/pdf/1912.10664) aligns object-size distributions between an additional training dataset and the target task. Its monotone variant preserves size ordering. This motivates inspecting the cassette size distribution, but does not establish that a tiny whole object and a tiny visible fragment are equivalent training examples. The present audit records nominal boxes. It does not justify a new resampling distribution by itself.

### Keep synthetic context plausible

Dvornik, Mairal and Schmid's [context augmentation study](https://arxiv.org/pdf/1807.07428) uses segmentation masks and context-guided placement. In its single-category comparisons, mean AP is 58.0 for the baseline, 56.9 for random placement and 62.0 for context-guided placement. Its multi-category gain is smaller. The useful constraint is that plausible placement matters. Copying a rectangular cassette box would also copy surrounding rack material or neighboring fronts. A new compositing system is not justified by the current evidence.

## Fixed comparison

Each 1,280 × 720 photograph supplies the following inputs:

| Arm | Added windows | Network input | Question |
|---|---|---|---|
| Full-image control | None | Original and reflected full image at 1,280 | Reproduce the current two-view output |
| Crops at original scale | x = 0–768 and x = 512–1,280, both full height | 768, giving 736 × 768 tensors after padding | Does changing context and adding candidates help? |
| Enlarged crops | The same windows | 1,280, giving 1,216 × 1,280 tensors | Does enlargement add useful recovery under this crop procedure? |

The windows overlap by 256 pixels. Full height preserves the genuine top and bottom image boundaries. After the windows were selected, a containment check found that 410 reference fronts fit within one crop and 248 within both, with more than one pixel of clearance from an artificial cut. No reference front falls outside both usable crop interiors. This checks geometric feasibility, not detection success.

The model, confidence threshold, native suppression and corner acceptance threshold remain those of the control. Crop predictions are mapped back with the actual scale and padding recorded by inference. Points outside the crop before native clipping become unavailable. This is stricter at genuine photograph edges than the baseline's native clipping, so every affected point is retained in the audit. The final pool contains complete records from the baseline, left crop and right crop in that order. The existing score-ranked suppression rule uses box IoU above 0.70. A higher-scoring crop prediction can replace a baseline prediction, so the comparison records losses as well as recoveries.

The first arm changes context and the number of candidates available to merging. It is not a pure measurement of context alone. Comparing the two crop arms measures resizing within this procedure, including interpolation, changed tensor and candidate counts, and padding. A difference does not by itself identify the cause of a baseline miss. Both reuse the same pixels and provide no new optical observation.

## What follows from the comparison

Fresh matching reports misses, extras, the duplicate subset, complete pairs within five and ten pixels, known-corner errors for partial fronts, and empty-photograph outputs. Each recovered or newly missed front retains its source identity. Both candidates fail to preserve the control's corner accuracy. Neither earns the added inference cost.

The fresh full-image control reproduced every earlier CPU record before crop inference started. The saved crop evidence covers retained native outputs, their pre-clipping coordinates and every filter and merge decision. It does not inventory low-confidence decoded candidates.

The [earlier experiment review](research-next-models.md) already covers partial-image sampling, partial box and classification weighting, synthetic occlusion, larger whole-image inference, Soft-NMS and a one-to-one detection head. R71 asks a different bounded question and now supplies a measured reason to retain full-image inference at these settings. Another crop or threshold sweep is a lower priority than representative real partial-front and difficult-background examples.

A useful next data comparison would add examples from new capture sessions while holding optimizer updates and total image presentations fixed. Preserve a separate capture group for evaluation. Include small complete fronts, identifiable partial fronts with zero or one locatable corner, and visually confusing background. Annotate each actual visible corner using the written definition. The sampling proportions should follow the captured failure distribution, rather than being chosen to fit these 20 photographs.

## References

1. F. C. Akyon, S. O. Altinuc and A. Temizel, “Slicing aided hyper inference and fine-tuning for small object detection,” ICIP, 2022, pp. 966–970, doi: 10.1109/ICIP46576.2022.9897990.
2. M. Najibi, B. Singh and L. S. Davis, “AutoFocus: Efficient multi-scale inference,” ICCV, 2019, pp. 9745–9755 in the CVF open-access proceedings. Section 4.3 provides the crop-boundary rule used here.
3. B. Singh, M. Najibi and L. S. Davis, “SNIPER: Efficient multi-scale training,” NeurIPS, 2018. Sections 3.3 and 4.2 explain negative-chip selection and its comparison.
4. X. Yu, Y. Gong, N. Jiang, Q. Ye and Z. Han, “Scale match for tiny person detection,” WACV, 2020.
5. N. Dvornik, J. Mairal and C. Schmid, “Modeling visual context is key to augmenting object detection datasets,” ECCV, 2018.
