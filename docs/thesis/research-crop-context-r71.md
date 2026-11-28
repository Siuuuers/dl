# Crop context and scale for the remaining partial fronts

## Decision before the experiment

The next small experiment keeps the reviewed-label detector fixed and adds two overlapping crops of each photograph. One arm uses the original pixel scale. The other enlarges the same crops. This tests whether the existing model can produce useful detections from a different presentation of the same image before committing to more training.

The [fixed protocol](evidence/crop-context-r71/protocol.json) was written before crop inference. The implementation records the source of every output, rejects boxes at artificial crop cuts and checks both detection and corner outcomes. The comparison will be evaluated under four annotation versions, with the extended 658-front reference as the primary development reference.

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

## How the result will guide a decision

Fresh matching must report misses, extras, the duplicate subset, complete pairs within five and ten pixels, known-corner errors for partial fronts, and empty-photograph outputs. Each recovered or newly missed front retains its source identity. A lower miss count alone is insufficient if the added inference damages accurate corners or produces many extra outputs.

The fresh full-image control must reproduce the earlier CPU matching and corner outcomes before crop inference starts. The saved crop evidence covers retained native outputs, their pre-clipping coordinates and every filter and merge decision. It does not inventory low-confidence decoded candidates.

The [earlier experiment review](research-next-models.md) already covers partial-image sampling, partial box and classification weighting, synthetic occlusion, larger whole-image inference, Soft-NMS and a one-to-one detection head. R71 asks a different bounded question. Its outcomes will decide whether crops deserve further work, with no automatic change to the selected method.

## References

1. F. C. Akyon, S. O. Altinuc and A. Temizel, “Slicing aided hyper inference and fine-tuning for small object detection,” ICIP, 2022, pp. 966–970, doi: 10.1109/ICIP46576.2022.9897990.
2. M. Najibi, B. Singh and L. S. Davis, “AutoFocus: Efficient multi-scale inference,” ICCV, 2019, pp. 9745–9755 in the CVF open-access proceedings. Section 4.3 provides the crop-boundary rule used here.
3. B. Singh, M. Najibi and L. S. Davis, “SNIPER: Efficient multi-scale training,” NeurIPS, 2018. Sections 3.3 and 4.2 explain negative-chip selection and its comparison.
4. X. Yu, Y. Gong, N. Jiang, Q. Ye and Z. Han, “Scale match for tiny person detection,” WACV, 2020.
5. N. Dvornik, J. Mairal and C. Schmid, “Modeling visual context is key to augmenting object detection datasets,” ECCV, 2018.
