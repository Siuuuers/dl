# Decide whether the two-corner model is good enough

**Current scope, clarified by the user on 2026-11-07 (Asia/Hong_Kong): find/localize cassette objects in the supplied baomai D405 images.** Keep the existing two-lower-corner pose target and use baomai as the foundation. Camera mounting, robot/gripper integration and physical calibration are outside this deliverable and do not block image evaluation. Continue from the [additional annotation pilot](additional-annotation-pilot-2026-11-07.md). The supplied epoch173 model remains the working default under the predeclared replacement rule.

**Latest model comparison, 2026-11-07:** the audited [PR #16](https://github.com/Siuuuers/dl/pull/16) warmup model improves all five primary measures against M0, but its 37 missed objects exceed the fresh control's 36. It therefore fails the required comparison against both controls. Retain epoch173 and preserve warmup as a promising candidate. The [warmup report](warmup-comparison-2026-11-07.md) records that result; the [first label comparison](gpu-comparison-2026-11-07.md) remains historical evidence.

Following the [completed case review and intake](warmup-detection-review-2026-11-07.md), the [annotation pilot](additional-annotation-pilot-2026-11-07.md) now provides a [three-image, 129-row candidate](../data/candidates/additional-pilot-2026-11-07/manifest.json). The candidate changes 87 boxes and five corner positions; the corner edits affect four rows. It adds or deletes no objects and changes no visibility flags. All three images are quarantined for 36 unresolved rows: 35 upper-plane bounds and the left box extent of `frame_0001` row 43. There are no eligible scoring images, and the other 317 recovered images have not completed row review. The pilot used no inference or training; prior upstream use and frame correlation remain limits on any later supplementary evaluation.

The [training pilot](training-pilot-2026-11-06.md) is complete and supplies an eleven-row deduplication proposal. The [annotation policy](annotation-policy.md) is adopted as a local working convention, informed by upstream branch/history research and the user's authorization. Complete reference/candidate datasets can be built and checked using the [training preparation guide](prepare-pose-training.md). The [training runner](train-pose-comparison.md) records bounded resource trials and completed-arm provenance; the evaluator accepts an explicitly verified completed-checkpoint bundle. Both the original label pair and the warmup pair are now evaluated. Any further training change should follow specific reviewed evidence and remain separately identified; preserve the current reference, thresholds and recorded comparison.

This extends [the measured baseline](baseline-2026-11-06.md), using its saved predictions from inference revision `3bc783f1e53a477a095ac29160674f07e393348d`. **No new model inference, training or reference edits** were performed for this quality review. The source-hashed [evidence record](quality-review-2026-11-06.json) contains the calculations, label audit and visual decisions. The [published-revision verification](quality-replay-verification-2026-11-06.json) records the clean diagnostic execution, 12 passing checks and exact comparison with the independent calculations.

## 1. Define the claim before choosing a target

“Good enough” must name an intended operation and conditions. A useful future claim is: *On a declared independent set of scenes from the intended camera/setup, this exact model and settings met predeclared corner, object, unlocatable-point and timing requirements.* The current evidence supports a reproducible development report, not that stronger claim.

| Requirement | What must be decided | Current evidence/status |
| --- | --- | --- |
| Current deliverable | Cassette object localization in baomai D405 images, retaining the two-corner task | User-confirmed scope; robot hardware/calibration is outside this work |
| Corner identity | Which physical corners and their order | Confirmed: actual lower-front corners, image-X order; unlocatable points remain `0 0 0` |
| Box extent and inclusion | Observed front-face extent; one identifiable visible front independent of corner availability | Adopted local policy `baomai-visible-front-v1`; frozen labels have not been fully migrated or certified |
| Localization tolerance | Maximum pixel error, or calibrated physical error | Unspecified; 2/5/10 px below are illustrations only |
| Required usable-output fraction | Minimum proportion of eligible objects with both corners returned within tolerance | Unspecified; always include missed objects in its denominator |
| Detection errors | Permitted missed objects and extra predictions, ideally with scene-level limits | Unspecified; preserve TP/FN/FP and empty-scene observations |
| Unlocatable-side behavior | Whether to suppress/flag an endpoint that cannot be located in the image | Existing labels say not to guess; confidence 0.50 does not reliably suppress such outputs |
| Timing | Maximum end-to-end latency on the intended hardware, and temporal stability if using video | Offline CPU timing exists; no deployment-camera or action-latency benchmark |
| Evaluation population | Camera, distance, lighting, motion/blur, occlusion, background and scene/session grouping | Twenty development images; independent deployment-like test evidence still needed |

Choose the tolerance from the application, not by increasing it until the current model passes. Pixels cannot be converted to millimetres without the relevant calibration and distance/scene geometry. Physical positioning also requires checking the downstream coordinate transform and action pipeline; a 2D pose score alone does not establish placement accuracy.

Do not equate a model confidence score with a probability of being within a specified pixel distance. Statistical confidence in a measured success rate is another distinct concept. Points on the same objects/images and related capture frames are correlated; 1,227 annotated points are not automatically 1,227 independent deployment trials.

For an illustration of evidence size, if success is defined for **independent, representative scene trials**, observing zero failures in 59 trials gives an exact one-sided 95% binomial lower bound of `0.05^(1/59) = 0.9505` on success probability. A 99% lower bound would require 299 all-success trials under those assumptions. These are mathematical examples, not the required sample size for this project. With failures or clustered scenes, use an appropriate interval and sampling unit. The existing 20-image development set does not satisfy this independent-test example.

### Why pose mAP is not a pixel guarantee

The [pinned standard pose validator](https://github.com/ultralytics/ultralytics/blob/v8.4.182/ultralytics/models/yolo/pose/val.py) uses dataset-specific OKS scales; the distributed two-point YAML supplies no override, so that standard path selects `[0.5, 0.5]`. Its annotated-box area scale is 0.53. The [OKS implementation](https://github.com/ultralytics/ultralytics/blob/v8.4.182/ultralytics/utils/metrics.py) therefore gives, for a visible point at distance `d` and box width/height `w,h`, approximately `exp(-d² / (1.06*w*h))`. It averages over visible reference points.

For a synthetic 100×20 px box with each corner displaced by 10 px, this is about 0.9539—despite failing a 5 px criterion. This illustrates the formula, not this checkpoint's accuracy. Pose mAP summarizes precision/recall across OKS matching thresholds; it is not the fraction of points inside a fixed pixel radius. The historical training run's resolved sigma settings were not independently recovered, so do not claim its published mAP definitely used the distributed default. Continue using direct pixel and both-corner diagnostics for application tolerances.

## 2. Measure usable corners, not only error among returned points

The reference has 1,227 visible endpoints and 584 objects with both endpoints visible. The remaining objects have one visible point (59) or none (21). The original median/p95 is conditional on returned visible endpoints. The following diagnostics retain eligible missed objects/points in the denominator.

At box confidence 0.25, matching IoU 0.50 and endpoint confidence 0.50:

| Illustrative maximum error | Visible endpoints within tolerance / all 1,227 | Objects with both corners within tolerance / all 584 fully visible objects |
| --- | ---: | ---: |
| 2 px | 737 / 1,227 = 60.07% | 263 / 584 = 45.03% |
| 5 px | 1,121 / 1,227 = 91.36% | 521 / 584 = 89.21% |
| 10 px | 1,182 / 1,227 = 96.33% | 568 / 584 = 97.26% |

Success includes equality at the radius and preserves left/right correspondence. Both-corner success means **both corners on the same object** meet the criterion; it is not a squared average endpoint rate. Its denominator excludes partially/unlocatable reference objects, so the two columns describe different populations. Extra detections remain separately reported and are not excused by high corner success.

At the illustrative 5 px radius, 63 fully visible objects fail: 14 lack both returned points and 49 have both points returned but at least one lies farther than 5 px. At 10 px, that becomes 14 unavailable pairs and two pairs with a returned corner outside the radius. The application tolerance changes which failure deserves priority.

For example, a hypothetical requirement of both corners within 5 px on at least 95% of eligible objects would not be met by 89.21%. Neither 5 px nor 95% is an approved project requirement. At 10 px, the development point estimate exceeds 95%, but that would still leave independent testing, unlocatable-side behavior and other requirements unresolved.

All three tested box thresholds (0.15/0.25/0.40) give the same both-corner successes at these three radii. Their object precision/recall trade off: 87.52%/95.03%, 90.84%/94.13%, and 93.39%/93.67%, respectively. A threshold change can reject results; it does not move a retained corner onto the correct physical boundary.

## 3. Check whether an endpoint should be returned at all

There are 101 reference endpoints marked unlocatable. At box confidence 0.25, 66 are on matched objects, allowing a corresponding prediction to be inspected. **61 of those 66 receive a finite in-frame coordinate with endpoint confidence at least 0.50.** None of the other five is rejected for low confidence; they fail the in-frame rule.

This is recorded as `returned_on_unlocatable_reference`. It measures disagreement with reference unlocatability; it is not a geometric error for a hidden point, because no hidden gold coordinate is available. Nor does the aggregate prove that every such point lies on an occluder. The visually inspected examples do establish unsupported points on a hand or gripper in several cases.

The original visible-endpoint median/p95 deliberately excludes `v=0` points. Those metrics remain valid for their defined scope, but cannot establish correct behavior on unlocatable sides.

A separate bounded replay changed only endpoint confidence, keeping box confidence 0.25, object matching and the 66-point denominator fixed:

| Endpoint cutoff | Visible endpoints returned / 1,227 | Both corners within illustrative 5 px / 584 | Returned on unlocatable reference / 66 |
| --- | ---: | ---: | ---: |
| 0.50 | 1,184 | 521 (89.21%) | 61 |
| 0.75 | 1,182 | 521 (89.21%) | 54 |
| 0.90 | 988 | 390 (66.78%) | 35 |
| 0.95 | 247 | 68 (11.64%) | 3 |

These tried settings show a tradeoff, not an accepted new default. At 0.75, seven reference-unlocatable outputs disappear but two visible outputs also disappear, and 54 of the 66 reference-unlocatable sides still receive coordinates. At 0.95, most visible outputs are lost. With box confidence and matching fixed here, every matched visible point was already returned at 0.50, so endpoint filtering alone cannot increase the 5 px both-corner success above 521/584.

## 4. What the label review actually found

### Training label text: all 137 files inspected

All label hashes match the manifest. The existing parser accepts all files; counts match 3,772 objects, 6,839 visible endpoints, 705 missing endpoints and 13 empty scenes. There are no duplicate numeric rows and no visible endpoints more than 2 px outside their corresponding boxes. Strict box/image arithmetic flags 39 rows, but the greatest overrun is only 0.0000064 px, consistent with decimal rounding and not a meaningful physical correction target.

This establishes mechanical integrity, **not visual correctness of all training labels**. No training PNGs were opened or fetched for this text audit.

### Twelve queued validation rows plus one localization example

The review inspected originals before comparing predictions where possible. All 12 queued rows were reviewed; no box/inclusion policy had yet been adopted at that stage. Zero label edits were accepted. The observations and decisions below record that historical review; the subsequently adopted working policy guides any separately versioned follow-up.

| Cases | Observation | Decision |
| --- | --- | --- |
| `new_000056` rows 1, 3, 4 | Visible lower corners are plausible; boxes include different amounts of upper/front structure | Keep points; settle the box convention before changing rectangles |
| `new_000056` row 15 | Right lower corner is hidden by a hand; the nearby proposal puts a confident point on the hand | Retain right `0 0 0`; do not fill it from the prediction |
| `new_000090` rows 4, 15 | Thin reference boxes omit visible upper front regions; model boxes include more upper structure | Box policy needs a consistent decision; neither rectangle is authoritative merely because it exists |
| `new_000090` row 43; `new_000009` rows 52, 53 | One side is visibly obstructed; model proposes hidden-side points | Preserve missing-side labels; record the model availability problem |
| `new_000009` rows 56, 57 | Severe/partial occlusion makes box extent and, for row 56, instance inclusion difficult to certify | Keep frozen reference; require an explicit inclusion/extent rule |
| `new_000009` row 8 | Tiny border sliver; queued prediction 60 is a duplicate of neighboring reference row 9 | Reject that proposed same-object correspondence; its 18.38 px diagnostic difference is not a certified row-8 corner error |
| `new_000008` row 10, right | Gold near (709,454) follows the exposed lower boundary; prediction near (705.69,439.52) lies about 14.48 px higher | Model localization error; retain reference. Endpoint confidence 0.934 does not make it accurate |

The final case was an additional targeted review beyond the 12-row queue. Soft/rounded image edges limit exact pixel adjudication, but do not explain its 14.85 px discrepancy. The findings do not justify rewriting the dataset to imitate model outputs.

### Adopted working box and inclusion rule

The [versioned annotation policy](annotation-policy.md) now adopts visible-front boxes and retains an identifiable visible front even when both actual lower corners are unlocatable. The current baomai point semantics take precedence over historical front-lip and five-point schemas. This is an explicit local choice under the user's authorization, not an additional supervisor-approval claim.

The observations above remain historical review decisions. Adopting the convention does not retroactively migrate those rectangles or replace validation references. Keep uncertain identity/extent cases recorded, and do not silently turn them into background negatives. The eleven-row training candidate remains deduplication only.

## 5. Order the improvement work

1. **Apply the adopted annotation convention and define the image-level comparison.** Use the versioned policy for new/revised labels. Report object misses/extras, visible-point availability, both-corner pixel accuracy and inference timing on the fixed images. Robot hardware/calibration is outside the current task. Review the concrete box examples before any separately versioned box-migration experiment.
2. **Reuse the completed training pilot and staged comparison.** The thirteen-image pilot found eleven duplicate rows; its exact candidate has a verifier and a complete paired-dataset staging procedure. Preserve that isolated experiment. For further ambiguous-corner edits, inspect originals first and use independent annotation to understand coordinate uncertainty; do not guess hidden points.
3. **Collect or select relevant training examples when labels are already correct.** Target confirmed failure modes rather than adding arbitrary images. Review original-image plus TXT pairs, preserve meaningful empty-scene negatives, and separate collection sessions/near-duplicate sequences across training and final testing. Do not use the reviewed validation images as new training examples.
4. **Run one controlled training comparison when a candidate dataset is ready.** Retain epoch 173; compare an unchanged-data fine-tune with a reviewed-data fine-tune from the same starting checkpoint, settings and predeclared checkpoint rule. PR #1 proposes a first 30-epoch, seed-42 comparison. Equal epochs are not equal optimizer steps when dataset sizes differ; record the budget. A single seeded gain is preliminary.
5. **Assess the complete outcome.** Compare misses/extras, endpoint availability, both-corner success at the agreed tolerance, unlocatable-side outputs, difficult subgroups and measured runtime. Do not accept a smaller conditional error obtained only by suppressing difficult examples. Retain the baseline if tradeoffs or uncertainty are unresolved.
6. **Freeze the chosen model/settings, then use independent test scenes.** The current 20 images have already guided model selection and these diagnostics. The excluded five test images are absent from this package; verify their untouched status or obtain new independent scenes. If test results guide another change, they become development feedback and a fresh final assessment is needed.

Hyperparameter changes are hypotheses, not guaranteed improvements. Learning rate, augmentation, pose/visibility loss weighting and architecture changes should follow the observed failure and a fixed comparison. No loss gain or new architecture is selected by this review.

The present evaluator/replay path verifies the reference model-card and manifest identity. Before a future candidate-weight/reference-version comparison, add explicit candidate identity support rather than overwriting the baseline model card, labels or saved-prediction hashes to satisfy the reader. An approved validation correction requires a separately versioned reference and a fair reevaluation of every compared model.

## 6. Reproduce the diagnostic report

After obtaining the baseline predictions and reference originals/labels, use the standard-library-only CLI:

```bash
python3 scripts/check_pose_quality.py
python3 scripts/diagnose_pose_quality.py --predictions reports/baseline-published-20261106/predictions.jsonl --radii 2 5 10 --output reports/quality-001
```

Use a new output directory. `--conf 0.25` selects an already inferred box threshold. The CLI validates reference identity and complete image coverage, reuses the baseline matcher, and writes aggregate JSON, per-image JSON/CSV and a completion marker. It never loads model weights or imports PyTorch/Ultralytics. Radii can be changed for diagnostic exploration, but choosing them does not approve an acceptance tolerance.

The reusable CLI fixes endpoint confidence at 0.50. The separate endpoint-cutoff illustration above is stored with its exact parameters and input prediction hash in the evidence JSON; it does not alter the fixed report or viewer defaults. No live camera, browser Save/reopen acceptance or training run is claimed here.

## Next conversation

Start with the [annotation pilot](additional-annotation-pilot-2026-11-07.md), inspect current main/open PRs, and reuse its versioned candidate, row reviews and provenance evidence. Keep epoch173 as the working model while retaining warmup as a promising candidate. Resolve which upper surfaces belong to the front, roof or rim, and adjudicate the occluded left box extent; then complete the next bounded review batch. Keep the current pilot images quarantined until their unresolved fields are resolved; freeze eligible image IDs and adjudicated label bytes in a separate supplementary reference before scoring. Preserve the frozen 20-image evaluator guard and existing results, and record capture/session, prior-use and correlation limits. The earlier 12-row validation queue is complete; do not reopen it as an automatic-relabeling task. The [training-image pilot](training-pilot-2026-11-06.md) and its eleven-row proposal remain separate data-quality evidence. Use the adopted local annotation policy for any new, separately recorded proposal. Current work is image-space object localization; camera mounting, gripper and calibration details are not prerequisites. Retain the established point semantics and missing-side labels unless original-image evidence supports a change. Earlier candidate-label recovery remains independent and unresolved.

## Method references

- [Ultralytics: data collection and annotation](https://docs.ultralytics.com/guides/data-collection-and-annotation/): objective annotation rules, representative data and review.
- [Ultralytics: model testing](https://docs.ultralytics.com/guides/model-testing/): representative unseen test data and the distinction between development evaluation and testing.
- [NIST AI RMF, Measure 2.1/2.3/2.5](https://airc.nist.gov/airmf-resources/airmf/5-sec-core/): documented measurements and evidence under conditions resembling intended use.
- [NIST statistical handbook: confidence intervals](https://www.itl.nist.gov/div898/handbook/prc/section2/prc241.htm): proportion intervals and exact binomial methods for small samples/few failures. The zero-failure examples above are derived under an explicitly independent Bernoulli-trial assumption.
