# Decide whether the two-corner model is good enough

**Current scope, clarified by the user on 2026-11-07 (Asia/Hong_Kong): find/localize cassette objects in the supplied baomai D405 images.** Keep the existing two-lower-corner pose target and use baomai as the foundation. Camera mounting, robot/gripper integration and physical calibration are outside this deliverable and do not block image evaluation. The supplied epoch173 model remains the working default under the predeclared replacement rule.

**Latest readiness and settings review, 2026-11-07:** the [fixed NMS experiment](postprocess-experiment-2026-11-07.md) completed 240 image/settings predictions without new training data. NMS 0.68 at confidence 0.25 reduces FP 63→61 with the same 39 FN and the exact same 625 matched predictions. The primary 0.60 setting fails because it adds a miss. The [folder command](predict-pose-folder.md) now offers `--nms-iou 0.68` explicitly; default NMS remains 0.70. This is a small development-reference result. Every nonempty validation image still has a matching error, so the present evidence supports a human-reviewed prototype rather than unattended operation on future scenes. The report also prioritizes improvements using the existing repository material.

**Latest model comparison, 2026-11-10:** the [completed seed-43/44 study](box-repeatability-comparison-2026-11-10.md), published in [PR #28](https://github.com/Siuuuers/dl/pull/28), retains epoch173, box confidence 0.25 and NMS IoU 0.70. Each corrected-box candidate loses one match against its same-seed original-label control; p95 improves slightly, while median and extra-detection changes are mixed. The earlier seed-42 detection gain did not repeat. This study is complete and needs no GPU rerun.

**Latest annotation result, 2026-11-10:** the [bounded reference amendment](reference-amendment-2026-11-10.md) is complete under the user's authorization to use assistant-marked labels. It changes 11 files: 13 retained rows modified, four invalid/duplicate rows removed and one omitted cassette added. The revised 20-image reference contains 661 objects and 1,228 locatable endpoints; 14 of the 33 queued rows are amended and 19 remain explicitly unresolved. All five models have been replayed fairly against both reference versions. M0 at confidence 0.25 now scores 27 misses/54 extras and 575/584 both-corner successes within 10 px. These are annotation effects on unchanged predictions, not training gains. Missed objects are now the user's first priority; 10 px is a provisional Euclidean per-corner image-space ceiling, with 5 px retained as a stricter diagnostic.

The [reviewed-annotation seed45 comparison](annotation-comparison-2026-11-11.md) is complete using the matched L4 runs returned in [PR32](https://github.com/Siuuuers/dl/pull/32). At confidence 0.25 on the PR30 revised reference, M0/control/candidate have **27/25/24 misses** and **54/58/55 extras**. The candidate improves the matched control by one miss and three extras, ties at **579/584** both-corner successes within 10 px, and loses one 5 px success (**531→530/584**). The miss difference depends on the fixed confidence threshold. Keep epoch173 as the working default and use the new [seed46 confirmation notebook](../notebooks/baomai_reviewed_annotations_seed46.ipynb) with the unchanged annotation candidate. The [33-row original-image review](persistent-failure-review-2026-11-10.md), [bounded reference amendment](reference-amendment-2026-11-10.md) and earlier box-only studies remain separate historical evidence.

**Earlier warmup comparison, 2026-11-07:** the audited [PR #16](https://github.com/Siuuuers/dl/pull/16) warmup model improves all five primary measures against M0, but its 37 missed objects exceed the fresh control's 36. It therefore fails the required comparison against both controls. The [warmup report](warmup-comparison-2026-11-07.md) and [first label comparison](gpu-comparison-2026-11-07.md) remain historical evidence.

Following the [completed case review and intake](warmup-detection-review-2026-11-07.md), the [annotation pilot](additional-annotation-pilot-2026-11-07.md) now provides a [three-image, 129-row candidate](../data/candidates/additional-pilot-2026-11-07/manifest.json). The candidate changes 87 boxes and five corner positions; the corner edits affect four rows. It adds or deletes no objects and changes no visibility flags. All three images are quarantined for 36 unresolved rows: 35 upper-plane bounds and the left box extent of `frame_0001` row 43. There are no eligible scoring images, and the other 317 recovered images have not completed row review. The pilot used no inference or training; prior upstream use and frame correlation remain limits on any later supplementary evaluation.

The [training pilot](training-pilot-2026-11-06.md) is complete and supplies an eleven-row deduplication proposal. The [annotation policy](annotation-policy.md) is adopted as a local working convention, informed by upstream branch/history research and the user's authorization. Complete reference/candidate datasets can be built and checked using the [training preparation guide](prepare-pose-training.md). The [training runner](train-pose-comparison.md) records bounded resource trials and completed-arm provenance; the evaluator accepts an explicitly verified completed-checkpoint bundle. Both the original label pair and the warmup pair are now evaluated. Any further training change should follow specific reviewed evidence and remain separately identified; preserve the current reference, thresholds and recorded comparison.

The original 2026-11-06 quality review below extends [the measured baseline](baseline-2026-11-06.md), using its saved predictions from inference revision `3bc783f1e53a477a095ac29160674f07e393348d`. **No new model inference, training or reference edits** were performed for that original review. The source-hashed [evidence record](quality-review-2026-11-06.json) contains its calculations, label audit and visual decisions. The [published-revision verification](quality-replay-verification-2026-11-06.json) records the clean diagnostic execution, 12 passing checks and exact comparison with the independent calculations.

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

| Illustrative maximum error | Visible endpoints within tolerance / all 1,227 | Objects with both corners within tolerance / all 584 objects with both reference corners locatable |
| --- | ---: | ---: |
| 2 px | 737 / 1,227 = 60.07% | 263 / 584 = 45.03% |
| 5 px | 1,121 / 1,227 = 91.36% | 521 / 584 = 89.21% |
| 10 px | 1,182 / 1,227 = 96.33% | 568 / 584 = 97.26% |

Success includes equality at the radius and preserves left/right correspondence. Both-corner success means **both corners on the same object** meet the criterion; it is not a squared average endpoint rate. Its denominator excludes partially/unlocatable reference objects, so the two columns describe different populations. Extra detections remain separately reported and are not excused by high corner success.

At the illustrative 5 px radius, 63 objects with both reference corners locatable fail: 14 lack both returned points and 49 have both points returned but at least one lies farther than 5 px. At 10 px, that becomes 14 unavailable pairs and two pairs with a returned corner outside the radius. The application tolerance changes which failure deserves priority.

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

1. **Use the completed reference amendment.** The [reviewed labels](../data/references/visible-front-2026-11-10/README.md) and fair five-model replay are available. Keep their 19 unresolved queue dispositions attached to the bounded reference and retain the historical 20-image labels/results as a separate version.
2. **Use the reviewed training intervention.** The [eight-file annotation overlay](training-annotations-2026-11-10.md) adds five omitted fronts, deletes four gripper-only rows and consolidates two redundant rows. It preserves all 6,839 original visible corner marks and adds two newly marked visible corners. The 32-image triage remains bounded, with unresolved cases recorded. Historical candidates and the training/validation split remain separate and unchanged.
3. **Test only a concrete new intervention.** The paired studies, including seeds 43/44, are already complete. Another run should start from a defined image-backed training-label change and a predeclared comparison, not another search for a favorable seed. Keep model initialization, non-label settings, selected checkpoint and measured training budget comparable. If a new reference is adopted, rescore every compared model against that same version and report the reference change separately from training effects.
4. **Judge usable output with explicit tolerances.** The user prioritizes missed objects and proposes 10 px; use a provisional 10 px Euclidean per-corner ceiling and retain 5 px diagnostics. Preserve misses/extras, endpoint availability, both-corner success, reference-unlocatable outputs and timing. On the revised reference, M0 confidence 0.15 yields 22 misses/82 extras versus 27/54 at 0.25: five fewer misses at 28 more extras. This is a concrete option for review, while an acceptable extra-detection workload and independent-scene acceptance remain unspecified.
5. **Keep development improvement separate from unseen-scene evidence.** Existing images can support annotation correction and controlled development comparisons. The additional 320 upstream images require semantic review and have prior-use/correlation limits; the current pilot remains quarantined. The five historical test images are absent from this package. Independent testing remains a later evidence requirement, not a reason to stop useful work on the supplied images.

Hyperparameter changes are hypotheses, not guaranteed improvements. Learning rate, augmentation, pose/visibility loss weighting and architecture changes should follow the observed failure and a fixed comparison. No loss gain or new architecture is selected by this review.

The evaluator already verifies completed candidate-checkpoint bundles. The new [reference replay script](../scripts/rescore_reference_amendment.py) verifies original dataset, model and prediction identity before applying separately hashed revised labels. Use this supported path for fair reference comparisons; preserve original model cards, labels and saved-prediction headers.

## 6. Reproduce the diagnostic report

After obtaining the baseline predictions and reference originals/labels, use the standard-library-only CLI:

```bash
python3 scripts/check_pose_quality.py
python3 scripts/diagnose_pose_quality.py --predictions reports/baseline-published-20261106/predictions.jsonl --radii 2 5 10 --output reports/quality-001
```

Use a new output directory. `--conf 0.25` selects an already inferred box threshold. The CLI validates reference identity and complete image coverage, reuses the baseline matcher, and writes aggregate JSON, per-image JSON/CSV and a completion marker. It never loads model weights or imports PyTorch/Ultralytics. Radii can be changed for diagnostic exploration, but choosing them does not approve an acceptance tolerance.

The reusable CLI fixes endpoint confidence at 0.50. The separate endpoint-cutoff illustration above is stored with its exact parameters and input prediction hash in the evidence JSON; it does not alter the fixed report or viewer defaults. No live camera, browser Save/reopen acceptance or training run is claimed here.

## Next conversation

Start from the [completed seed45 result](annotation-comparison-2026-11-11.md) and [seed46 confirmation handoff](training-annotations-seed46.md). PR32 returned a valid matched L4 pair; archive verification, fresh M0/control/candidate inference, both-reference scoring, independent numerical checks and complete saved-prediction replay are finished. No seed46 training result exists yet. The next required input is `baomai-reviewed-annotations-seed46-results.zip` from the new pinned notebook, with the same `annotations-2026-11-10` candidate and original epoch173 initialization. Keep epoch173 at confidence 0.25/NMS 0.70 while awaiting confirmation. Evaluate the new pair under its own compatible pinned source and report seeds45/46 separately, preserving all outcomes. Seed45 replay must use clean `ed4ee162738685bff93d6cad1402d95e128bceb4`, because the new seed46 recipe registry changes source hashes. Misses remain first priority; retain extras, endpoint availability, 5/10 px both-corner, single-corner and hidden-side diagnostics. Nineteen of the current candidate's 24 misses fall outside the two-corner denominator. There is no automatic promotion or threshold retuning. Earlier box-only candidates, unresolved reference/training rows and the quarantined additional-frame pilot remain separate. Supplied-image cassette localization remains the scope. Inspect the stacked PR state before resuming.

## Method references

- [Ultralytics: data collection and annotation](https://docs.ultralytics.com/guides/data-collection-and-annotation/): objective annotation rules, representative data and review.
- [Ultralytics: model testing](https://docs.ultralytics.com/guides/model-testing/): representative unseen test data and the distinction between development evaluation and testing.
- [NIST AI RMF, Measure 2.1/2.3/2.5](https://airc.nist.gov/airmf-resources/airmf/5-sec-core/): documented measurements and evidence under conditions resembling intended use.
- [NIST statistical handbook: confidence intervals](https://www.itl.nist.gov/div898/handbook/prc/section2/prc241.htm): proportion intervals and exact binomial methods for small samples/few failures. The zero-failure examples above are derived under an explicitly independent Bernoulli-trial assumption.
