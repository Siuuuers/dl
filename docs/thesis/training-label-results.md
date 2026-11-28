# Completed controlled training-label comparison

28 November 2026, Asia/Hong_Kong.

**Replacing the original training annotations with the frozen reviewed annotations improves the measured result against the reviewed reference in this paired fine-tuning run.** With the same two-view inference rule, missed fronts fall from **47 to 15**, extra outputs from **85 to 51**, and complete pairs within five pixels rise from **541/583 to 572/583**. The duplicate subset increases from **6 to 13**. Both older evaluation references favour the original-label arm.

This is evidence for a specific training-label intervention under fixed conditions. It does not establish that a reference is physically correct, isolate the value of the five supplementary training photographs, or demonstrate performance on a new capture session. The [fixed protocol](training-label-comparison.md) and the [independent evidence audit](evidence/training-label-return/README.md) define the comparison.

## 1. What was held fixed

Both arms used the same 137 original training photographs, including 13 empty images, the same supplied epoch-173 starting checkpoint, the same architecture and native losses, seed 46, and the same 30-epoch recipe. The five supplementary photographs were excluded. Each arm received 4,110 image presentations, 1,050 batches and 78 optimizer updates. Training used batch size four, image size 1,280, FP32 and AdamW on the same NVIDIA L4 runtime. Selection used the final epoch, with the same supplied validation labels for training monitoring.

The saved receipts have identical initial parameter-state hashes, all 30 epoch sample-order hashes, optimizer update indices and runtime identities. Training settings differ only in the dataset and output directory paths. A four-update resource trial was discarded; neither full arm continues its trained state. The trial and both full arms record the same initial parameter state. Individual augmentation parameter draws are not archived, so sample-order verification must not be described as a direct check of every augmented pixel.

| Training-label arm | Photographs | Annotated fronts | Available corner slots | Unavailable corner slots | Empty photographs |
| --- | ---: | ---: | ---: | ---: | ---: |
| Original annotations | 137 | 3,772 | 6,839 | 705 | 13 |
| Frozen reviewed annotations | 137 | 3,784 | 6,837 | 731 | 13 |

Exactly 92 training-label files differ numerically. The intervention includes those frozen changes to front instances, box coordinates, corner coordinates and availability flags as a package. It does not isolate any one type of amendment, and it does not introduce every subsequently clarified annotation-definition rule. No labels were edited by this experiment or by this result audit.

## 2. Results against the reviewed reference

All rows below use the same 20 development photographs with 654 reference fronts, including 583 with both lower corners available. Box confidence is strictly above 0.25, endpoint confidence is at least 0.50, and same-class box matching requires IoU at least 0.50. Native NMS uses 0.70; the additional suppression after clipping removes pairs with prediction-to-prediction IoU strictly above 0.70. Duplicates are included within extras. The pair denominator includes missed complete fronts and unavailable predicted pairs.

| Training annotations | Inference | Misses | Extras | Duplicate subset | Both within 5 pixels | Both within 10 pixels |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Original | Original view | 47 | 85 | 6 | 545/583 | 566/583 |
| Original | Two-view corner average | 47 | 85 | 6 | 541/583 | 566/583 |
| Reviewed | Original view | 15 | 51 | 13 | 570/583 | 582/583 |
| Reviewed | Two-view corner average | 15 | 51 | 13 | 572/583 | 583/583 |

The two-view rule retains the original-view boxes and scores and only averages paired corner coordinates. Its detection counts therefore stay unchanged within each arm. It gains two five-pixel successes for the reviewed-label arm, but loses four for the original-label arm. Averaging is not guaranteed to improve every trained detector or threshold criterion.

The controlled label comparison must use the same inference row on both sides. At five pixels, the original-view comparison gains 31 complete pairs and loses six, for a net gain of 25. The two-view comparison gains 37 and loses six, for a net gain of 31. Of those 37 gains, 13 recover previously missed complete fronts and 24 improve fronts already detected by both arms. Another recovered complete front, `new_000055/R021`, has a maximum corner error of approximately 5.607 pixels and therefore succeeds only at ten pixels. At ten pixels the two-view comparison gains 17 and loses none. [Paired corner identities and replayed measurements](evidence/training-label-return/measurement-replayed.json) preserve the individual outcomes.

### Detection according to corner availability

| Available reference corners | Reference fronts | Original-label arm detected | Reviewed-label arm detected |
| --- | ---: | ---: | ---: |
| Two | 583 | 569 | 583 |
| One | 54 | 30 | 44 |
| Zero | 17 | 8 | 12 |
| One or zero combined | 71 | 38 | 56 |

The reviewed-label arm recovers 33 previously missed fronts: 14 with two available corners, 15 with one, and four with none. One previously detected front becomes missed: `new_000008/R034`, a right-edge front with one available corner. The net reduction is 32 misses. Detection of the incomplete-corner group rises from **38/71 (53.52%) to 56/71 (78.87%)**. Its remaining 15 misses comprise ten one-corner and five zero-corner references. [Paired detection records](evidence/training-label-return/paired-detection-cases.json) provide boxes, flags and both states.

Thus 583/583 pairs within ten pixels still does not mean every cassette front is detected. It excludes the 71 fronts without a complete reference pair. Corner availability records what can be annotated; the flag alone does not identify the physical reason for an unavailable point. All three reference-empty photographs produce zero outputs in both arms, which is a result on those three familiar scenes rather than an estimated general false-alarm rate.

### Duplicate tradeoff

Nonduplicate extras fall from 79 to 38, while duplicates rise from six to 13. Of the 13 reviewed-label duplicates, 12 have their largest overlap with an already matched incomplete-corner reference: ten one-corner and two zero-corner fronts. One maps to a complete pair. These are geometric associations for review, not independent physical verdicts.

Each of those duplicate predictions has IoU at or below 0.70 with the prediction matched to its highest-overlap reference. The range is approximately 0.475–0.700; the border case at `new_000009` is approximately 0.699690. Their survival is therefore consistent with the stated suppression rule. This does not justify choosing a lower cutoff from these examples: stronger suppression can also remove nearby distinct fronts. [Duplicate records](evidence/training-label-return/reviewed-duplicate-cases.json) preserve the relevant identities and overlaps.

## 3. Sensitivity to older references

The same predictions can be rescored against different labels without repeating neural inference. For the common two-view rule:

| Evaluation reference | Training annotations | Misses | Extras | Duplicate subset | Both within 5 pixels | Both within 10 pixels |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Untouched supplied, 664 fronts | Original | 34 | 62 | 15 | 531/584 | 572/584 |
| Untouched supplied, 664 fronts | Reviewed | 50 | 76 | 12 | 513/584 | 566/584 |
| Earlier fixed, 666 fronts | Original | 23 | 49 | 17 | 544/584 | 584/584 |
| Earlier fixed, 666 fronts | Reviewed | 38 | 62 | 14 | 523/584 | 575/584 |
| Reviewed, 654 fronts | Original | 47 | 85 | 6 | 541/583 | 566/583 |
| Reviewed, 654 fronts | Reviewed | 15 | 51 | 13 | 572/583 | 583/583 |

Against the supplied reference, five-pixel gains/losses are 9/27 and ten-pixel gains/losses are 3/9. Against the earlier fixed reference, they are 6/27 and 0/9. Detection likewise reverses: the reviewed-label arm recovers four but newly misses 20 supplied-reference fronts, and recovers three but newly misses 18 earlier-reference fronts. The full [12-row replay](evidence/training-label-return/audit.json) also includes both original-view results under each reference.

The frozen training-label intervention and evaluation-reference sensitivity answer different questions. Holding the reviewed evaluation reference fixed supports a favourable effect of this training-label substitution under the tested recipe. Changing the evaluation reference shows that the measured ranking depends strongly on the annotation definition. Neither result selects which version is physically correct. All three references reuse the same photographs and are not three independent validation datasets.

## 4. Relationship to the retained method

The previously retained two-view candidate gives 18 misses, 51 extras, five duplicates, 569/583 pairs within five pixels and 583/583 within ten against the reviewed reference. The new reviewed-label arm gives 15, 51, 13, 572/583 and 583/583 respectively. It therefore improves some reviewed-reference measures while increasing duplicate outputs. On the supplied and earlier fixed references, its five-pixel counts are 513 and 523, compared with the retained candidate's 514 and 525. The [earlier audit](detection-errors.md) documents that candidate.

These cross-run observations are descriptive. The historical candidate used 142 photographs and a different recorded GPU run; comparing it with this 137-photograph arm does not isolate the effect of the five additional images. The controlled claim belongs to the two arms within this returned run. No deployment model or default inference setting is changed by the analysis.

## 5. Verification and limits

PR158 is frozen at `6bde3b7a759604d605965d468e6102bf99bf7946`, with execution source `acaf5d4b4043583fdd81f5ea848b3a6a68ada728`. The compact evidence archive has 414 members. All 395 archived source/data files match that source commit, including 89 execution-source files and 306 frozen input files. All committed checkpoint wrappers, their inner parts and both assembled model hashes verify. The label plan is rebuilt exactly from the archived source.

The exact pinned evaluator reproduces all 12 measurement rows and every prediction, suppression, view association, per-image score and reference-case record. A separately written confidence-ordered matcher also reproduces all match identities, detection counts and complete-pair states for 240 image/arm/view/reference combinations. The pinned evaluator retains clipped nonpositive-area outputs as unmatched false positives; such records are not silently discarded. This is a saved-output and receipt audit, not a new training or inference run.

One auxiliary archive member needs an explicit qualification: `training/original_labels/progress.json` starts with the four-byte prefix `arm\n` and is therefore not valid JSON. Removing that prefix for comparison yields bytes exactly identical to the valid final `receipt.json`. The producer's completion verifier reads the final receipt. The archive inventory correctly records the prefixed original bytes, which remain untouched. This observation does not change the verified final receipt or the exactly reproduced evaluation results.

The remaining limits are one paired seed, repeated use of development photographs, and an already trained common starting model. The experiment estimates conditional fine-tuning after that checkpoint; it does not retrain the complete model history under alternative labels. Physical annotation accuracy, independent inter-annotator repeatability, robot positioning tolerance and generalization to a new capture session remain unmeasured.

## 6. Next decision

Keep this as a completed positive result for the reviewed-reference training-label intervention, with the adverse duplicate and older-reference findings attached. Before changing labels again, inspect the original pixels using the documented [annotation definition](../annotation-definition.md). The [bounded inspection priorities](evidence/training-label-return/inspection-priorities.json) include reference-reversal regions, partial fronts, duplicate clusters, the single newly missed front and one ordinary control region. Selection is diagnostic; it does not turn these regions into an independent test set.

A model disagreement can identify a useful place to look but is not proof that a label is wrong. Apply a correction only when the physical boundary or visibility decision is supported by the image and definition, preserve its provenance, and evaluate every candidate against the same corrected reference. The present result audit makes no annotation corrections and does not require another unplanned model family or threshold sweep.
