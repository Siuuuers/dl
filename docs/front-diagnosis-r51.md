# R51: completed diagnosis, confidence bottleneck and next comparison

27 November 2026, Asia/Hong_Kong.

**[PR143](https://github.com/Siuuuers/dl/pull/143) contains the completed R49
inference diagnosis.** Its fresh hard-NMS baseline exactly reproduces the saved
PR137 candidate on all 20 original development photographs. The submitted linear
Soft-NMS pipeline removes one final unmatched output and recovers no missed
targets. The model checkpoint and training labels are unchanged; PR35 remains
selected under the existing frozen-reference gate.

The R50 notebook was a repaired handoff for this same inference question. The
successful R49 return now provides the required tensors and traces, so a new run
of the equivalent R50 notebook is unnecessary just to obtain them again.

## 1. Measured result

The [independent CPU audit](../results/evaluation/pr143-r49-20261127/README.md)
checks the archive, returned script, member inventories, decoded tensors,
native/Soft-NMS reconstruction, postclip filters, scores and miss-stage traces.
R50 is additionally scored as a separate label-version replay of the same output.

| Reference | Pipeline | Misses | Unmatched | Formal duplicates | Both corners within 5px |
| --- | --- | ---: | ---: | ---: | ---: |
| Frozen666, primary | Hard | 34 | 91 | 17 | 511/584 |
| Frozen666, primary | Soft | 34 | 90 | 17 | 511/584 |
| Historical663 | Hard | 30 | 90 | 17 | 511/584 |
| Historical663 | Soft | 30 | 89 | 17 | 511/584 |
| R38, development | Hard | 15 | 84 | 14 | 566/583 |
| R38, development | Soft | 15 | 83 | 13 | 566/583 |
| R50 proposal | Hard | 15 | 84 | 13 | 566/583 |
| R50 proposal | Soft | 15 | 83 | 12 | 566/583 |

All reference versions use the same scenes; they are not independent test sets.
Matching remains IoU >=0.50 and endpoint confidence >=0.50. Ordinary suppression
uses IoU 0.70, confidence strictly greater than 0.13, cap 300 and the existing
postclip filter. All raw/final matched-anchor assignments remain unchanged.

The final selected anchor set differs only at `new_000090 / anchor10313`, hard
postclip output **P060**, score approximately 0.13649. It is a repeated-front
hypothesis for R041 in R38/R50, with an extent reaching toward the gripper. The
assigned owner P043 remains. It is categorized as another unmatched output under
the older frozen666/historical663 references because their box extent differs.

The raw-stage distinction matters:

| R38 pipeline state | Outputs | Misses | Unmatched | Formal duplicates |
| --- | ---: | ---: | ---: | ---: |
| Hard NMS before postclip | 732 | 15 | 93 | 22 |
| Soft-NMS before postclip | 745 | 15 | 106 | 34 |
| Hard NMS after postclip | 723 | 15 | 84 | 14 |
| Soft-NMS after postclip | 722 | 15 | 83 | 13 |

Raw Soft-NMS adds 14 proposal identities and removes one. The 14 additions are
subsequently removed by the unchanged postclip filter. **Raw Soft-NMS already
recovers zero targets**, so the postclip filter did not erase a measured recall
recovery. The no-postclip comparison is already present in this archive; it does
not need another GPU run.

## 2. The 15 measured miss stages

All 15 missed rows have identical reference geometry in R38 and R50. The stages
are 10 confidence-gate cases, three without an IoU-eligible decoded box, and two
ordinary-NMS cases.

| Image / R38 row | Measured stage | Best decoded IoU | Highest score among boxes with IoU >=0.50 |
| --- | --- | ---: | ---: |
| 008 / R001 | Confidence | 0.9184 | 0.09077 |
| 008 / R033 | Confidence | 0.9267 | 0.01100 |
| 008 / R034 | Confidence | 0.8397 | 0.07217 |
| 009 / R051 | No eligible geometry | 0.4906 | none |
| 009 / R054 | Confidence | 0.7658 | 0.02385 |
| 010 / R030 | Confidence | 0.7795 | 0.05840 |
| 010 / R033 | Confidence | 0.9046 | 0.10709 |
| 011 / R055 | Confidence | 0.6308 | 0.00341 |
| 011 / R056 | Confidence | 0.5711 | 0.01045 |
| 011 / R057 | No eligible geometry | 0.3065 | none |
| 011 / R059 | Confidence | 0.7508 | 0.09232 |
| 011 / R060 | No eligible geometry | 0.4027 | none |
| 088 / R051 | Confidence | 0.6573 | 0.01772 |
| 089 / R048 | Ordinary NMS | 0.7345 | 0.79548 |
| 090 / R039 | Ordinary NMS | 0.9298 | 0.69167 |

The best-IoU box and highest-score eligible box can be different proposals. For
example, 008/R001 already has a decoded box with IoU about 0.918, but its score is
only 0.0324. Even the highest score among all eligible boxes is 0.0908, below 0.13.
The previous final-output review could not reveal that available geometry.

These are measured pipeline locations. They do not by themselves prove why the
network learned a particular score, whether a training label is physically wrong,
or which training intervention will improve it. Scores are not calibrated
probabilities of physical truth.

### The two NMS competitions

For 089/R048, eligible anchor 17091 has score 0.79548 and target IoU 0.51033.
Higher-score anchor 17090 survives with score 0.81868 but target IoU 0.49507.
Their pair overlap is 0.96759; linear Soft-NMS reduces the eligible score to
approximately 0.02578, below 0.13.

For 090/R039, eligible anchor 9354 has score 0.69167 and target IoU 0.55292.
Winning anchor 9355 has score 0.72389 but target IoU 0.49370. Their pair overlap
reduces the eligible score to approximately 0.07408, also below 0.13.

These cases show score/extent ranking competition close to the evaluation gate.
They do not establish that the reference is physically exact. Any annotation
amendment still requires original-image evidence and a versioned decision.

## 3. A bounded confidence replay, without new inference

After the diagnosis, one post-hoc CPU probe replays the returned tensors at
confidence 0.05 instead of 0.13, using native hard NMS and the same postclip filter.
Its [replay script](../scripts/replay_front_confidence_r51.py),
[compact score summary](../results/evaluation/pr143-r49-20261127/confidence-probe/summary.json)
and [proposal-level mechanisms](../results/evaluation/pr143-r49-20261127/confidence-probe/mechanisms.json)
are published beside the return audit.
It first reproduces all 20 native baseline proposal-index sets, boxes, keypoints
and postclip outcomes exactly at 0.13. This is one diagnostic probe, not a sweep
or a promoted operating threshold.

| Reference | Misses, 0.13 to 0.05 | Unmatched, 0.13 to 0.05 | Formal duplicates, 0.13 to 0.05 |
| --- | ---: | ---: | ---: |
| Frozen666 | 34 to 27 | 91 to 164 | 17 to 36 |
| Historical663 | 30 to 25 | 90 to 165 | 17 to 35 |
| R38 | 15 to 9 | 84 to 158 | 14 to 36 |
| R50 | 15 to 9 | 84 to 158 | 13 to 35 |

The six recovered R38/R50 rows are `008:R001`, `008:R034`, `010:R030`,
`010:R033`, `011:R059` and `090:R039`. There are no newly missed R38/R50 targets.
Five are confidence-gate cases. The NMS case 090/R039 recovers through a different
lower-score, narrower proposal, rather than by unsuppressing the original
high-score eligible proposal. Both-corner 5px success remains 566/583.

The measured exchange is **six recovered fronts for 74 more unmatched outputs**.
It demonstrates useful available geometry and a substantial global-threshold
cost. It does not justify changing the default confidence. Unmatched outputs are
not automatically background, and explicit excluded fronts/ownership holds
remain unchanged.

## 4. Next isolated training question

The prepared study isolates native classification BCE on already-assigned
partial targets, factor 1 versus factor 2, keeping native assignment, soft target
scores and the unweighted normalizer. [R30/PR109](front-review-r30.md) jointly
weighted [box and classification terms](../scripts/partial_target_loss.py).
[R31](../scripts/partial_box_loss_r31.py) weighted boxes only; its later seed-47
screen failed, as recorded in the [prior-trial review](project-direction-r50.md).
R51 isolates classification. It is not the first partial-target weighting study,
a new detector architecture, or a proven way to raise correct scores without cost.

The [new notebook](../notebooks/baomai_partial_classification_r51.ipynb) and
[experiment guide](partial-classification-r51.md) specify this paired comparison.

Both arms use the same 142 photographs / 3,999 fronts: 3,397 with two locatable
corners, 465 with one and 137 with zero. They keep the original epoch 173
initializer, seed 46, 30 epochs, batch 4, input 1280, FP32 and AdamW; each has 1,080 batches,
80 expected optimizer updates and 4,260 image presentations. Native box, pose and
visibility loss formulas, labels, holds, references and the 0.13 operating point
remain fixed.

The [actual-native CPU loss/gradient guard](../results/evaluation/pr143-r49-20261127/partial-classification-gradient-guard.json)
passed. Factor 1 reproduces the native loss and all 498 network-parameter gradients
bit-for-bit. At fixed prediction outputs, factor 2 doubles only assigned-partial
classification gradients; untreated score gradients, box/keypoint/sigma gradients,
RLE gradients, assignment and normalization remain exact. Empty, fully visible,
changed-visibility and native class-weight cases also pass. The guard uses real
training images resized to 320 by 192 pixels and synthetic edge-case batches;
it runs no optimizer updates. The planned training input remains 1280.

Unchanged loss formulas and fixed-output checks do not mean trained boxes and
corners will remain unchanged: shared features can change during optimization.
Native soft targets include quality information, so the outcome is not a simple
guaranteed score increase. No GPU result for this new comparison is claimed by
the diagnosis report.

The user's thesis topic remains open. The professor's stated assignment is
annotation and training; the detection lecture supplies background concepts.
New captures can later extend independent-session evidence. Existing-data
diagnosis and one controlled follow-up can proceed now, with the development
data reuse stated explicitly.

## 5. Focused PDF and reproduction

`r51-diagnosis-and-next-steps.pdf` has 14 pages:

- Pages 1-3: measured results, all 15 stage identities and the sole removed output.
- Pages 4-11: all 15 misses, two original/decoded/retained-output cards per page.
- Page 12: the two NMS ranking competitions.
- Page 13: the single confidence probe and its cost.
- Page 14: the next isolated training question and scope limits.

The earlier [R50 error atlas](front-error-review-r50.md) remains the complete
119-case visual review. R51 adds the newly returned mechanism evidence without
reprinting all unchanged corner and unmatched-output cards. No label is changed
by the report renderer.

```bash
python scripts/export_front_diagnosis_r51.py \
  --archive /path/to/baomai-r49-diagnosis-results.zip \
  --images-dir /path/to/verified-originals \
  --confidence-probe results/evaluation/pr143-r49-20261127/confidence-probe/summary.json \
  --output /path/to/r51-diagnosis-and-next-steps.pdf
```

The return audit must pass before rendering. The report verifies the archive
inventory, original image hashes, R50 label hashes, probe baseline, all 15 miss
identities, native guard receipt and guarded source hashes, paragraph bounds and
title/text widths. [PDF verification](../results/review/front-diagnosis-r51-20261127/verification.json),
[page index](../results/review/front-diagnosis-r51-20261127/pdf-index.json),
[case data](../results/review/front-diagnosis-r51-20261127/case-diagnostics.json)
and the [rendered-page QA record](../results/review/front-diagnosis-r51-20261127/visual-verification.json)
bind the delivered document to its evidence.

Source pins: PR143 head `a59f07dce21c5aedcb088c725dc14225a53a301c`;
return SHA-256 `892c8ee79ff470990bb47dc59f947ad6bd2577a86bc9827e9dd57f81c56213f3`;
returned script SHA-256 `188df92036d5dc8fd8e13fc8d8815728a56c33ced654de2cf4716788eac47811`;
R50 manifest SHA-256 `f639459bdd986f6553b9e00f0ecfdf1eee3c7f5344e95c300a4a17a66352c04f`.
