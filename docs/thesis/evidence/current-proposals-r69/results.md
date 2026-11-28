# Current reviewed-checkpoint proposal stages

Fresh CPU inference collected every decoded FP32 proposal from the frozen PR158 reviewed-label checkpoint for all 20 development photographs in both original and reflected views. The confidence, NMS, matching and corner thresholds remain unchanged. No optimizer, training or label editing is used.

## r67 measurements and runtime comparison

| Runtime | Output stage | Misses | Extras | Duplicate subset | Pair ≤5 px | Pair ≤10 px |
|---|---|---:|---:|---:|---:|---:|
| archived_l4 | native_original | 18 | 57 | 19 | 570/583 | 582/583 |
| archived_l4 | native_reflected | 12 | 47 | 17 | 568/583 | 583/583 |
| archived_l4 | postclip_original | 18 | 50 | 13 | 570/583 | 582/583 |
| archived_l4 | postclip_reflected | 12 | 39 | 9 | 568/583 | 583/583 |
| archived_l4 | two_view | 18 | 50 | 13 | 572/583 | 583/583 |
| fresh_cpu | native_original | 18 | 57 | 20 | 570/583 | 582/583 |
| fresh_cpu | native_reflected | 12 | 47 | 17 | 568/583 | 583/583 |
| fresh_cpu | postclip_original | 18 | 50 | 14 | 570/583 | 582/583 |
| fresh_cpu | postclip_reflected | 12 | 39 | 9 | 568/583 | 583/583 |
| fresh_cpu | two_view | 18 | 50 | 14 | 572/583 | 583/583 |

The archive/fresh comparison records 666 target-level index/outcome differences across five output stages and four references. Of these, 666 are matched-output index changes with unchanged detection, known-corner availability and 5/10-pixel statuses. Detection gains/losses are 0/0; complete-pair five/ten-pixel status changes are 0/0. Full numerical differences and per-image scores are retained. A changed record index does not by itself establish a changed physical match.

The duplicate-category comparison is separate. Its ledger preserves any unmatched-output reference-IoU boundary crossing and the corresponding fresh native suppression witness. A closest fresh raw box to an archived output is explicitly a diagnostic association; archived full decoded anchors remain unavailable.

## Fresh r67 first-lost stages

| View | Stage | Fronts |
|---|---|---:|
| original | confidence_gate | 12 |
| original | matched | 640 |
| original | native_nms | 1 |
| original | no_iou_eligible_decoded_box | 5 |
| reflected | confidence_gate | 10 |
| reflected | matched | 646 |
| reflected | native_nms | 1 |
| reflected | no_iou_eligible_decoded_box | 1 |

## Three restored focus targets

| Target | View | Stage | Best raw IoU | Highest IoU-eligible score | Eligible raw boxes |
|---|---|---|---:|---:|---:|
| r67/new_000009/R055 | original | confidence_gate | 0.595507 | 0.01454296 | 2 |
| r67/new_000009/R056 | original | no_iou_eligible_decoded_box | 0.446675 | None | 0 |
| r67/new_000053/R022 | original | confidence_gate | 0.589806 | 0.00016235 | 3 |
| r67/new_000009/R055 | reflected | confidence_gate | 0.690462 | 0.00772521 | 3 |
| r67/new_000009/R056 | reflected | confidence_gate | 0.525817 | 0.08451592 | 1 |
| r67/new_000053/R022 | reflected | confidence_gate | 0.749237 | 0.00014884 | 2 |

The best-overlap and highest-scoring eligible raw boxes are separate records. Each includes exact geometry, candidate identity, stage survival and confidence-qualified errors at known corners. A geometrically eligible proposal does not automatically supply a useful corner estimate.

The duplicate-competition ledger compares each scored duplicate with the matched outputs sharing its eligible reference: their mutual box overlap, confidence order, individual reference overlaps and known-corner errors remain visible. Matching IoU 0.50 to the same reference does not imply that their mutual overlap exceeds native NMS 0.70.

## Fixed corner-rule runtime continuity

The separately frozen corner-qualified suppression rule is also applied unchanged to this checkpoint's fresh completed two-view outputs. This is a runtime continuity check, not a new threshold comparison.

| Runtime | r67 misses | Extras | Duplicate subset | Pair ≤5 px | Pair ≤10 px |
|---|---:|---:|---:|---:|---:|
| archived_l4 | 18 | 44 | 11 | 572/583 | 583/583 |
| fresh_cpu | 18 | 44 | 12 | 572/583 | 583/583 |

All-image selected-index agreement: True. Cross-runtime discrete target changes after the rule: 116. Fresh baseline-to-rule target changes across four references: 0.

## Scope and evidence

The compressed raw archive contains 40 complete FP32 tensors, without quantization. The trace records preprocessing, native candidate rank, native suppression witnesses, native returned indices, and postclip lineage. Native outputs are reconstructed exactly from captured tensors and independently replayed before stage results are interpreted.

- These are 20 repeatedly inspected development photographs, not an independent test set.
- Raw tensors describe only the frozen PR158 reviewed-label checkpoint and the recorded fresh CPU runtime. Other checkpoints remain outside this full-stage collection.
- The archived L4 and fresh CPU runtimes differ. Numerical records and scored outcomes are compared explicitly; prediction byte identity is not presumed.
- Equal confident scores are preserved. Native tied choices are observed and checked against the active maximum score; display ranking uses anchor index only to make tied rows legible, without changing native inference.
- A best-overlap candidate is selected using annotations for diagnosis. It may depict an adjacent physical front and is not a deployable ranking or matching rule.
- Known-corner errors are recorded only where the annotation supplies that corner; predicted endpoint availability is a separate confidence test.
- Fixed-pool support and maximum one-to-one matching are upper bounds under the existing IoU relation. They do not establish recoverable accuracy, a confidence cutoff, or an optimal suppression rule.
- First-lost stages explain this computation, not the training cause of a low score or poor box. No training, loss change, label change or optimizer update is performed.
- Original and reflected postclip outputs are separately scored to expose view support; the completed two-view method still preserves original boxes and averages only eligible associated corners.
- Collection timings are local resource observations, not a detector speed benchmark or cross-hardware comparison.
- The notebook's separately packaged corner-suppression companion uses saved R68 predictions and its own fixed protocol; it does not modify these freshly collected tensors.
