# Current missed-front proposal stages

This CPU replay explains the 18 missed reviewed fronts of the retained method at confidence > 0.25. It changes no labels, outputs, or model weights.

| Stage where IoU-eligible support disappears | Missed fronts |
|---|---:|
| no iou eligible decoded box | 3 |
| confidence gate | 12 |
| ordinary nms | 3 |
| postclip suppression | 0 |
| retained assignment contention | 0 |

All 18 missed fronts have incomplete corner references: 13 have one available corner and five have none. The retained measurement remains 636/654 matched fronts, 51 extra predictions including five duplicates, and 569/583 complete pairs within five pixels.

## Continuity checks

Both hash-pinned archives and every inventoried member were verified. The archived model and photograph hashes agree. All 20 historical 0.13 native outputs, native anchor indices, and postclip outputs replay exactly. At 0.25, all 20 current native record orders and postclip identities agree; all 60 reference-image comparisons preserve matching, missed/extra/duplicate identities, and corner threshold outcomes.

The historical tensors were generated on a Tesla T4; the current cache was generated on CPU. The maximum coordinate difference is 0.00018310546875 pixels, and the maximum box or corner score difference is 3.75509262085e-06. These are measured agreement checks, not byte identity.

## Geometric support and competition

| Candidate stage | Candidates | References with support | Maximum one-to-one matches | Candidates eligible for multiple references |
|---|---:|---:|---:|---:|
| decoded | 386400 | 651 | 651 | 9 |
| above confidence | 6438 | 639 | 639 | 0 |
| native nms | 693 | 636 | 636 | 0 |
| postclip | 687 | 636 | 636 | 0 |

A reference with one eligible proposal is not necessarily independently recoverable. The one-to-one bounds account for shared candidates; the actual scorer also retains its original greedy assignment. Multiple raw proposals are not automatically multiple physical fronts or duplicate errors. The detailed cases include every above-threshold eligible proposal lost to NMS and its actual suppressor. The existing extra postclip pass removes six duplicate predictions: native outputs have 57 extras including 11 duplicates; postclip outputs have 51 extras including five duplicates. Both stages match the same 636 references.

## Every current missed front

| Photograph / reviewed row | Available corners | Stage | Best decoded IoU | Highest eligible score | Matched at historical 0.13 |
|---|---:|---|---:|---:|---|
| new_000008 / 1 | 1 | confidence gate | 0.918381 | 0.090770 | no |
| new_000008 / 33 | 1 | confidence gate | 0.926685 | 0.011005 | no |
| new_000008 / 34 | 1 | confidence gate | 0.839737 | 0.072173 | no |
| new_000009 / 51 | 1 | no iou eligible decoded box | 0.490614 | none | no |
| new_000009 / 54 | 0 | confidence gate | 0.765790 | 0.023850 | no |
| new_000010 / 14 | 0 | confidence gate | 0.842348 | 0.146784 | yes |
| new_000010 / 30 | 1 | confidence gate | 0.779465 | 0.058395 | no |
| new_000010 / 32 | 0 | confidence gate | 0.839197 | 0.167130 | yes |
| new_000010 / 33 | 0 | confidence gate | 0.904613 | 0.107094 | no |
| new_000011 / 55 | 1 | confidence gate | 0.630831 | 0.003410 | no |
| new_000011 / 56 | 1 | confidence gate | 0.571046 | 0.010447 | no |
| new_000011 / 57 | 1 | no iou eligible decoded box | 0.306544 | none | no |
| new_000011 / 59 | 1 | confidence gate | 0.750754 | 0.092318 | no |
| new_000011 / 60 | 0 | no iou eligible decoded box | 0.402694 | none | no |
| new_000088 / 51 | 1 | confidence gate | 0.657288 | 0.017721 | no |
| new_000089 / 43 | 1 | ordinary nms | 0.625526 | 0.490754 | yes |
| new_000089 / 48 | 1 | ordinary nms | 0.734533 | 0.795476 | no |
| new_000090 / 39 | 1 | ordinary nms | 0.929821 | 0.691672 | no |

## Interpretation limits

- Exploratory diagnosis of 20 repeatedly used development photographs, not independent test evidence.
- The two archives record identical model and photograph SHA-256 values; this replay does not load model weights or photograph bytes.
- Decoded tensors came from the historical Tesla T4 runtime. Current retained outputs came from a newer CPU execution with different Python and PyTorch versions. Numeric bytes are not identical; measured identities and threshold outcomes are verified separately.
- Only the original-view trained checkpoint has these decoded tensors. Reflected proposals and future newly trained checkpoints are not diagnosed.
- IoU eligibility is a geometric relation to the reviewed annotations. Many eligible proposals can describe the same physical object, and one proposal can overlap multiple references.
- Maximum one-to-one matching is a diagnostic upper bound for each fixed candidate set, not a measured detector result or a deployable oracle.
- A stage label identifies where eligible support disappears in this fixed computation. It does not prove an optimal intervention, correct physical annotations, or recoverable performance without extra predictions.
- No eligible decoded box means no box reaches this reference IoU threshold in the finite decoded tensor; it does not mean the neural network had no visual response.
- The audit does not change the current confidence threshold, suppression policy, labels, retained method, or two-view corner measurements.
