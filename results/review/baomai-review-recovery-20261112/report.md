# Cassette model readiness and recovered corner-review decisions

Repository: `Siuuuers/dl`  
Date: 2026-11-12, Asia/Hong_Kong  
Purpose: repository-targeted recovery record for the interrupted review

## 1. Current decision and recovery status

**Keep the PR35 seed46 checkpoint for assistant-reviewed cassette proposals.** Its current explicit settings are image size **1280**, box confidence **0.13**, NMS IoU **0.70**, and endpoint cutoff **0.50**. PR47 supplied a valid matched training comparison, but its candidate did not meet the declared replacement criterion. No new inference or training was performed during this continuation, and no GitHub push was performed for this recovery.

The original-image review and the numerical replay described below were completed before the executor reset. The previously unpublished annotation parent, local commit `05dcc90`, had been inspected and verified: it preserved 17 supported box corrections and one duplicate deletion. Its complete local repository and evidence capsule subsequently became inaccessible. The original commit and full capsule have **not** been recovered.

The compact correction payload is now reconstructed and verified from remotely pinned original labels and the preserved decisions. All three original label hashes match their pinned dataset records. The reconstructed box-review files for 023 and 031 exactly match the parent SHA256 values recorded before the reset. Independent comparison confirms 17 box changes, one duplicate deletion, nine endpoint-slot changes, exact row maps, valid geometry and preservation of unrelated tokens. The [correction record](correction-record.json) binds every output hash. This recovers the declared corrections, not the old complete repository or evidence capsule; the original PNGs were not downloaded or reviewed again after the reset.

The published starting point remains [PR48](https://github.com/Siuuuers/dl/pull/48), revision `f8d758ade0b721d494306e3002d26a3d56d807cb`, which records the [completed PR47 evaluation](https://github.com/Siuuuers/dl/blob/f8d758ade0b721d494306e3002d26a3d56d807cb/docs/consistency-results-2026-11-12.md).

## 2. The task and target are already settled

The user's clarified scope is to find and localize cassette objects in the supplied baomai D405 images. The existing target is one `cassette_front` class with two actual lower-front corner slots, ordered by image X. Camera mounting, robot integration, and physical calibration are outside this deliverable and do not block image evaluation. Whole-original-image review remains part of the selected workflow.

The branch distinction in the original assignment has already been investigated. The [adopted annotation policy](https://github.com/Siuuuers/dl/blob/f8d758ade0b721d494306e3002d26a3d56d807cb/docs/annotation-policy.md) cites baomai's [pinned data README](https://github.com/an94-doll/embedding_vision/blob/351eb3cf960ccc34829e2bc72e1fa4645acc6295/data/README.md) for the one-class, two-corner target. Historical baomai editor instructions inform the visible-front box convention. The separately documented yolo-pose branch uses front/back classes and five object-relative points; those labels should not be mixed into this task.

Missing points are independently represented as `0 0 0`. A box encloses observed front material; it does not reconstruct hidden dimensions. A confident model coordinate does not establish that a physical corner is visible or correctly located.

## 3. What the current model actually achieves

The selected PR35 checkpoint SHA256 is:

`420c00f73eeba24e587342400878b7d96371d3e745772113809aa705ec8163b1`

The current development reference, `baomai-visible-front-2026-11-11`, contains **663 objects**, **1,228 locatable endpoints**, **98 unlocatable endpoint slots**, and **584 objects with both corners locatable**. Its manifest SHA256 is:

`2e60b3e81195448cc565a1000b4305e3215baba2bf0ee0c976e6551c2e64937c`

The reference's working preferences prioritize missed objects, use **10 px as a provisional image-space objective**, and retain **5 px as a secondary diagnostic**. These radii are neither annotation-placement allowances nor validated deployment acceptance criteria.

At confidence 0.13, using the unchanged matching IoU of 0.50:

| Measure | PR35 result |
| --- | ---: |
| Predictions | 728 |
| Matched reference objects | 643 / 663 = 96.98% |
| Missed reference objects | 20 |
| Scored extras | 85 |
| Scored precision | 643 / 728 = 88.32% |
| Returned locatable endpoints | 1,208 / 1,228 |
| Both corners within 5 px | 529 / 584 = 90.58% |
| Both corners within 10 px | 577 / 584 = 98.80% |
| Miss-free nonempty images | 3 / 17 = 17.65% |
| Nonempty images with neither misses nor scored extras | 0 / 17 |
| Predictions across three empty images | 0 |

The three miss-free images are `new_000010`, `new_000054`, and `new_000058`. They still contain four, three, and two scored extras respectively. “Scored extras” means unmatched predictions under the frozen reference. It does not mean every unmatched prediction has been individually established as a physical false positive.

Before the reset, a standard-library numerical replay reconstructed the confidence-ordered matching from the saved PR35 predictions and frozen reference labels. All 20 image counts and exact assignment identities agreed with the preserved report. The replay also reproduced the corner-group counts, 2/5/10 px successes, and unlocatable-side counts. It loaded neither model weights nor original images and did not produce new predictions.

## 4. The headline corner score has a narrower population

| Locatable corners in the reference | Objects | Matched | Missed | Detection recall |
| --- | ---: | ---: | ---: | ---: |
| Zero | 19 | 12 | 7 | 63.16% |
| One | 60 | 53 | 7 | 88.33% |
| Two | 584 | 578 | 6 | 98.97% |
| Fewer than two, combined | 79 | 65 | 14 | 82.28% |

**Fourteen of the twenty misses are outside the 584-object two-corner denominator.** The pair metric correctly includes all eligible reference objects, including eligible detection misses. Its limitation is its population: it cannot describe completeness for the 79 objects with fewer than two locatable corners.

The model also returns an eligible coordinate on **69 of 77 reference-unlocatable slots among matched objects**. This comprises 20/24 slots in the zero-corner group and 49/53 in the one-corner group. This measures disagreement with reference unlocatability. It is not a hidden-corner geometric error rate, because the reference contains no reliable hidden coordinate. Some inspected examples place points on a hand or gripper; the aggregate alone does not establish that interpretation for every output.

The practical conclusion is that the model is useful for proposing objects and visible corners for review. The existing evidence does not establish complete unattended detection. Reviewing only returned boxes cannot reveal a cassette that has no proposal; the complete original image must remain visible to the reviewer.

## 5. Why PR47 did not replace PR35

| Primary result at confidence 0.13 | PR35 | Matched L4 control | Fourteen-box candidate |
| --- | ---: | ---: | ---: |
| Matched objects / 663 | 643 | 641 | 642 |
| Misses | 20 | 22 | 21 |
| Scored extras | 85 | 88 | 88 |
| 5 px pairs / 584 | 529 | 527 | 530 |
| 10 px pairs / 584 | 577 | 578 | 578 |

The candidate's sole gained match against its L4 control, `056:14`, was already represented by a proposal in both models. Its IoU moved from 0.499891458 to 0.505847964, crossing the fixed 0.50 boundary after a maximum box-edge shift of approximately 0.267456 px. That numerical gain does not establish recovery of a previously unproposed front.

Against PR35, the candidate loses the scored `090:43` match while retaining a nearly identical proposal. Its net gain of one 5 px pair combines three gained and two lost success identities. The 10 px gain reflects assignment among alternatives already present. Review of 14 unique candidate-only proposals established nine existing-front alternatives, four background/rack outputs, one unresolved interpretation, and **no additional supported visible-front recovery**.

The L4 control itself differs from A100-trained PR35. The within-L4 comparison measures the annotation intervention under matched recorded conditions; the PR35 comparison decides operational replacement. Cross-control differences cannot be attributed to the fourteen box edits. Preserve the supported annotations, retain PR35, and keep this fixed experiment closed.

## 6. Recovered six-row corner decisions

The following decisions were reached through original-image review before the executor reset. Coordinates are in the original **1280 × 720** image. A listed coordinate is a locatable corner (`v=2`); “missing” means the canonical `0 0 0` triple. These are explicit point/visibility amendments, not changes hidden inside a box-only operation.

| Image | Original row | Row after the earlier duplicate deletion | Left corner | Right corner |
| --- | ---: | ---: | --- | --- |
| `new_000023` | 33 | 32 | (482, 550) | Missing |
| `new_000023` | 34 | 33 | (486, 580) | Missing |
| `new_000023` | 35 | 34 | (490, 607) | Missing |
| `new_000031` | 22 | 22 | Missing | (742, 458) |
| `new_000031` | 23 | 23 | Missing | (737, 491) |
| `new_000031` | 24 | 24 | Missing | (732, 518) |

Row references must remain explicit. In `new_000023`, original row 22 was the redundant both-missing duplicate of original row 36. The earlier package retained original row 36, its corner tokens, and its supported box correction; it deleted only original row 22. Consequently original row 36 becomes candidate row 35, and original rows 33–35 become candidate rows 32–34. No corresponding deletion occurred in `new_000031`.

The reconstructed files apply these six decisions to the intended surviving physical objects, retain the earlier supported box corrections, and preserve unrelated row tokens. The original-to-candidate row maps, normalized coordinates, canonical missing states, X order and containment were independently checked against the actual files. All classes and box tokens stay identical between each box-review file and its endpoint amendment. The nine slot changes comprise three visible-left to missing changes, three missing-right to visible changes, and three visible-left coordinate corrections; three right slots stay missing. Total visible and missing endpoint counts are unchanged. The integer positions have several-pixel descriptive boundary uncertainty, recorded in correction-record.json; the ranges are not statistical error bars. Earlier original-first peer reviews and subsequent informed verification support these decisions, with no independent human or supervisor certification claimed. The six decisions establish an annotation intervention. Their effect on model detection, localization, or hidden-side output remains unmeasured.

## 7. Useful progress with no new captures

The small recovered correction package is complete and verified. It addresses known contradictory supervision without changing the architecture or claiming an accuracy gain. The earlier completeness review increased the targeted review-history union to **67 of 137 training images**, leaving **70** outside those recorded scopes. These counts describe review history, not fully certified images.

Further review should remain bounded: select a small cohort before viewing predictions, inspect whole originals first, then check omissions, duplicates, visible-front extent, and actual corner identity. Keep unresolved cases explicit. A mechanical verifier can prove preservation and valid encoding; it cannot establish the physical truth of a point on an occluder.

After enough relevant corrections accumulate, a training comparison should answer one declared question, with fixed data, initialization, recipe, evaluation reference, and acceptance rule. Do not automatically retrain after each small label edit. The completed higher-resolution, threshold, and earlier label-extension experiments already provide decisions; repeating searches on the same development images would not create independent evidence.

The separate upstream 320-frame intake also has documented annotation, overlap, and capture-grouping limitations. It is potential supervision material, not automatically an independent test set. Reusing the current development images, augmenting them, or generating more predictions cannot establish reliability on unseen capture sessions.

## 8. Sources and operational continuity

Use the [explicit reviewed-proposal command](https://github.com/Siuuuers/dl/blob/f8d758ade0b721d494306e3002d26a3d56d807cb/docs/predict-pose-folder.md#assistant-reviewed-proposals). Generic runtime defaults still select epoch173/0.25; obtaining a newer result archive alone does not select PR35 automatically.

The [PR48 evidence directory](https://github.com/Siuuuers/dl/tree/f8d758ade0b721d494306e3002d26a3d56d807cb/results/verification/baomai-consistency-results-20261112) preserves the published comparison and replay provenance. The [quality guide](https://github.com/Siuuuers/dl/blob/f8d758ade0b721d494306e3002d26a3d56d807cb/docs/model-quality.md) and [reference manifest](https://github.com/Siuuuers/dl/blob/f8d758ade0b721d494306e3002d26a3d56d807cb/data/references/visible-front-2026-11-11/manifest.json) preserve the settled scope and provisional objectives. The separately identified recovery payload contains the corrected label bytes and its own verification record. It does not replace the inaccessible complete historical capsule.

## 9. Use and publication status

The recovery package contains three pinned original TXT files; the three box-review TXT files; two later endpoint-amended TXT files; a complete before/after correction record; the readiness snapshot; this report; a draft PR description; and the bounded reconstruction utility. The two endpoint files supersede only their corresponding box-review files. Image045 keeps its box-review file. These are the newly reviewed files, not a complete training dataset: the existing eight-file PR35 annotation overlay remains a separately pinned parent and is not duplicated here.

From the extracted folder, run:

```bash
python3 -B rebuild_review.py --check
```

This verifies the reconstruction from the included pinned source labels. It does not run inference, recreate the lost full evidence archive, or certify every legacy annotation. The supplied files are deliberately not represented as a standard training-stager manifest. Existing training code, evaluation references, selected checkpoint and runtime defaults are unchanged. A future training experiment needs explicit candidate integration and a fixed protocol.

No GitHub upload or draft PR was created during this continuation. The recovered publication record states that automatic approval review rejected the earlier upload because explicit approval for its exact external payload was required. The current archive and draft description make the replacement recovery payload concrete for approval. Its proposed destination is a new draft in Siuuuers/dl, based on PR48; it does not merge PRs, replace model weights or trigger training.
