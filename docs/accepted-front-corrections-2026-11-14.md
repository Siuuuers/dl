# Train the accepted front corrections

**The next run is a controlled annotation comparison.** It keeps the existing one-class, two-corner model and all original images. It applies supported corrections where no physical question remains, while preserving exact PR35 labels on every unresolved training photo in both arms. This tests a limited correction set; it does not certify the whole dataset or establish deployment readiness.

Use [baomai_accepted_fronts.ipynb](../notebooks/baomai_accepted_fronts.ipynb). The separate [full-review notebook](../notebooks/baomai_observed_fronts.ipynb) preserves its original blocked contract and is a historical review handoff.

## Decisions from the physical clarification

| Question | Decision | Effect |
| --- | --- | --- |
| Broad upper region in new_000037, atlas p.6 / full photo p.39 | The user identifies it as wall. | Close this exact non-target question; add no cassette. |
| Upper structure in new_000017, p.21 | Independent original-image review supports continuous fixture/backing geometry. This is an inference, not a user-confirmed wall identification. | Close the non-target question; add no cassette. |
| Shadowed material in the new_000030 opening, p.7 / full photo p.32 | Cassette material is confirmed. The lower triangle's front ownership and instance count remain uncertain. | Do not split, add or delete an object from color alone. |
| Thin band in new_000117, p.8, with analogous new116/new123 cases | Small height does not establish whether it is another front or the upper cassette's lip/underside. | Keep the ownership question. |
| Top strips in new_000128, p.9 | Neighboring sizes and positions support hypotheses, but the clipped image does not reveal an unambiguous seam. | Preserve uncertainty; do not infer hidden corners. |

The additive [physical decision record](../results/review/physical-adjudication-20261114/decisions.json) binds the original photos and prior review. It changes no label rows. After the two non-target closures, **24 training case groups across 21 photos** remain. The separate development reference still has **13 held rows across 9 photos**.

No brightness requirement, white-shoulder requirement, minimum pixel height, or in-rack-only rule has been adopted. An identifiable cassette front can be dark, clipped or held away from the rack.

## What the experiment changes

For every training photo, the reference arm uses the exact effective PR35 label bytes. The candidate uses the full reviewed label bytes when that photo has no remaining physical hold; otherwise it uses the same PR35 bytes as the reference. The whole unresolved photo is preserved, rather than selectively deleting a doubtful object.

| Training inventory | Fresh PR35-label control | Limited candidate |
| --- | ---: | ---: |
| Photos | 137 | 137 |
| Objects | 3,771 | 3,788 |
| Locatable corner slots | 6,841 | 6,858 |
| Missing corner slots | 701 | 718 |
| Empty photos | 13 | 13 |

There are **116 photos eligible for reviewed labels**, of which **71 actually differ from the PR35 control**; 45 already match it. The other 21 photos retain exact legacy supervision in both arms. The candidate has 126 objects with no locatable corners, 466 with one and 3,196 with two.

The retained legacy photos are new_000001, new_000005, new_000024, new_000030, new_000033, new_000034, new_000041, new_000042, new_000044, new_000049, new_000062, new_000074, new_000094, new_000115, new_000116, new_000117, new_000121, new_000123, new_000124, new_000127 and new_000128. Their remaining case IDs are derived from the verified prior review and the two explicit adjudications, then recorded in the new [candidate](../data/candidates/accepted-front-corrections-2026-11-14/manifest.json) and [protocol](../experiments/accepted-front-corrections-2026-11-14.json).

The refreshed 166-page atlas records the resolved wall on page 6 and the remaining ownership questions on pages 7–9. Its [source/page audit](../results/review/physical-adjudication-20261114/atlas-audit.json) and [render verification](../results/review/physical-adjudication-20261114/atlas-qa.json) preserve the exact original photos, label versions and page mapping. The full atlas continues to show the **3,791-object full review candidate**, including held rows. It is not a preview of the narrower 3,788-object training treatment. Its review questions and page numbers remain useful.

This scope follows the existing [annotation policy, section 5](annotation-policy.md#5-empty-missing-and-unresolved-are-different-states). Legacy uncertainty remains in the experiment; preserving it equally does not guarantee that its effect cancels.

## Why not remove dark or small targets?

Deleting an object row is not a general ignore-region mechanism. The pinned [Ultralytics detection loss](https://github.com/ultralytics/ultralytics/blob/v8.4.182/ultralytics/utils/loss.py) assigns positives from the supplied object labels; predictions left unassigned receive background classification supervision. Thus convenience deletion can teach the model to suppress a real difficult target.

For an identifiable front whose actual corners cannot be located, canonical missing corner slots preserve the box/class target. In the pinned pose loss, a missing slot contributes no corner-coordinate loss, but it is still a negative target for corner-presence confidence. It is not a universal ignore flag. The [YOLO26 pose assigner](https://github.com/ultralytics/ultralytics/blob/v8.4.182/ultralytics/utils/tal.py) supports objects with no visible keypoints by falling back to box IoU.

These representations address different questions:

- **Known front, known observed extent, missing corners:** keep the object and mark only the unavailable corners missing.
- **Unknown front ownership or instance count:** preserve the physical review issue; missing corners do not decide whether an object exists.
- **Confirmed wall, rack, hand or gripper:** do not add a cassette label for it.

Neighboring geometry can help identify an object, but it does not make an occluded feature observable. Equal physical size also does not imply equal pixel size under perspective.

## Run the notebook

1. Open the accepted-fronts notebook in a fresh GPU runtime.
2. Run all cells in order. Input verification checks exact source, photo, label, adjudication, protocol and reference identities before trainer installation.
3. Allow the four-batch resource trial to finish. It does not initialize either complete training run.
4. Let both arms complete 30 epochs in the same runtime. Each independently starts from the original epoch173 checkpoint with seed46, batch4, input1280, FP32 and AdamW at 0.0001.
5. Keep the saved results ZIP and return it for paired evaluation. The final epoch checkpoint is preselected; do not choose a favorable epoch afterward.

The notebook preserves completed runs and verifies their identities before reuse. It saves to Drive in Colab and provides a download. If an input check fails, keep the error and use the correct pinned notebook; do not edit labels or status fields to make it pass. If the resource trial fails, preserve its receipt and log for diagnosis.

No GPU is available in the preparation workspace, so this handoff has not executed training. A valid notebook and dataset are prerequisites for an experiment, not evidence that the model improved.

## Evaluate the result without changing the question

The fresh matched control is the primary comparison. The retained PR35 checkpoint is an additional deployment-continuity comparator. Use the fixed box confidence 0.13, endpoint confidence 0.50, NMS IoU 0.70, input1280, association IoU 0.50 and 5/10 original-image-pixel radii.

Score all comparators on the older frozen visible-front-adjudication reference, then separately on the completion reference as a mandatory diagnostic. Both have 20 photos, 663 objects, 1,227 locatable slots and 584 objects with two locatable corners. The completion reference changes nine boxes; it does not resolve the 13 held rows. If the reference choice reverses the apparent winner, report the comparison as inconclusive.

Report missed objects, duplicates and other extra predictions, empty-scene outputs, and recovery by zero/one/two-locatable-corner groups. Include eligible missed objects in both-corner success denominators. Keep returned coordinates on reference-missing sides separate from localization error: no hidden reference coordinate exists to measure their distance against.

Inspect gained and lost object identities. Thin boxes are particularly sensitive to small boundary changes: for two otherwise identical boxes of height h displaced vertically by d, their IoU is (h-d)/(h+d) when 0 <= d < h. A hypothetical five-pixel box displaced by two pixels gives IoU 3/7, below 0.50, even though the displacement is small. A changed match alone is not proof of a newly recovered or lost physical object.

These photos and thresholds have already been used in development. One paired seed can test this correction set under its stated conditions; it cannot supply independent deployment evidence. PR35 stays selected until the completed comparison supports a reviewed change.
