# R44 — qualify supplementary photos, then measure their training value

**Next step: adjudicate the remaining physical boundaries before adding these photos to training.** R44 corrects annotation geometry and records uncertainty; it does not demonstrate a new model improvement. Keep the present box-and-two-corner task and model architecture while testing whether qualified supplementary photographs help.

## What the documented training history establishes

The pinned metadata and training receipts support this scoped history:

| Stage | Training / development photos | Additional `frame_*` photos in documented training | Role |
| --- | ---: | ---: | --- |
| Original epoch173 | 137 / 20 | 0 | Original cassette checkpoint, initialized from official COCO pose weights. |
| Selected PR35 seed46 candidate | 137 / 20 | 0 | Thirty-epoch fine-tune from epoch173; label edits within existing photos. |
| PR123 / R37 candidate | 137 / 20 | 0 | Controlled thirty-epoch run; checkpoint later used for R41 mining. |

Sources: [upstream training configuration](https://github.com/an94-doll/embedding_vision/blob/351eb3cf960ccc34829e2bc72e1fa4645acc6295/metadata/training_config.json), [frozen image manifest](https://github.com/Siuuuers/dl/blob/a4ead2ee8de56a5e9e125947b78d09b1fce84b92/data/manifest.json), [PR35 training record](https://github.com/Siuuuers/dl/blob/a4ead2ee8de56a5e9e125947b78d09b1fce84b92/docs/training-annotations-seed46.md), and [PR123 return audit](https://github.com/Siuuuers/dl/blob/a4ead2ee8de56a5e9e125947b78d09b1fce84b92/results/evaluation/pr123-r37-20261126/archive-audit.json).

Both fine-tunes document 4,110 training-image presentations: 137 photos × 30 epochs. R37's 31 `old_*` photos remained in training with predecessor labels in both arms; exclusion from the new annotation treatment did not remove those photographs. The [R37 manifest](https://github.com/Siuuuers/dl/blob/a4ead2ee8de56a5e9e125947b78d09b1fce84b92/data/candidates/front-review-2026-11-26-r37/manifest.json) and [R38 analysis](https://github.com/Siuuuers/dl/blob/a4ead2ee8de56a5e9e125947b78d09b1fce84b92/docs/front-review-r38.md) preserve that distinction.

The separate upstream pool contains **320 `frame_*` photographs**. Its documented use here is inference and annotation review: 12 selected photographs currently form the development review. These are existing upstream captures, not renamed `new_*`/`old_*` files. Other upstream projects reused this pool; therefore “never used anywhere for training” would be too broad. See the [pinned intake record](https://github.com/Siuuuers/dl/blob/a4ead2ee8de56a5e9e125947b78d09b1fce84b92/docs/additional-frame-intake-2026-11-07.json).

Absence of these 320 photos from the documented checkpoint lineage makes supplementary training a reasonable experiment. It does **not** prove that this absence caused the present misses, duplicates, or annotation errors. The original historical training sessions were not independently rerun during this review.

[PR131's returned commit](https://github.com/Siuuuers/dl/commit/7b6c312c492a65a216a678b855f62cb63ee2bdd4) supplies the byte-identical R43 CPU review package, not a newly trained checkpoint. The R44 upstream recheck on **2026-11-26** still found `baomai` at [`351eb3cf960ccc34829e2bc72e1fa4645acc6295`](https://github.com/an94-doll/embedding_vision/tree/351eb3cf960ccc34829e2bc72e1fa4645acc6295); no newer branch-head content was found in that check.

## What R44 makes ready—and what remains held

The exhaustive R44 pass changes geometry on **100 source rows**, comprising **35 individual physical corner locations and 122 box boundaries**. Each corner location is one left or right point's x,y coordinates, not a complete left-and-right pair. These are annotation changes, not learned model changes. The current review still contains **328 held source rows** across 11 photographs; see the [R44 manifest](../data/reviews/additional-fronts-2026-11-26-r44/manifest.json).

Only **`frame_0383` has no unresolved fields**, containing 43 fronts in **one photograph**. Forty-three objects do not provide 43 independent capture conditions. No supplementary training split has been assigned by this review.

The most useful next human decisions are:

1. **`frame_0160`, source row 37, `ymin`.** This is that photograph's only remaining hold. Identify the highest visible material belonging to this front, including its own molded rim while excluding the receding upper plane. Resolve it from the original pixels or explicitly retain the hold.
2. **`frame_0001`, source row 11, upper extent.** Use this representative difficult case to clarify how the owned shaded front is distinguished from the neighboring rim and receding surface. Apply the resulting reasoning case by case; it does not authorize a fixed inset or automatic changes to other rows.

The [PR30 policy](https://github.com/Siuuuers/dl/blob/fda37dc364f8c6ea54ad23819482dbf21be0534a/docs/annotation-policy.md) and [reviewed examples](https://github.com/Siuuuers/dl/blob/fda37dc364f8c6ea54ad23819482dbf21be0534a/docs/reference-amendment-2026-11-10.md) include both removal of definite rack overhang and restoration of owned front material omitted by thin-band boxes. Consequently, neither “gray means rack” nor “crop to the bright stripe” is a valid general rule.

A diagnostic outline can show the observed contour and uncertain segments beside the original image. Store it as review evidence, separately from the eleven-token pose labels. It requires no segmentation model or architecture change. Keep unavailable physical corners canonical; the [labeling guide](https://github.com/Siuuuers/dl/blob/a4ead2ee8de56a5e9e125947b78d09b1fce84b92/docs/front-labeling-guide.md) explains the distinction between a physical corner, a box corner, and an occluder boundary.

## Gate for the next controlled notebook

Export only whole photographs whose visible target inventory and required geometry have passed the agreed review gate. **Do not train a held photograph using only its accepted rows:** remaining visible cassettes would become unlabeled training background. Preserve held images and their records outside the training export.

Once a useful supplementary set is qualified, freeze its originals, complete labels, hashes, provenance, and split assignment. Preserve the existing 20 development photos and their references outside training. Freeze the same 137-photo baseline supervision in both experiment arms:

| Arm | Supervision |
| --- | --- |
| A: control | The declared 137-photo baseline and its exact labels. |
| B: supplement | Exactly A, plus the qualified supplementary photographs. |

Both arms should load the same pinned initialization and use the same declared seed, architecture, augmentation policy, optimizer configuration, and inference settings. Match the **optimizer-update budget**, or explicitly report additional training exposure as part of the treatment. Equal epochs with a larger dataset do not imply equal optimization effort. Record image presentations, effective batch size, and any supplementary-image oversampling.

## Keep the outcomes distinguishable

Evaluate both arms against the same frozen references and matching policy, retaining prediction caches and settings. Report separately:

- **Missed references:** visible target fronts without an accepted prediction match.
- **Duplicates:** redundant predictions for the same physical front, distinguished from legitimate adjacent fronts.
- **Other unmatched predictions:** outputs lacking an accepted reference match, with representative cases inspected for localization, ownership, or reference problems.
- **Corner accuracy:** visible-corner success at 5 and 10 native-image pixels, with explicit denominators. Report matched-object accuracy alongside an end-to-end measure that does not silently discard missed objects; unavailable reference corners are excluded from location scoring.

Replaying unchanged predictions against corrected labels measures reference revision, not learning. A training comparison needs newly produced predictions from both arms against the same references.

The frozen 20-image split has been used for repeated development decisions, and the 12 selected review images include disagreement-mined cases. Neither constitutes an independent generalization test. Capture/session grouping is unavailable, and nonidentical image hashes do not establish independent scenes. The next experiment can support a controlled development conclusion; demonstrating performance on new sessions will require separately collected, independently held captures.
