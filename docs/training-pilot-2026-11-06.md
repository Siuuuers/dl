# Training-label review and proposed duplicate removal

## Result

A targeted review of original training images found **11 duplicate physical-instance annotations in five images**. The proposed overlay removes the redundant rows and preserves every retained row byte-for-byte. Training objects would change from **3,772 to 3,761**, with all **6,839 visible endpoint labels retained**. The frozen source labels, validation set and checkpoint are unchanged.

This is a new candidate derived from the reference at `1c201e35a33727ac0ae59f1d11a3d7fbb72b5ae7`. It does not recover or validate PR #2's historical 3,765-object candidate. No model inference or training ran during this stage, so a performance benefit has not been measured.

The [candidate manifest and five labels](../data/candidates/dedup-2026-11-06/README.md) are ready for review. The [source-hashed evidence record](training-pilot-2026-11-06.json) contains the selection, numeric overlap audit, individual visual decisions, source image/label hashes and proposed row mappings. These are assistant review decisions, not a claim of human approval.

## Scope and method

The review began with seven originals spanning dense new scenes, occlusion, truncation, ordinary visible corners, old-source scenes and an empty scene. It added `new_000075` after a top-border duplicate concern, then five images from a numeric overlap queue. Actual PNG bytes and dimensions were verified against `data/manifest.json` for all **13 selected training images**. Across the review team, **54 target rows** and one full-frame empty control were inspected. The images contain 409 reference rows in total; this was not a complete per-row audit of those 409 rows or of the full training dataset.

Original pixels were inspected before reference overlays. Duplicate decisions require agreement on the same physical front face from a primary and an independent agent review. A box-overlap score only selected cases for inspection. Model predictions were not used to move corners or decide annotation truth.

The mechanical audit checked all 137 training label files against their frozen hashes and examined **75,145 within-image row pairs**. It found zero exactly repeated text or numeric rows, but twelve pairs with box IoU at least 0.50. Different numeric boxes can still describe the same physical cassette. The visual review also identified a duplicate pair below that threshold, so the queue does not establish that all remaining annotations are correct.

## Exact proposed changes

All row numbers below refer to **one-based original source lines**. Remove the listed rows together; do not interpret later deletions against a file whose row numbers have already shifted. The manifest also records each retained row's resulting candidate position and exact text.

| Training image | Remove original rows | Retain original rows, in corresponding order | Objects before → after | Physical evidence |
| --- | --- | --- | --- | --- |
| [new_000019](../data/images/train/new_000019.png) | 56, 57, 58, 59, 60 | 63, 64, 65, 66, 67 | 68 → 63 | The same five successive lower-center fronts are represented twice behind the gripper. |
| [new_000075](../data/images/train/new_000075.png) | 1, 2, 3 | 26, 24, 25 | 26 → 23 | Three top-border fronts each have two annotations following the same lower contour and side boundaries. |
| [new_000102](../data/images/train/new_000102.png) | 15 | 54 | 54 → 53 | Both rows describe one top-truncated front; the next distinct front lies below its lower seam. |
| [new_000112](../data/images/train/new_000112.png) | 35 | 38 | 38 → 37 | Both rows describe the same lowest front behind the gripper, with its left corner hidden and right exposed. |
| [new_000051](../data/images/train/new_000051.png) | 21 | 22 | 23 → 22 | Both rows describe one top-truncated front with the same sloping lower contour. |

Every removed row has both endpoints recorded as `0 0 0`. The retained representation already contains the visible endpoints supported by the image. No new coordinates or visibility flags are invented. Retaining these existing rows is not a new certification of their exact box extent or pixel accuracy.

| Quantity | Frozen reference | Proposed overlay applied |
| --- | ---: | ---: |
| Training images | 137 | 137 |
| Training objects | 3,772 | 3,761 |
| Training visible endpoints | 6,839 | 6,839 |
| Training unlocatable endpoints | 705 | 683 |
| Empty training images | 13 | 13 |
| Validation images / objects | 20 / 664 | 20 / 664 |
| Validation visible / unlocatable endpoints | 1,227 / 101 | 1,227 / 101 |

The reduction of 22 unlocatable endpoint entries comes solely from deleting eleven redundant instance rows. It is not a claim that hidden physical corners became visible.

## Overlapping pairs that must remain

Two queued pairs describe **different adjacent cassettes**:

- [new_000030](../data/images/train/new_000030.png), rows 6 and 29: their visible lower-left corners lie about 32.93 px apart vertically on consecutive fronts. Row 29's broad box includes part of the upper front, creating an extent-review issue. Both instances remain.
- [new_000123](../data/images/train/new_000123.png), rows 32 and 33: the lower-left corners lie about 24.25 px apart on distinct slanted fronts. Row 33's right corner is beyond the image boundary and remains `0 0 0`. Both instances remain.

Their box IoUs are approximately 0.5534 and 0.5428. In contrast, the confirmed `new_000019` duplicate at rows 60/67 has IoU about 0.3410. **Do not run prediction-style suppression on ground-truth annotations to remove overlapping rows.**

After applying the proposed overlay in memory, 74,636 within-image pairs remain. The only two pairs at IoU ≥ 0.50 are the distinct pairs above. This is a mechanical inventory result, not proof that every lower-overlap annotation is correct.

## Other reviewed cases and the remaining annotation decision

Selected ordinary points in `new_000043`, `old_000003` and `old_000018` provided no supported coordinate changes. A top-truncated cassette can still have both lower corners visible. Conversely, the occluded and out-of-frame sides in selected `new_000012` and `new_000128` rows remain unlocatable. Very thin fragments in `new_000128` are unchanged pending a clear inclusion rule. The full-frame `new_000131` empty control remains empty: rack, gripper and support structure are not cassette fronts.

There is useful historical evidence for the box convention. The upstream [annotation-editor instructions at `6f7112a`](https://github.com/an94-doll/embedding_vision/blob/6f7112a553e1759fc755bda1f18ee56cb34dd315/detect/data/data_last/picture/work/ANNOTATION_EDITOR.md) say to draw the visible front-face extent. That belongs to the earlier epoch-127/277-image workflow, whose keypoint convention also differs. It does **not** establish inheritance by the current epoch-173/157-image dataset. The current [baomai data README](https://github.com/an94-doll/embedding_vision/blob/351eb3cf960ccc34829e2bc72e1fa4645acc6295/data/README.md) remains authoritative for the two actual lower-front corners ordered by image X.

Before changing box extents or inclusion of tiny/fully unlocatable fragments, confirm whether the current rule is visible front-face boxes excluding the rack, and whether an identifiable front remains an instance when neither target corner can be located. This uncertainty does not require retaining two rows for the same physical front.

## Verification

Run from the repository root using Python 3:

```bash
python3 scripts/verify_pose_candidate.py
```

The read-only verifier checks all 157 frozen reference label hashes, schemas and counts; train-only candidate mappings; source image-hash metadata; exact row texts and row remapping; both-unlocatable removals; byte-preserving deletion; absence of unlisted overlay labels; and aggregate counts. It reports code hashes and explicitly states that it does not verify image bytes, stage images, build a dataset or assess physical correctness. The declared Git source revision is provenance metadata; reference bytes are enforced through hashes.

The real candidate passed. Four bounded invalid-input probes were rejected: a changed retained coordinate even after its file hash was updated, an incorrect retained-row mapping, an extra validation label file, and a changed reference label. The original reference and proposal files were not modified by those probes. The image-byte checks for the 13 inspected originals are a separate part of the review evidence.

## Next experiment

1. Review this exact eleven-row proposal against the linked originals and row evidence. Keep the decisions about duplicate identity separate from unresolved box-extent changes.
2. Materialize a separate complete training dataset with the proposed five label replacements, all other source labels intact, and the same frozen split. Verify actual image bytes before training. This overlay alone is not a runnable dataset.
3. Verify that the trainer resolves the intended labels. In the pinned Ultralytics implementation, image paths determine their sibling label paths; pointing a candidate YAML at the original `images/` tree can load original labels. Use a separate paired `images/` and `labels/` tree and check the resolved paths. Do not overwrite the frozen manifest or weaken the baseline preflight.
4. Follow [PR #1](https://github.com/Siuuuers/dl/pull/1)'s controlled comparison: unchanged epoch 173, a matched fine-tune on unchanged data, and a fine-tune on the proposed data, holding the declared settings fixed. Add explicit candidate-checkpoint provenance support to evaluation before that stage. Use validation for development and retain the baseline if evidence is inconclusive.
5. Agree on an application tolerance and obtain independent scenes before claiming the model is good enough for its intended use. The prior validation scores and illustrative pixel radii are unchanged by a label-only proposal.

The completed validation review in [PR #8](https://github.com/Siuuuers/dl/pull/8) and this training pilot should be reused in later conversations. Historical candidate recovery and native-browser Save/reopen acceptance remain separate unresolved items; neither is evidence that this candidate has been trained or deployed.
