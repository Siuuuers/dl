# Six-image consistency annotation candidate

**Status: proposed annotations; model effect unmeasured.** The [completed review](../../../docs/training-consistency-review-2026-11-11.md) supports 14 box-only replacements in three existing training images. The selected PR35 checkpoint and its explicit 0.13 reviewed-proposal setting remain current. No training or inference was run, and no new GPU job is requested here.

## Accepted delta

| Image | Existing source rows changed | Reason |
| --- | --- | --- |
| `new_000018` | 24, 25, 26, 32, 33, 34 | Bound exposed fronts beside the gripper/thumb; remove hidden width and neighboring material |
| `new_000034` | 55, 56, 57 | Bound three narrow exposed fronts beside the gripper |
| `new_000049` | 9, 14, 24, 25, 26 | Remove excess rack extent in row9; expand four lower-center boxes to visible front material |

The new delta contains **14 bbox replacements, no additions/deletions and no endpoint edits**. Every existing class and corner token in the three files remains exact, including missing triples. Retained row order and all other source bytes stay unchanged. Nine other proposed complete box replacements remain held; unresolved identity and physical-corner questions are recorded separately. This is an incremental proposal, not a full policy migration or supervisor certification.

## Composition

This overlay is defined against the frozen original labels. It inherits the exact eight-file [annotations-2026-11-10](../annotations-2026-11-10/README.md) candidate used for the selected PR35 model, including its label bytes, change objects and twelve accepted decision objects. The three new files are disjoint from those eight. The cumulative overlay therefore has **11 changed training label files**, with **17 replacements, five additions and six deletions against the original dataset**. The additions/deletions and three earlier replacements are inherited; this review changes only fourteen boxes.

The non-adopted `partial-fronts-2026-11-11` and `visible-extent-2026-11-11` extensions remain separate and are not accumulated. Original labels and all historical candidates remain intact.

| Effective dataset | Images | Objects | Visible endpoints | Missing slots | Empty images |
| --- | ---: | ---: | ---: | ---: | ---: |
| Frozen training original | 137 | 3,772 | 6,839 | 705 | 13 |
| Selected PR35 eight-file overlay | 137 | 3,771 | 6,841 | 701 | 13 |
| This cumulative candidate | 137 | 3,771 | 6,841 | 701 | 13 |
| Original in-training validation | 20 | 664 | 1,227 | 101 | 3 |

The separate 663-object corrected development reference is unchanged. It is not inserted into the in-training validation set.

## Inspect and verify

- [Manifest and full original-to-candidate row maps](manifest.json)
- [Accepted decisions, inheritance and retained uncertainty](review.json)
- [Unchanged annotation verifier result](verification.json)
- [Independent composition/input verification](composition-verification.json)
- [Prospective selection, final adjudication and evidence capsule](../../../results/review/baomai-training-consistency-20261111/README.md)

From the repository root:

```bash
python -B scripts/verify_pose_annotation_candidate.py \
  --candidate data/candidates/consistency-2026-11-11/manifest.json
```

This checks every original label hash, declared operation, unmodified row byte, row map, geometry and count. It does not inspect physical image content. The separate composition check additionally verifies the six actual original PNGs, the pre-image selection, exact inheritance and all 270 existing corner triples in the review cohort. Evidence reproduction commands are in the linked evidence README.

The existing generic staging script accepts this manifest explicitly if a later fixed study is selected. This review has not staged a new training dataset or prepared a training notebook. The completed visible-extent study wrapper names its historical candidate and must not be repurposed silently.
