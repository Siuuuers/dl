# Box refinement candidate — 2026-11-07

**Status: proposed for a controlled training comparison; no improved model established.** This separate overlay changes 16 box edges in three existing training images after a complete review of 104 objects and one genuine empty control. The original dataset and the eleven-row deletion candidate remain separate.

| Training image | Original rows | Accepted single-edge changes |
| --- | ---: | --- |
| `new_000030` | 29 | Rows 9, 10, 11, 23, 24, 25, 26: bottom; row 29: top |
| `new_000012` | 38 | Rows 15, 16: bottom; rows 26, 27: left; rows 37, 38: right |
| `new_000128` | 37 | Rows 10, 11: bottom |
| `new_000131` | 0 | Empty control retained; no overlay file needed |

[manifest.json](manifest.json) records source hashes, source/candidate row text, exact proposed pixel coordinates and reasons. [review.json](review.json) contains all three primary ledgers and the independent final adjudication. The [physical evidence ZIP](../../../results/review/baomai-box-supervision-20261107/evidence.zip) contains original pixels, source labels and the review crops.

All object rows, row order, class tokens and both endpoint triplets are preserved. Unedited rows remain byte-identical; only the selected box axis is serialized for each changed row. The 0.0001-pixel tolerance handles decimal serialization, not annotation accuracy. No hidden corner is inferred and no point is snapped to a box edge.

Seven instance identities and several legacy bounds remain unresolved and unchanged. This is an incremental experiment under [the annotation policy](../../../docs/annotation-policy.md), not full policy certification or human approval. It does not create a new validation or test reference.

Both arms retain 137 training images, 3,772 objects, 6,839 visible endpoints, 705 missing endpoint slots and 13 empty scenes. The unchanged validation split has 20 images, 664 objects, 1,227 visible endpoints, 101 missing slots and three empty scenes.

## Verify and run

```bash
python scripts/verify_pose_box_candidate.py
python scripts/stage_pose_training.py \
  --candidate data/candidates/box-refinement-2026-11-07/manifest.json \
  --output ../box-supervision-staged
```

The verifier checks all original labels, policy/review hashes, the complete changed-edge inventory, unchanged class/point tokens, geometry and unchanged counts. Staging additionally verifies every original PNG and both copied datasets. These checks do not prove the physical judgments.

Follow [the comparison plan and GPU handoff](../../../docs/box-supervision-2026-11-07.md). The explicit `box-supervision-warmup-v1` recipe and this manifest must be selected together. Keep epoch173 in use until the predeclared comparison passes.
