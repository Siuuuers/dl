# Reviewed training annotations

**Status: accepted as a bounded annotation intervention; model benefit is unmeasured.** Use this overlay for the prospective seed45 control/candidate experiment described in [the training handoff](../../../docs/training-annotations-2026-11-10.md).

The [manifest](manifest.json) reconstructs every changed label from explicit original-row deletions, replacements and appended additions. The [review](review.json) records original-image hashes, accepted decisions, source row identities, manual geometry and unresolved cases. Frozen labels are unchanged.

| Change | Count |
| --- | ---: |
| Training label files changed | 8 |
| Omitted fronts added | 5 |
| Non-target gripper rows deleted | 4 |
| Redundant aggregate/duplicate rows consolidated and deleted | 2 |
| Retained rows amended | 3 |
| Existing corner marks transferred exactly | 2 |
| Original visible corner marks preserved | 6,839 |
| New visible corner marks | 2 |

Training totals change from **3,772 objects / 6,839 visible endpoints / 705 missing endpoints** to **3,771 / 6,841 / 701**. All 137 training images, 13 empty training images and the complete original 20-image validation split remain fixed. Added identifiable fronts with hidden physical corners retain `0 0 0` on the hidden sides.

This candidate is independent of the historical eleven-row deduplication and sixteen-edge box candidates. Those changes are not silently accumulated. The original labels are the control. This is not a completed migration of every label to the visible-front policy.

## Verify and stage

From a checkout containing the hydrated original images:

```bash
python scripts/verify_pose_annotation_candidate.py \
  --candidate data/candidates/annotations-2026-11-10/manifest.json

python scripts/stage_pose_training.py \
  --candidate data/candidates/annotations-2026-11-10/manifest.json \
  --output ../reviewed-annotations-staged
```

The verifier checks exact source/candidate bytes, full row maps, accepted review provenance, canonical visibility, geometry and donor transfers. The complete staging pass additionally checks all original PNG bytes and creates independent control/candidate copies. Neither step runs training or certifies the untouched labels visually.

Only `reviewed-annotations-seed45-v1` is currently registered for this candidate kind. Train both arms freshly from epoch173 and keep the final epoch-30 checkpoints. The [GPU notebook](../../../notebooks/baomai_reviewed_annotations.ipynb) provides the complete run-and-return procedure.
