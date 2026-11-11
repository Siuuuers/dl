# Reviewed training annotations

**Status: both planned annotation comparisons are complete.** The [seed45 result](../../../docs/annotation-comparison-2026-11-11.md) and [seed46 confirmation](../../../docs/annotation-confirmation-2026-11-11.md) preserve this same eight-file overlay. At the primary operating point, the seed46 candidate ties its control at 24 misses and reduces extras from 58 to 52; a strict miss reduction does not repeat, while primary extra reduction is favorable in both pairs. The candidate improves the seven chosen primary headline measures against epoch173 and remains a useful experimental checkpoint. Epoch173 is still the working default while the [26-object queue](../../../results/evaluation/baomai-annotations-seed46-20261111/review-queue.md) receives focused review. No new seed run or label change is requested by this result.

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

The explicitly registered recipes are `reviewed-annotations-seed45-v1` and `reviewed-annotations-seed46-v1`. Both completed studies started each arm freshly from epoch173 and evaluated finalized epoch-30 checkpoints. The [seed46 GPU notebook](../../../notebooks/baomai_reviewed_annotations_seed46.ipynb) remains the recorded reproduction procedure, including PR34's completed-run Drive persistence; the original seed45 notebook and its source pin remain unchanged.
