# Aperture and observed-front review candidate

Status: **reviewed, not released for training**. No GPU run or inference was performed.

[Review report](../../../docs/aperture-label-review-2026-11-14.md) · [Accepted operations](../../../results/review/aperture-labels-20261114/accepted-review.json) · [Remaining cases](../../../results/review/aperture-labels-20261114/review-status.json)

This full overlay preserves the frozen fixed-geometry parent and adds182 row refinements (376 box bounds, with six explicitly reviewed endpoint-row edits), four observed-front additions and seven non-target/duplicate removals across42 training images. There are71 TXT payloads relative to the original manifest. Each file is a complete replacement for the corresponding original training label. Never mix these files blindly with older overlays.

Effective training counts:137 images /3,770 objects /6,854 locatable endpoint slots /686 missing slots /13 empty images. Original validation stays unchanged; the atlas uses the separate development reference. Existing weights have not learned this candidate.

Unresolved material ownership, front counts and exact boundaries are recorded explicitly. Fixed physical size and stack support are not sufficient to label an unseen corner as visible. new030 cassette material is user-confirmed, while its internal seam and box remain under review.

The immutable parent and exact accepted review are pinned by `manifest.json`. `release_ready` is false and `training_recipe_id` is null. Build a released recipe only after remaining decisions or explicit image quarantine.

```bash
python -B scripts/compose_observed_pose_review.py --root . verify \
  --candidate data/candidates/aperture-reviewed-labels-2026-11-14
```
