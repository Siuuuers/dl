# Observed label refinement with fixed cassette geometry

**Reviewed draft; not a complete corrected release. No model has been trained on it.**

This is the current review revision over the immutable [reconciled parent](../reconciled-labels-2026-11-14/README.md). It includes69 further box operations, one existing visible left-corner Y refinement, and one confirmed both-missing duplicate removal. Four training files differ from the parent; the complete overlay contains41 training label files. All37 other inherited payloads are exact.

[Report and figures](../../../docs/fixed-geometry-label-review-2026-11-14.md) · [Exact accepted operations](../../../results/review/fixed-geometry-20261114/accepted-review.json) · [Current unresolved fields](../../../results/review/fixed-geometry-20261114/review-status.json) · [Manifest and row maps](manifest.json)

The user confirms identical cassette size, no seated sliding/tilting, and equal rack widths. The revision uses this knowledge for physical correspondence; boxes still describe observed front material, and naturally hidden corners remain missing. New006 source53 receives a conservative partial right-bound correction, with its exact tight extent still held. New030 source12's proposed deletion was rejected pending material identification.

Effective training:137 images /3,773 objects /6,849 locatable slots /697 missing slots /13 empty images. Original validation and the separate development reference stay unchanged. The manifest keeps `release_ready:false` and `training_recipe_id:null`; old GPU recipes do not silently consume this new kind.

From the repository root, with original PNGs and the original model hydrated:

```bash
python -B scripts/refine_reconciled_pose_labels.py --root . verify \
  --candidate data/candidates/fixed-geometry-labels-2026-11-14
```

The verifier checks the unchanged parent, exact reviewed operations, original→parent→child row mapping, unchanged endpoint visibility, unlisted bytes, geometry, source identities and validation. The lone endpoint operation changes new012 source30 left Y to634px while retaining its exact X and `v=2`. New128 original12 is removed; later output row numbers shift, so always use the recorded original IDs for review.

The overlay contains labels only. Use original images plus these effective labels in a separate working copy for the [existing editor](../../../tools/cassette_labeler/README.md). Editor saving and mechanical verification do not establish physical acceptance of an ambiguous surface.
