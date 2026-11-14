# Complete visible-front review candidate

**Accepted repairs composed; training is blocked by remaining physical questions.** See the [full report](../../../docs/complete-label-review-2026-11-14.md), [case table](../../../docs/remaining-physical-questions-2026-11-14.md) and [prepared notebook](../../../notebooks/baomai_observed_fronts.ipynb).

This immutable candidate preserves compatible earlier repairs and overlays 92 training label files. It contains 3,791 objects, 6,861 locatable slots, 721 missing slots and 13 empty photos across 137 training photos. Source images, source labels and the original 20-image in-training validation split remain unchanged. The review manifest deliberately retains `release_ready:false`; the separate [protocol](../../../experiments/observed-fronts-2026-11-14.json) controls readiness and currently blocks 26 training groups and 13 development-reference rows.

Manifest SHA-256: `9d138930b523fc8d124e116a58aa584a23c30a81d874551fcb5625a1978639a8`.

```bash
python scripts/verify_observed_training_candidate.py --root . \
  --candidate data/candidates/complete-visible-fronts-2026-11-14/manifest.json --inspect
```

Omit `--inspect` to enforce the production readiness check. The current expected outcome is a clear blocker before staging/GPU use. This artifact contains labels, not newly trained weights or measured model gains.
