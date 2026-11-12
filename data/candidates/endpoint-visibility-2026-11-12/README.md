# Endpoint visibility candidate

**Status: proposed supervision candidate, with no new trained model or accuracy result.** Keep PR35 selected for reviewed proposals at input 1280, box confidence 0.13, NMS IoU 0.70 and endpoint cutoff 0.50.

This candidate copies the selected [PR35 annotation parent](../annotations-2026-11-10/README.md) exactly, then amends nine endpoint slots on six original training rows. It is an endpoint-only intervention relative to PR35. No additional boxes, object identities, additions or deletions change.

The seventeen box corrections and duplicate deletion preserved in [PR49's recovery record](../../../results/review/baomai-review-recovery-20261112/correction-record.json) remain a separate proposal. They are not included here. In particular, `new_000023` retains its original 36-row object list, including the separately recorded duplicate concern; this keeps the endpoint intervention isolated rather than combining unmeasured changes.

## Exact composition

The cumulative overlay contains ten training-label files: eight files copied byte-for-byte from `annotations-2026-11-10`, plus two files derived directly from the frozen originals. The eight inherited change records and twelve inherited review decisions are also copied exactly.

All row numbers below are one-based **original source rows** and remain the same in this candidate. The earlier recovery package's shifted 023 row numbers belong to its separate duplicate-deletion overlay.

| Image | Original and candidate row | Left slot | Right slot |
| --- | ---: | --- | --- |
| `new_000023` | 33 | Correct visible position to `(482, 550)` | Retain missing |
| `new_000023` | 34 | Correct visible position to `(486, 580)` | Retain missing |
| `new_000023` | 35 | Correct visible position to `(490, 607)` | Retain missing |
| `new_000031` | 22 | Change to missing | Change to visible `(742, 458)` |
| `new_000031` | 23 | Change to missing | Change to visible `(737, 491)` |
| `new_000031` | 24 | Change to missing | Change to visible `(732, 518)` |

Positions use the original 1280 × 720 image coordinates. Each missing slot is independently encoded as numeric `0 0 0`; visible slots use `v=2`. The [review](review.json) contains all twelve before/after slot triples, the six exact source/result row texts, reasons, source-image hashes and the manual coordinate ranges. The [manifest](manifest.json) provides full identity row maps for both new files.

The new delta contains three visible-coordinate corrections, three visible-to-missing changes and three missing-to-visible changes. Three right-hand slots in 023 stay missing. Every class/box token, other endpoint, other row and line ending in these two files is preserved.

| Measure | PR35 parent | This candidate |
| --- | ---: | ---: |
| Changed training-label files against originals | 8 | 10 |
| Training images | 137 | 137 |
| Training objects | 3,771 | 3,771 |
| Visible endpoints | 6,841 | 6,841 |
| Missing endpoint slots | 701 | 701 |
| Original validation images / objects | 20 / 664 | 20 / 664 |

The separately corrected 663-object development scoring reference is unchanged too. Equal endpoint totals do not establish correct supervision: assigning visibility to the wrong side can leave those totals unchanged.

## Evidence and uncertainty

The source revision is published PR49 commit `ab998df15cf7b8e6d92a3ba95bea6a70230f4145`. The candidate is bound to the exact PR35 parent manifest and PR49 correction record by SHA256. The [provenance summary](provenance.json) records the copied parent files, original source bindings and deterministic generation steps.

The endpoint judgments were made from original pixels in the earlier review. An independent reviewer recorded initial observations before labels and other reviewers' conclusions. The continuation reviewer and root already knew the recovered findings; their final checks were informed reviews. Building this candidate does not constitute another visual adjudication, independent human review or supervisor sign-off.

Integer points are manual placements on rounded, soft visible bends. The per-slot ranges are descriptive annotation uncertainty, not statistical confidence intervals or physical-calibration tolerances. A missing slot means the actual corner is not reliably locatable; no hidden coordinate is inferred from boxes, opposite corners, stack spacing or predictions. Other rows and boxes are not certified by this pass.

## Verification and use

From the repository root, the dedicated endpoint verifier checks this explicit amendment format:

```bash
python -B scripts/verify_pose_endpoint_candidate.py --candidate data/candidates/endpoint-visibility-2026-11-12/manifest.json
```

The `training_endpoint_overlay` kind separates these audited point changes from the older annotation contract, which protects inherited visible-point tokens. The existing parent verifier remains unchanged. This candidate adds no training recipe and no request for a GPU run. Any later comparison needs a fixed endpoint-specific protocol and an explicit training admission decision.

Mechanical checks can establish source identity, preservation, slot transitions, geometry and exact composition. They cannot show that training on this candidate improves detection, corner accuracy or unseen-scene reliability. The current model selection therefore remains PR35.
