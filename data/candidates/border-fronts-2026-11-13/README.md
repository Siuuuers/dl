# Reviewed border-front annotation candidate

This candidate completes three findings from the six-image PR58 follow-up pilot:
two omitted top-clipped fronts in `new_000027`, and one top-bound correction in
`old_000010`. It extends the selected PR35 supervision parent with those accepted
changes. **PR35 remains the selected model. This candidate has not been trained.**

## Accepted changes

Coordinates below use the original 1280 × 720 image. Box order is
`xmin, ymin, xmax, ymax`. Row numbers refer to original source rows unless stated
otherwise.

| Image / operation | Box | Left lower corner | Right lower corner |
| --- | --- | --- | --- |
| `new_000027`, append row 11 | `[0,0,86,20]` | `0 0 0`, outside frame | `(82,16,v2)` |
| `new_000027`, append row 12 | `[86,0,317,18]` | `(89,15,v2)` | `(313,13,v2)` |
| `old_000010`, replace row 1 | Only `y_min: 16.3056024 → 0` | Original tokens preserved | Original tokens preserved |

Two assistant reviewers independently viewed full originals and source crops
before nominating coordinates. Their point nominations differ by approximately
1–2 pixels. Root reconciliation selects one reviewer's image-supported marks,
all within both reviewers' recorded plausible neighborhoods. The rounded turns
remain soft; roughly 2–3 px of manual placement uncertainty is recorded. These
neighborhoods are not calibrated confidence intervals or subpixel ground truth.

Only the first new front's left corner is outside the image. The three in-frame
lower corners remain visible despite top clipping. No missing point is filled
from rack spacing, symmetry, another frame, or model predictions.

The `old_000010` change fixes a demonstrated omission of visible material. It
does not newly certify the inherited other bounds or endpoint positions.

## Exact composition

The complete ten-file overlay preserves the eight label files, eight change
records and twelve decisions from `annotations-2026-11-10` exactly. It adds two
complete label files. Its manifest binds the parent, original sources, current
annotation policy, review decisions and exact operations. Existing source row
identity is preserved; `new_000027` additions append after its ten original rows.

| Effective dataset | Images | Objects | Locatable endpoints | Missing slots | Empty images |
| --- | ---: | ---: | ---: | ---: | ---: |
| PR35 training parent | 137 | 3,771 | 6,841 | 701 | 13 |
| This proposed training revision | 137 | 3,773 | 6,844 | 702 | 13 |
| Original training-validation split | 20 | 664 | 1,227 | 101 | 3 |

The separately adjudicated 663-object development reference is unchanged. Other
historical candidates, including the endpoint and composed-box experiments, are
not silently accumulated here. This is a bounded extension of the named parent,
not the final reconciliation of every accepted annotation decision.

## Verify with the existing contract

From the repository root:

```bash
python scripts/verify_pose_annotation_candidate.py \
  --candidate data/candidates/border-fronts-2026-11-13/manifest.json
```

The unmodified verifier checks all 157 original label identities, exact declared
operations, row maps, geometry, canonical missing slots, unchanged existing
visible endpoints and effective counts. It does not inspect physical image
content. The [review evidence](../../../results/review/border-fronts-20261113/README.md)
separately records original-image identity checks, both nominations, final
adjudication, figures and focused independent verification.

The manifest's `source_revision` is the historical local review revision,
`85ce9515c172d9563051c4472a1479a150c00ee4`, which is not published. The original
source labels, manifest, annotation policy, PR35 parent and verifier inputs are
byte-identical at public PR57 base `036479af6152dab0a4d20edb156ffb0a2fbff243`.
The public branch preserves this candidate and its recorded verification hashes
without rewriting the original review provenance. The verification command
above does not require the historical local commit.

Continue whole-image review and reconcile a coherent supervision revision
before defining another matched training experiment. No new training recipe,
GPU run, reference relabeling, inference-default change, or model-improvement
claim is introduced by this candidate.
