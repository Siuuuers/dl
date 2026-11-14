# Reconciled training labels — review candidate

**Status: reviewed, not released. No model has been trained on this revision.**

This overlay unifies the compatible accepted historical training corrections and the newly reviewed `new_000038` three-front/clipped-front repair. Its 40 TXT files replace the matching original training files only when constructing this explicit revision; all other original labels remain byte-identical. It is not a complete images-and-labels folder for the HTML editor or an input accepted by an old training recipe.

[Full explanation and figures](../../../docs/label-refinement-2026-11-14.md) · [Manifest and original row maps](manifest.json) · [Exact operations and provenance](review.json) · [Evidence index](../../../results/review/label-refinement-20261114/README.md)

## What is included

There are **124 distinct field/object operations**: 66 box changes, 20 duplicate/spurious-instance deletions, 22 added fronts and 16 endpoint-slot changes. Of these, 119 are exact accepted historical operations, deduplicated across their inherited copies; five are the new `new_000038` repair. Deletions and edited fields use immutable original row IDs, so an earlier deletion cannot move a later correction onto the wrong cassette.

All 17-file PR63 operations survive. Sixteen of those label files remain byte-identical; `new_000102` also receives the earlier accepted deletion of original row15, deliberately shifting original row53 to output52 and original54 to output53. Exact row maps are in the manifest.

| Effective data | Images | Objects | Locatable slots | Missing slots | Empty images |
| --- | ---: | ---: | ---: | ---: | ---: |
| Training after this overlay | 137 | 3,774 | 6,849 | 699 | 13 |
| Original validation, unchanged | 20 | 664 | 1,227 | 101 | 3 |

The separate 663-object development scoring reference is unchanged and is not inserted into training. Missing corners remain canonical `0 0 0`; hidden right corners in `new_000038` are not guessed. The accepted record preserves ordinary manual placement uncertainty.

## What remains open

The [ledger](../../../results/review/label-refinement-20261114/unresolved/ledger.json) retains 88 documented open case groups across 64 training images. These are unresolved review findings, not a count of proven errors. Some group many source rows; others describe possible omitted fronts. Seven generic scope/precision caveats are separately non-blocking. See the [compact decision summary](../../../results/review/label-refinement-20261114/unresolved/notes.md).

The manifest therefore records `release_ready: false`, `training_recipe_id: null` and no full-dataset or human certification. This is a concrete reviewed-data draft. Resolving relevant physical-identity/corner/extent questions, or explicitly versioning a quarantine, must precede a complete corrected training release.

## Verify

With the original PNGs and epoch173 weight bytes hydrated, run from the repository root:

```bash
python -B scripts/verify_reconciled_pose_labels.py --root . verify \
  --candidate data/candidates/reconciled-labels-2026-11-14
```

The verifier reconstructs every annotation payload byte from the hash-bound specification, runs existing historical validators in their recorded policy contexts, checks all 157 original image/label identities, geometry, original row maps, unchanged validation and frozen reference/model identities. Candidate-root `README.md` is documentation only; every other candidate file must be declared and exact.

To reproduce in a fresh directory:

```bash
python -B scripts/verify_reconciled_pose_labels.py --root . build \
  --spec results/review/label-refinement-20261114/composition/specification.json \
  --output /tmp/reconciled-labels-review
```

The build refuses an existing output directory and never stages or runs training. A mechanical pass verifies identity, composition and geometry; the original-image reviews provide the separate physical evidence.
