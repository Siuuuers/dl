# Sequence-refined cassette fronts: review before merge

This directory contains the complete **137-file training-label review** composed from the full 14 November review and the explicitly accepted 17 November operations. It includes the thirteen empty training labels. `manifest.json` distinguishes the complete file inventory from the files changed in this pass and preserves original-row/addition identities when displayed TXT row numbers shift.

The new full atlas renders these label files on the actual photo pages. It does not reuse the old atlas's annotation pages with corrections placed only in an appendix. The separately pinned 20-photo development reference remains outside this directory and unchanged. Original in-training validation and the development reference are different label revisions; neither is silently substituted into a training run.

## Current result

The composed labels contain **3,795 objects, 6,860 locatable corners, and 730 missing corners**. This pass changes17 photos with20 replacements,9 additions, and5 deletions. Nine case groups remain on eight training photos; they are nine distinct border rows. See the [review report](../../../docs/sequence-label-review-2026-11-17.md) for physical decisions, current row numbers, remaining questions, and the completed earlier model comparison.

## Reproduce

After reviewing the operation file, compose and verify the dated preview:

```bash
python scripts/compose_sequence_front_review.py --root .
python scripts/compose_sequence_front_review.py --root . --verify-only
```

The accepted operations are in `results/review/sequence-refinement-20261117/accepted-review.json`. Every replacement/deletion is checked against the exact full-reviewed base row, and adjacent-frame evidence is tied to original image hashes. Existing corner tokens are preserved unless an explicit target-frame corner decision accompanies the operation. A neighboring frame's visible corner is not automatically a visible corner in the target frame.

Render the full PDF with:

```bash
python scripts/render_label_atlas.py \
  --root . \
  --candidate data/candidates/sequence-refined-fronts-2026-11-17 \
  --status results/review/sequence-refinement-20261117/review-status.json \
  --output output/pdf/cassette-label-atlas-sequence-refined-2026-11-17.pdf \
  --audit results/review/sequence-refinement-20261117/atlas-audit.json
```

Do not pass the older `--adjudication` option: the newly composed status already includes those inherited closures and the current case decisions. The earlier two-file `video-continuity-proposal-2026-11-14` remains a historical preview; it is not this complete label set.

## Scope

The boxes enclose the visible portions of the cassette's target front face. They may span an intervening occluder when separated fragments belong to the same front. They do not enclose the whole cassette body or guessed hidden extents. Cyan means no recorded hold; it does not certify every coordinate.

This is a **review-only candidate awaiting PDF review before merge**. It changes no frozen source labels, historical candidate, notebook, training recipe, model weights or evaluation reference. A later training experiment needs its own explicit inputs. The completed PR64 comparison did not support replacing PR35, which remains selected.
