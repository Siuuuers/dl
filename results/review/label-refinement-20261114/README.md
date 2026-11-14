# Label refinement and PR35 provenance evidence

This review answers two questions: which label bytes the selected model was trained with, and which supported corrections can be assembled before another GPU run. **PR35 used the older eight-file overlay, including the unchanged defective `new_000038` file.** The new 40-file review candidate is untrained and explicitly retains unresolved decisions.

[Main report](../../../docs/label-refinement-2026-11-14.md) · [Reconciled candidate](../../../data/candidates/reconciled-labels-2026-11-14/README.md)

## Evidence map

| Question | Evidence |
| --- | --- |
| Exact returned archive, checkpoint, receipt and staged label identities | [PR35 audit](provenance/audit.json), [archive payload hashes](provenance/archive-files.json), [per-file label and image audit](provenance/label-image-hash-inventory.json) |
| Which accepted changes stayed in separate experiments? | [Historical composition explanation](composition/historical-inventory.md), [historical inventory](composition/summary.json), [all exact accepted operations and their source records](../../../data/candidates/reconciled-labels-2026-11-14/review.json) |
| What inputs define the final revision? | [Hash-bound source inventory](composition/source-inventory.json), [repository-relative specification](composition/specification.json), [comparison with PR35 and the prior candidate](composition/comparison-to-selected-and-current.json) |
| New three-front and clipped-front correction | [Accepted adjudication](visual/new_000038-accepted-adjudication.json), [three-front figure](visual/new_000038-accepted-three-fronts.png), [clipped-front figure](visual/new_000038-accepted-bottom-front.png), [render metadata](visual/new_000038-accepted-render-index.json) |
| Which missing corners are supported? | [Endpoint-status review](visual/held-endpoint-review-status.json) |
| Concrete physical-identity question for the owner | [new006 question](visual/new006-question.json), [context and original-pixel crop](visual/new006-front-ownership-question.png) |
| Fresh check of previously held thin fragments | [new128 review](visual/new_000128-fragment-review.json), [top-band comparison](visual/new_000128-row1-fragment-review.png), [right-edge comparison](visual/new_000128-row12-fragment-review.png) |
| Remaining training decisions and separate validation notes | [Complete ledger](unresolved/ledger.json), [compact explanation](unresolved/notes.md) |
| Independent payload reconstruction and corruption probes | [Independent verification](composition/independent-verification.json), [durable-path integration](verification/portable-integration.json), [final repository verifier receipt](verification/final-candidate.json) |

The independent review binds a scratch candidate with temporary descriptor paths. The final repository candidate uses durable relative paths. Its 40 label hashes are byte-identical to the independently checked payload; `portable-integration.json` records both manifest identities and every matching label hash. The path change is not an annotation revision.

## Checked results

- 124 operations across 40 training files: 66 boxes, 20 deletions, 22 additions and 16 endpoint slots.
- Final training: 137 images, 3,774 objects, 6,849 locatable slots, 699 missing slots and 13 empty images.
- Original validation: 20 images, 664 objects, 1,227 locatable slots, 101 missing slots and three empty images, unchanged.
- All 157 original PNG/label identities, historical strict validators, current/historical policy snapshots, frozen reference generations, base model and relevant existing code identities verified.
- Independent review traced 240 historical origin records, checked 75 modified original rows, 1,358 unchanged rows within changed files, 20 deleted rows and 22 appended rows. It rejected a changed retained token and an extra validation-label file in disposable corrupt copies.
- New038 rows2–15 are unchanged. The maximum checked decimal serialization discrepancy was 0.0000064 original-image pixels; this is numerical encoding error, not physical annotation accuracy.
- 88 open documentary case groups across 64 training images remain explicit. They are not 88 proven errors. Ordinary placement ranges and seven generic scope/precision caveats are not automatically converted into new human-decision blockers.

No GPU execution, inference, original-label rewrite, validation migration, model promotion or hidden-corner ground-truth estimation occurred. All reviewer decisions remain assistant review with explicit uncertainty; agreement does not establish independent human certification.

## Reproduce

The candidate README contains the single verification/build command. To regenerate the uncertainty ledger into a fresh directory:

```bash
python -B scripts/build_training_review_ledger.py --root . \
  --output /tmp/training-review-ledger
```

The ledger records exact source hashes, evidence selectors, original row IDs and one index entry for every training image. Its suggested crops help locate review questions; they are not accepted box annotations.

To render the new038 figures from original pixels:

```bash
python scripts/render_new_000038_accepted_review.py --repo-root . \
  --spec results/review/label-refinement-20261114/visual/new_000038-accepted-adjudication.json \
  --out-dir /tmp/new038-review-figures
```

Use the same Pillow/font versions from the render metadata to reproduce figure bytes. These annotated figures are review evidence and must not replace original PNGs in training. The main report gives the separate PR35 archive audit command.
