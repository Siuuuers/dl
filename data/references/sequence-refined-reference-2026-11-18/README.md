# Sequence-refined development reference: 18 November 2026

This separately versioned reference contains **20 photos, 666 front instances,
1,227 locatable endpoints, 105 missing endpoints, and three empty photos**.
It follows the 14 November completion reference with eight row replacements and
three additions across four photos: new009, new010, new086, and new088.

New009 row 56 is split into a tightened upper front and new row 58. New010
rows 10-14 are corrected, and new rows 33/34 record its two visible lower stem
fronts, ordered from upper to lower. New086/new088 row 57 is retained and tightened
to the small visible front rather than deleted for size. All existing point
tokens are preserved; the three additions have both true corners missing.

Five held-row questions are resolved. Eight held rows remain on seven photos;
`remaining_current_rows` is the authoritative list for this version. Cyan means
there is no recorded hold for that row, not that every label is certified.

Original validation labels and past evaluation references are unchanged. This
reference is not silently substituted into training validation. The same saved
models have been [rescored separately on both references](../../../results/review/sequence-refinement-20261118/model-replay/README.md).
Changing scores between references are annotation effects, not model learning.

## Reconstruct or verify

```bash
python scripts/compose_sequence_reference.py \
  --review results/review/sequence-refinement-20261118/development-reference-review.json \
  --verify-only
```

Omit `--verify-only` to reconstruct the dated output from the bound parent and
review ledger. All 20 TXT files are included. Stable row maps preserve original
and added identities; existing row numbers are unchanged in this pass.

Manifest SHA256:
`0de64be3797dce2e43d4cb98d09620eab969bb95725ed9e75682dcfcd53c7f96`.
