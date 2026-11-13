# Border-front adjudication evidence

Date: 2026-11-13, Asia/Hong_Kong. This continuation completes three known
annotation findings after recovery of the PR58 evaluation. The selected model
remains PR35; no new inference or training was performed.

- [Readable readiness and continuation report](../../../docs/readiness-and-border-review-2026-11-13.md)
- [Complete proposed label overlay](../../../data/candidates/border-fronts-2026-11-13/README.md)
- `reviewer-a.json` and `reviewer-b.json`: separate original-first assistant
  nominations, source hashes, visibility reasoning and coordinate uncertainty.
- `reviewer-b-prelabel-observations.json`: nominations frozen before source-label
  inspection by reviewer B.
- `adjudication.json`: final choices, exact source identities, held cases and
  unchanged review coverage.
- `mechanical-verification.json`: output from the existing annotation validator.
- `independent-verification.json`: focused independent inspection of inheritance,
  exact new operations, source bytes and protected inputs.
- `remaining-review-queue.json`: metadata-only ordering of the 67 images outside
  the recorded review union; its entries have not been newly pixel-reviewed.
- `figures/`: raw source crops beside explanatory marks, enlarged by
  nearest-neighbor replication. No sharpening or generated detail is used.

## Source pixels and scope

Both full originals were hash-checked against `data/manifest.json` and the
[immutable public upstream manifest](https://github.com/an94-doll/embedding_vision/blob/351eb3cf960ccc34829e2bc72e1fa4645acc6295/data/manifest.json).
Their source paths are `data/images/train/new_000027.png` and
`data/images/train/old_000010.png`; exact SHA256 identities are in
`adjudication.json`. The original PR58 evidence record contains copies under
`review/training-pilot/originals/`; the [public derivative](../../verification/baomai-composed-results-20261113/README.md)
omits all duplicated PNGs and retains the recorded hashes and crop geometry.
Historical `original_available_in_prior_capsule` fields refer to that original
record, not to PNGs inside the public derivative. The two annotated figures
remain available in this directory.

The preserved review and verification records name local review revision
`85ce9515c172d9563051c4472a1479a150c00ee4`. That historical revision is not
published. All original labels, the source manifest, annotation policy, PR35
parent and verifier/scorer inputs are byte-identical to the public PR57 base
`036479af6152dab0a4d20edb156ffb0a2fbff243`. The candidate manifest and reviewed
label files are retained byte-for-byte, including their original hashes; the
existing verifier can validate them from this public branch without the local
review commit.

The final three visible corner marks are within both reviewers' recorded manual
neighborhoods. These neighborhoods are subjective placement uncertainty, not
statistical confidence intervals. Root inspected originals and crops, and then
reconciled the nominations; root had prior exposure to the recorded findings
and is not described as a blind reviewer. Assistant agreement is not independent
human certification.

Accepted geometry supplies two added rows and one selective box-bound repair.
Farther-right `027` fragments and the `001`/`013` gripper-boundary issues remain
unresolved. Finishing these previously reviewed images leaves coverage at
70/137, with 67 outside the recorded passes. A coverage record does not certify
every box or corner in an image.
