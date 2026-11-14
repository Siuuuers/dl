# Fixed cassette geometry: observed-label review evidence

[Main report](../../../docs/fixed-geometry-label-review-2026-11-14.md) · [Candidate](../../../data/candidates/fixed-geometry-labels-2026-11-14/README.md)

- [physical-constraints.json](physical-constraints.json): the user's equal-size, fixed-seating and equal-rack-width statement, with its scope.
- [accepted-review.json](accepted-review.json): exact parent identity, all71 accepted operations, original/parent row mappings, source hashes, before/after tokens, manual ranges, root adjustments and withheld deletion.
- [review-status.json](review-status.json): field-specific addendum to the immutable88-case ledger. It accounts for all163 explicit early fields, closes the new006 right-jaw case and retains87 documented open case groups.
- `reviewers/`: original-first reviewer proposals and independent physical rechecks. These preserve proposed-versus-accepted distinctions and dissent. Figure filenames inside these raw reviewer records describe the scratch inspection log; the portable accepted figures are below and the original PNG hashes bind their source evidence.
- [figures/index.json](figures/index.json): reproducible source-pixel before/after figures, with hashes, original crop coordinates and source row IDs.
- [Gripper-opening material question](figures/new030-aperture-question.png): independent comparison of original crops from new030 and existing empty-scene frames; source hashes and crop details are in [the aperture review](reviewers/new030-aperture.json). No coordinates are transferred across frames.

## Verification

The [revision checker](../../../scripts/refine_reconciled_pose_labels.py) runs the frozen parent checker unchanged, then reconstructs this exact child. It permits only accepted box changes, coordinate refinements of already visible endpoints, and deletions of canonical both-missing rows. It rejects hidden-point restoration, validation changes, undeclared token changes, multiline replacement rows, extra payload files and outputs nested inside existing candidates.

The [figure renderer](../../../scripts/render_observed_label_revision.py) uses original PNGs and exact accepted rows. To reproduce its seven generated figures, choose a fresh directory:

```bash
python -B scripts/render_observed_label_revision.py --root . \
  --output /tmp/fixed-geometry-review-figures
```

The separate aperture comparison is source evidence from the independent reviewer, not generated model output. All figures use original pixels; none use image generation or enhanced pixels.

[Independent reconstruction](independent-verification.json) verifies every one of the41 TXT payloads and all row maps without importing the builder or its parsing helpers. It preserves all22 inherited additions,20 inherited deletions and4,367 unchanged parent rows, including validation. The only visible-point token change is the declared new012 source30 Y refinement. Six targeted corruptions are rejected: multiline text, an undeclared endpoint coordinate, hidden-to-visible restoration, a wrong parent row, corruption of an inherited file, and an output nested inside the frozen parent. The receipt separately records a later metadata-only switch from the equivalent local source commit to its published GitHub commit; no labels or code changed.

[Artifact checks](artifact-checks.json) record reproduced figures, hashes, local-link checks and unchanged parent/policy identities. [Final verification](verification.json) records the candidate identity and complete mechanical result. No GPU, inference, model selection change or quality gain is claimed.
