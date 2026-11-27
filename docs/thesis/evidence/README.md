# Thesis evidence and reproduction

## R53 and cached fusion

[cached-fusion-results.md](cached-fusion-results.md) and [cached-fusion-results.json](cached-fusion-results.json) preserve the fixed two-view replay, exact PR149/150 archive/member hashes, baseline checks, all four references and gained/lost target identities. The [portable script](../../../scripts/analyze_r53_cached_fusion.py) accepts the two published return ZIPs and the repository source. Full fused arrays and per-image details can be regenerated with its documented flags. No network inference is required for the replay.

The original R53 return archives remain in [PR149](https://github.com/Siuuuers/dl/pull/149) and [PR150](https://github.com/Siuuuers/dl/pull/150). They are not duplicated in this folder. The [result interpretation](../r53-results.md) keeps the treatment conclusion separate from reproduction success.

## R54 and R55

[r55-automatic-output-audit.zip](r55-automatic-output-audit.zip) is the unchanged previous analysis capsule, SHA-256 `39b1efeb7364cf872bb2e33654fcb8f929390a045669f9c17f630287366f9c9b`. It includes the exact R54 verified reference/prediction snapshot and original scorer, the confidence-frontier analysis, the R55 analyzer, complete per-image R55 results, and historical notes. Contents are research records, not new model inference.

Extract into a separate directory, enter its `r55-analysis` folder, and run:

```bash
python -B analyze_automatic_pairs.py \
  --inputs ../r54-analysis/inputs.json \
  --scorer ../r54-analysis/pose_scoring.py --output replay
```

The script verifies the exact input/scorer hashes and reproduces the eight original model/reference baseline tables before analyzing offered pairs. [automatic-pair-results.md](automatic-pair-results.md) provides the readable results; [automatic-pair-summary.json](automatic-pair-summary.json) is a compact projection of all operating points. The capsule retains the full result whose SHA-256 is `52d51e955c44b4cad24368424cfd43b3302455381311959c13dbbab2d6ff45b8`.

R55 applies finite-coordinate, original-bounds and both-endpoint-score checks to retained predictions, then recomputes the original matcher against all reference fronts. A supported pair additionally requires two locatable reference corners and both errors within tolerance. Unmatched accepted pairs and incomplete-reference pairs remain in the offered-output denominator; their lack of support is not proof of physical error.

## Figures and new notebook

The [figure data](../figures/figure-data-r56.json) retain all plotted numerators, denominators, case identities and provenance. The [figure guide](../figures/captions-r56.md) explains reproduction.

[flip-bootstrap-r56.json](../flip-bootstrap-r56.json) records actual execution of the R56 notebook's first two cells against a fresh published source checkout. It is implementation/preflight evidence, not model recognition evidence.
