# Box geometry evidence

This CPU replay describes the geometry of retained predictions and unmatched
references. It changes no label, prediction, operating threshold or official
assignment. Start with [results.md](results.md); the interpretation and bounded
research recommendation are in [the research note](../../research-next-models.md).

## Inputs and scope

The sole data input is the frozen
[`baomai-annotation-reference-r60-results.zip`](../annotation-reference/baomai-annotation-reference-r60-results.zip)
archive: **972,658 bytes**, SHA-256
`ce8df4bcf223b6bee49fa67e829a443acb572e4a6aad75fa0f92657a16cc2601`.
It contains native predictions, three reference label sets, all canonical views,
saved scoring tables and source identities. The script checks the archive and
member hashes and frozen helper-source hashes before analysis.

The supplied model and retained two-view method are measured separately on
supplied664, frozen666 and reviewed r50. The canonical predictions, suppression
and view associations are reconstructed exactly, and **120 per-image scoring
tables** are reproduced. All **1,984 reference rows per method** contribute to
group statistics; **197 missed-reference records** across the six comparisons
are retained in cases.json. All geometry is measured in original 1280 x 720
image pixels. The object confidence is strictly above 0.25, NMS/postclip IoU is
0.70, reference-matching IoU is at least 0.50 and endpoint confidence is at least
0.50, exactly as in the archived comparison.

## Diagnostic association

For an officially unmatched reference, the diagnostic selects the retained
same-class prediction with highest box IoU, with lower prediction index breaking
ties. If every overlap is zero, it selects no candidate. It does not rematch
anything: a candidate may be reused or may already belong to another reference
in the official one-to-one score. The candidate's official owner is recorded.
Overlap alone does not establish physical identity.

The descriptive near-overlap group uses maximum IoU in [0.30, 0.50). The 1.50
width/height ratio flags summarize extent discrepancies; neither cutoff changes
evaluation or proposes an output filter. Continuous values remain available.
Signed edge errors are prediction minus reference: positive x is rightward and
positive y is downward. Width/height ratios divide prediction extent by reference
extent. The reported Gaussian Wasserstein distance has pixel units and no fitted
normalization constant. Its center/extent expression is independently checked
against the equivalent half-sum of four squared edge errors.

## Reproduce

Run from the repository root with Python's standard library:

```sh
python scripts/analyze_box_geometry_r63.py --root . --output /tmp/box-geometry-replay
```

The four deterministic generated outputs are:

| File | Contents |
| --- | --- |
| [summary.json](summary.json) | Provenance, definitions, verification, groups and continuous summary statistics |
| [cases.json](cases.json) | Every missed reference, diagnostic candidate and signed geometry comparison |
| [results.md](results.md) | Readable tables, including the ten reviewed-reference near misses |
| [manifest.json](manifest.json) | Byte counts and SHA-256 hashes of the other three generated files |

No generated output includes an absolute output path or timestamp. A second
independent invocation into another directory must produce identical bytes.
Existing differing files are preserved and cause a failure. This README is
documentation, outside the four-file replay manifest.

This is development-data diagnosis. It does not measure performance on new
captures, explain the training-time cause, establish annotation correctness, or
promote a model.
