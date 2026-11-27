# R48 amendment verification

The [review guide](../../../docs/front-review-r48.md) explains all decisions.
This directory stores text evidence; the PDF is delivered separately.

| Evidence | Purpose |
| --- | --- |
| [verification.json](verification.json) | Exact parent/operation replay,88-file package membership, original and cue identities, export counts and PDF hashes |
| [visual-coverage.json](visual-coverage.json) | All 12 supplementary originals,507 current rows,47 generated inspection sheets and ten requested development targets |
| [pdf-index.json](pdf-index.json) | All 53 pages,15 development requests,507 supplementary identities and 56 detail cards |
| [pdf-qa.json](pdf-qa.json) | Full raster review, measured text-bound checks and corrected layout verification |
| [builder-guards.json](builder-guards.json) | Positive replay and rejection of the 11 concrete invalid-package cases tested during implementation |

The amendment PDF is `r48-annotation-amendments.pdf`, 53 pages,
34,570,293 bytes, SHA-256
`09171728e4c08646c13721f0c966a09fbe29f07854a71c1da53b4304eefde83d`.

The review contains 44 supplementary side amendments and nine development-row
amendments. It preserves every unavailable corner and all instance membership.
Twelve supplementary rows retain 31 held fields; two development fields remain
explicitly held. Annotation completeness and source replay do not constitute
independent human sign-off or a model-performance claim.

The separate [fixed-cache label-sensitivity replay](../../evaluation/front-errors-r48-20261127/reference-sensitivity.json)
leaves all target assignments, misses and total unmatched outputs unchanged.
The [PR137 error atlas evidence](../../evaluation/front-errors-r48-20261127/README.md)
continues to use the original R38 reference.
