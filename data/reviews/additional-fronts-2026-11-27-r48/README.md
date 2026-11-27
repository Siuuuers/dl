# R48 supplementary visible-front proposals

This is a separate assistant-reviewed successor to the exact R47 review manifest
`34fe23ff1744243f2a74fb3466bb7c9106b6ad31636f8180658ffe326e26b2ea`.
It is not an automatic training or model-selection update.

The [audit ledger](audit-ledger.json) records every one of the 507 current rows
across all twelve originals, with exact before/after lines, original-image
hashes, numeric coordinate operations, reasons and remaining holds. Forty-four
rows have one side-boundary edit each. Every corner coordinate and visibility
token is byte-preserved. No row is added, deleted or resurrected.

`proposals/` contains all twelve current label files. The
`complete-candidates/labels/` directory contains exactly six annotation-complete
photographs: `frame_0001`, `frame_0146`, `frame_0160`, `frame_0178`,
`frame_0371`, and `frame_0383`. Training split assignment remains unassigned.
The remaining six photographs contain twelve held rows and 31 unresolved fields;
they are excluded as whole photographs from complete-candidate export.

The only resolved hold is `frame_0001:source:043/xmin`, now nominal x779.
The left corner remains exactly `0 0 0`. All other prior holds are carried
forward. The original full-width image and magnified ownership comparison
support this boundary; they do not reveal a hidden endpoint.

[outlines.json](outlines.json) supplies eleven open diagnostic side traces.
They are approximate explanations of visible contour portions, never training
masks or reconstructed hidden polygons. [release-status.json](release-status.json)
binds the generated combined manifest, exact exports and their hashes.

See [the amendment report](../../../docs/front-review-r48.md) for all requested
decisions, coordinates, uncertainty and reproduction. The existing R47
notebook and returned PR137 scores retain their original inputs.
