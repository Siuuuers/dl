# R52 supplementary side amendments

Agent-assisted visual review of all 12 original camera photographs and all 507 current rows, following the latest `frame_0160` feedback. Ten rows receive 11 nominal side-boundary refinements; 497 lines remain exact. There is no independent human adjudication or new model training.

The review preserves all 1,014 corner slots, visibility, 12 held rows / 31 held fields, historical identities, current row order, and the six complete / six held whole-photo gates. Coordinates use the original 1280 × 720 image with x right and y down. The 1.829–2.506 pixel side changes carry nominal 2-pixel visual uncertainty, not a claim of measured subpixel physical accuracy.

- `audit-ledger.json`: exact original-pixel review decisions for every current row, with stable source IDs and output TXT rows.
- `manifest.json`: source hashes, every side change, full row map, export gates, and generated-file hashes.
- `geometry-before.json`, `geometry-after.json`: independent four-side diagnostic screens with separately counted strict interior and ordinary edge contacts and a ranked review queue.
- `proposals/`: all 12 current photo labels, including held photos for review.
- `complete-candidates/labels/`: only the six whole photographs with no inherited unresolved fields; training splits remain unassigned.

Read [the review report](../../../docs/front-review-r52.md) for scope, all coordinates, limitations, shadow policy, and reproduction commands. Run `python3 scripts/build_front_review_r52.py --check` from the repository root to verify every committed output against its frozen decisions and parent labels. The builder can also verify the 12 original JPEGs and regenerate before/after evidence into a separate output directory.

Geometry flags are review cues, not annotation-error counts. Other projected corners can legitimately fall inside a tilted object's axis-aligned envelope. The checker never edits a label automatically.
