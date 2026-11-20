# Current round 12 annotations

This directory contains the complete saved review: **157 YOLO pose label files**, including empty scenes, for the original images under `data/images/`.

The [manifest](manifest.json) binds each file to its image and label hashes. Compared with r11, **45 photographs** change: **597 existing rows** are amended and **one visible fragment** is appended as photo 023 row 37. All unrequested files and all 31 accepted old-series labels retain their r11 bytes.

The complete review has **3,789 training fronts and 662 development fronts**. Use the [eligible training candidate](../../candidates/front-review-2026-11-20-r12/README.md) for the controlled experiment; unresolved photographs and accepted old-photo exclusions use identical PR35 labels in its two arms. Review and training counts therefore differ deliberately.

See the [round 12 handoff](../../../docs/front-review-r12.md), [annotation policy](../../../docs/annotation-boundaries-r12.md) and [case ledger](../../../results/review/front-review-20261120-r12/review-status.json). The changed-photo PDF is rendered from these exact labels. The immutable original source remains in `data/labels/`.
