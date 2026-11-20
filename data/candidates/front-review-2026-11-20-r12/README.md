# Round 12 controlled training candidate

Use [baomai_front_review_r12.ipynb](../../../notebooks/baomai_front_review_r12.ipynb). Its [protocol](../../../experiments/front-review-2026-11-20-r12.json) and [candidate manifest](manifest.json) select this exact version.

Both arms retain 137 training photos and 13 empty scenes. The PR35 reference has 3,771 fronts; the eligible candidate has 3,777. Exactly 78 training photographs differ between arms. Fifteen unresolved training photographs and the separate 31 accepted old-photo exclusions use exact PR35 labels in both arms. Original 20-photo/664-front validation and frozen666/663 scoring references remain unchanged.

This directory is an overlay, not a standalone complete editable dataset. The [complete 157-file annotation review](../../reviews/fronts-2026-11-20-r12/README.md) includes all current review decisions, including held cases. Training staging combines this overlay with the immutable source and verifies the result. The [runner receipt](../../../results/review/front-review-20261120-r12/runner-verification.json) records the actual staged counts and file checks.

No new training or model promotion occurred during preparation. See the [handoff](../../../docs/front-review-r12.md) for the fixed recipe and remaining questions.
