# Round 5 controlled training candidate

The [manifest](manifest.json) composes the [r5 review](../../reviews/fronts-2026-11-19-r5/manifest.json) with exact PR35 supervision on the whole held `new_000023` photograph. There are 94 training photographs with different label bytes between the reference and candidate arms.

Use the [pinned notebook](../../../notebooks/baomai_front_review_r5.ipynb) and [protocol](../../../experiments/front-review-2026-11-19-r5.json). Both arms start independently from original epoch173 with the established seed46, 30-epoch recipe. Original PNGs, 13 training negatives, the 664-front training validation and fixed 666/663 scoring references are retained.

This is a controlled supervision comparison with `full_dataset_certified: false`. PR70's r4 results do not support model promotion; PR35 remains selected. See the [handoff](../../../docs/front-review-r5.md) for the unresolved physical-feature question and exact interpretation limits.
