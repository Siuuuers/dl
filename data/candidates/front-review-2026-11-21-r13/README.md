# Round 13 training candidate

This [manifest](manifest.json) selects eligible supervision from the complete r13 review for the matched PR35 comparison. Both arms contain 137 training photographs and 13 empty scenes. Reference/candidate fronts are 3,771/3,777, with 78 differing training label files.

The 15 unresolved training photos and the separate 31 old-series treatment exclusions retain exact PR35 labels in both arms. Original 20-photo/664-front validation and frozen 666/663 evaluation references remain fixed. The complete current review contains all 157 annotation files, including development-review amendments that are not applied to original training validation.

Use the [r13 notebook](../../../notebooks/baomai_front_review_r13.ipynb) and return `baomai-front-review-r13-results.zip`. See the [handoff](../../../docs/front-review-r13.md) and [actual runner verification](../../../results/review/front-review-20261121-r13/runner-verification.json). No r13 training or automatic model promotion occurs during preparation.
