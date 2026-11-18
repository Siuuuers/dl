# Controlled r4 training candidate

This versioned overlay tests the accumulated r4 review against exact PR35 supervision. **94 training photographs differ between the arms.** All 137 training photos and 13 empty scenes remain. The candidate has 3,789 fronts, 6,853 locatable corners and 725 missing corner slots; PR35 has 3,771 fronts, 6,841 locatable corners and 701 missing slots.

The complete [review](../../reviews/fronts-2026-11-18-r4/manifest.json) remains separate. `new_000014` base row 11 is unresolved, so the entire photograph retains identical exact PR35 labels in both arms. The other training photos use exact r4 bytes, including the two explicitly recorded subpixel serialization repairs. The original 20-photo, 664-front training-validation labels stay unchanged.

Both arms independently initialize from epoch173 and use the established seed46/30-epoch recipe. The [protocol](../../../experiments/front-review-2026-11-18-r4.json) fixes final-epoch selection and the 666-front primary plus 663-front historical reference. No reference/threshold/seed/epoch search or automatic promotion is performed.

Run [the prepared notebook](../../../notebooks/baomai_front_review_r4.ipynb) after reviewing the [handoff guide](../../../docs/front-review-r4.md). Preparing this candidate does not train a model. `full_dataset_certified` stays false, and PR35 remains selected.

```bash
python scripts/verify_front_review_r4.py
```
