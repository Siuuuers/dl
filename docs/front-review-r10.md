# Round 10: complete new-series annotation review

Round 10 applies the user's marked examples to every **126 `new_*` original photographs**. The **31 `old_*` review files remain byte-identical to round 9**, as requested. All 157 original PNGs and original label files remain intact.

The saved baseline is round 9 at `848242c6aedc0a521daf0471e1eda514431f5684`. This is a new annotation package and controlled-comparison notebook. **PR35 remains the selected model; no round 10 GPU experiment has been run.** The completed [PR76 evaluation](../results/evaluation/pr76-recovery-20261119/README.md) explains the retained model decision.

## What the annotations mean

The [boundary policy](annotation-boundaries-r10.md) defines two related but independent targets:

- The box encloses the visible cassette front and its own frontal rim, using the supported leftmost, highest, rightmost and lowest points. Rack rails, the rear perforated plane, hands, grippers and neighboring cassettes do not define its limits.
- The landmarks locate the actual lower front-face corners. A visible side face or rim may extend beyond a corner. The corner does not automatically move when a box edge changes.

A rectangular box can contain rack or an occluder between supported portions of the target. That alone does not make it oversized: an axis-aligned rectangle cannot trace a sloping or partially occluded contour exactly. The relevant check is which material establishes each of its four extrema.

An identifiable front can have zero, one or two locatable corners. Each unavailable corner remains exactly `0 0 0`; a missing point by itself is not an unresolved instance. Likewise, repeated stack spacing is not sufficient evidence to invent another hidden front.

The user's later-photo preference was interpreted as **original baomai cyan**, meaning `data/labels`, rather than every inherited cyan field or the PR35 control overlay. Restorations preserve exact original tokens only after matching the same physical front across row versions. Supported additions, duplicate removals and corrected box ownership are retained when the original labels have a specific visible defect.

## Traceable changes

| Round 10 change, relative to round 9 | Count |
| --- | ---: |
| New photographs inspected | 126 |
| Photographs with annotation changes / unchanged | 65 / 61 |
| Existing rows revised | 269 |
| Existing box / left / right field groups revised | 134 / 124 / 47 |
| Instances added / removed | 2 / 2 |
| Remaining cases / affected photographs | 18 / 16 |

A row can have more than one revised field. The two added fronts are in `new_000017` and `new_000043`, each with both corners unavailable. The complete review has 3,790 training fronts and 661 development fronts; those are review counts, separate from the actual controlled-training inputs below.

The [amendment summary](../results/review/front-review-20261119-r10/amendment-summary.json) records the final photo, operation and field counts. The [coverage record](../results/review/front-review-20261119-r10/request-coverage.json) binds one review entry to every new photograph. The [operations](../results/review/front-review-20261119-r10/operations.json) retain exact round 9 before-lines and the resulting labels, with source hashes and physical reasons.

In `new_000007`, round 9 rows **7 and 14 are removed** at the user's explicit direction. Round 9 rows **46–49 have tighter rack-free bounds** and become current rows **44–47** after those deletions. The [row map](../results/review/front-review-20261119-r10/row-maps.json) records every surviving identity and addition. No positional row number is used as a substitute for object matching.

The red point examples in `new_000008` support moving specific landmarks to the front-face turns while keeping the cassette's own rim in its box. A second independent inspection supports those changes and the selected `new_000004` and `new_000013` corrections. One uploaded screenshot, `04-427838.jpg`, matches `new_000013`, which was absent from the abbreviated list of photo names; the [reference index](../results/review/front-review-20261119-r10/user-reference-index.json) records the match.

Specific ownership and separator questions remain in the [case ledger](../results/review/front-review-20261119-r10/review-status.json). For example, the lower pale continuation near `new_000025` row 57 does not show the independent separators needed to justify both additional blue-box instances. The separate [pixel inspection](../results/review/front-review-20261119-r10/new025-secondary-evidence.json) records the visible transitions and the limits of that conclusion. It does not assert that hidden cassettes are absent.

Fifteen training photographs retain at least one case: `new_000002`, `014`, `015`, `020`, `025`, `029`, `030`, `037`, `038`, `040`, `041`, `044`, `045`, `049` and `074`. The remaining development-photo question is `new_000089`. In particular, the thumb-side subdivisions in `037` and `038` lack independently traceable separators; they remain proposals for review rather than assumed instance counts.

## Reading the complete PDF

The PDF contains every new photograph, including unchanged photographs and all empty scenes. Its first page is a clickable photo index. Each photograph has all current labels, a changed-field list, missing-point rows and any specific open question.

| Mark | Meaning |
| --- | --- |
| Cyan | Field inherited unchanged from round 9 |
| Magenta | Field changed in round 10 |
| Amber | Field with an unresolved physical question |
| Circle / square | Left / right lower front corner |
| `44[46]` | Current row 44 corresponds to round 9 row 46 |
| `*` / `+` | Edited current row / added instance |
| `D7` and a dashed outline | Deleted round 9 row 7, with no current training row |

When a field is both changed and held, the page retains its amber indication. Correctly missing corners have no invented image marker; their rows are listed below the photograph. Color records the review state, not a guarantee of physical truth.

Original 1280 x 720 RGB pixels are embedded losslessly. The saved-PDF check compares every embedded image with its source PNG, verifies page IDs and navigation, and rejects overlapping or clipped tags and truncated footer blocks. Tag placement never changes annotation coordinates. The final file and verification details are recorded in the [PDF delivery record](../results/review/front-review-20261119-r10/pdf-delivery.json).

## The new comparison notebook

Use [baomai_front_review_r10.ipynb](../notebooks/baomai_front_review_r10.ipynb). It pins source commit `bae1ed46bf3337956dbd60ac85aa5bcc759bbbcd` and protocol SHA-256 `0b73d423fcaff8f7b7d72da0e6bc33acb5814f2d8435fffa0e163b3efd1bd48f`; the [handoff record](../results/review/front-review-20261119-r10/notebook-handoff.json) also records the exact notebook hash. Select a CUDA GPU in Colab, run the cells in order, and return the completed result ZIP. The notebook verifies inputs before installing the trainer or starting the resource trial. Completed arms and result files can be verified and recovered from Drive under the same source and protocol identity.

There are three distinct supervision groups:

| Photograph group | Reference arm | Candidate arm |
| --- | --- | --- |
| Eligible new training photos | PR35 labels | Accepted round 10 review labels |
| New training photos with any unresolved case | Exact PR35 labels | The identical PR35 labels |
| 31 confirmed old training photos, excluded from this treatment | Exact PR35 labels | The identical PR35 labels |

All 137 training photos, including 13 empty scenes, remain in both arms. Original validation stays at 20 photos, including three empty scenes, with its original 664-front labels. Corrections to the separately versioned development review do not silently replace training validation or the frozen scoring references.

The materialized reference arm contains **3,771 training fronts** and the candidate **3,778**. Exactly **77 training label files differ between the arms**. That count compares the complete accepted package with PR35, including inherited accepted revisions; the 65 changed photos above compare round 10 with round 9. They measure different changes.

The fixed comparison uses independent epoch-173 initialization for each arm, seed 46, 30 epochs, batch 4, image size 1280, FP32 and AdamW at learning rate 0.0001. It selects the final epoch and performs no seed, threshold or epoch search. The four-batch resource trial checks execution and never initializes either full run.

Evaluate both fresh arms and retained PR35 against the fixed **666-front primary** and **663-front diagnostic** references, reporting them separately. The current development review is available as separately identified exploratory evidence. Preparing these files does not establish improved model performance or unseen-session reliability.

## Reproduction and verification

The [actual-input verification](../results/review/front-review-20261119-r10/verification.json) checks materialized images and labels. The [runner verification](../results/review/front-review-20261119-r10/runner-verification.json) exercises the same candidate dispatcher and staging code used for training.

All seven notebook code cells compile and contain no saved execution outputs. The exact input-preflight code from the notebook was also executed successfully against the real images, labels and downloaded 106,838,542-byte epoch-173 checkpoint. Its [receipt](../results/review/front-review-20261119-r10/notebook-input-preflight.json) confirms the input identities; no CUDA trial or full training arm was run.

```bash
python -B scripts/verify_front_review_r10.py
python -B scripts/check_front_review_r10.py --output /tmp/front-review-r10-check
python -B scripts/build_front_review_r10_pdf.py
```

Choose a fresh output directory outside the checkout for the staging check. To reconstruct the review in a checkout without the generated r10 trees, use `scripts/build_front_review_r10.py --build` with the explicit `root-audit.json`, `early-audit.json`, `middle-audit.json` and `late-audit.json` audit paths. Existing generated trees are protected from accidental overwrite. The checks prove exact replay, scope and file identity; the recorded physical uncertainty remains visible.
