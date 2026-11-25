# R36 verified subset and GPU comparison

R36 follows the PR119 R35 result. Four verified box corrections are applied across three photographs; two training rows in `new_000034` form the next controlled training change. Three requests remain partly or fully pending. Their unresolved fields keep their exact R35 values, so the verified subset can be evaluated without claiming that the entire request is complete.

## Accepted scope

| Image / row | Accepted change in original pixels | Unresolved request |
|---|---|---|
| `new_000034:34` | Box left edge `426 → 435`, beside the observed left corner near `x436` | None. The audit explicitly corrects the requested point name from right to left. |
| `new_000034:35` | Box left edge `509 → 511`; physical top `258 → 255`, restoring the R34 box fields | Preserve the left point `(518.35, 297.09)` pending a marked intended corner. |
| `new_000010:24` | Box right edge `910 → 913` includes its own pale rim | Development only. Preserve both point coordinates until the intended point is identified. |
| `new_000057:1` | Box left edge approximately `777 → 782` | Development only. |
| `new_000006:53` | Entire R35 row retained | The referenced yellow/red markup is unavailable; the target right edge remains pending. |

All keypoint tokens, instance identities, classes and visibility values remain unchanged. An observed corner is a physical turn on the cassette front; it need not coincide with a box extremum. Image-coordinate y increases downward. The focused review, exact before/after rows, original-image hashes, uncertainty and pending-field reasons are recorded in `results/review/front-review-20261126-r36/row-review.json` and `docs/annotation-boundaries-r36.md`.

The full review lives in `data/reviews/fronts-2026-11-26-r36/`. The eligible training overlay lives in `data/candidates/front-review-2026-11-26-r36/`. The fixed protocol is `experiments/front-review-2026-11-26-r36.json`; its exact treatment is `results/review/front-review-20261126-r36/training-input-delta.json`.

## Controlled training question

Does this verified two-row training-boundary delta improve visible-front detection and lower-front corner localization under the unchanged native recipe?

The reference uses eligible R35 supervision; the candidate uses the eligible verified R36 subset. Both arms retain 137 training images, 3,784 objects, 6,837 locatable corners, 731 unavailable corners and 13 empty scenes. Held photograph `new_000015` and the 31 excluded old-series photographs retain identical supervision in both arms. Original trainer validation remains 20 images with 664 objects. Development amendments do not enter training or trainer validation; the earlier R35 development deletions remain outside both training arms.

Each arm starts independently from the original epoch-173 initializer. Both use seed 46, batch 4, input size 1280, FP32, AdamW and unmodified native Pose26 loss for 30 epochs: 1,050 batches, 4,110 image presentations and 78 optimizer updates. The four-batch resource trial is discarded. Final-epoch checkpoint selection is fixed. Architecture, sampling, loss and inference settings are unchanged.

The composer requires a nonempty eligible training delta and rejects class, row-count, corner-availability or keypoint changes. Fully held rows must match R35 byte-for-byte. Partially verified rows can change only independently accepted fields, with every pending coordinate preserved. The three pending requests appear in the candidate manifest, protocol, exact delta and notebook. Clarified follow-up requests require a separately reviewed snapshot and source pin.

## Run and return

Open `notebooks/baomai_front_review_r36.ipynb` in Colab, choose an L4 or A100 GPU, and select Run all. It verifies the published source and protocol, all 157 original image hashes and the epoch-173 initializer, runs the discarded resource trial and matched pair, then produces `baomai-front-review-r36-results.zip`. Compatible completed runs can be recovered from Drive; incomplete or incompatible folders are preserved.

The ZIP contains both final checkpoints, logs, receipts, protocol and exact training delta. It does not include the original photographs or initializer. Source publication precedes notebook generation:

```bash
python3 -B scripts/build_front_review_r36_notebook.py --source-revision FULL_PUBLISHED_SOURCE_SHA
python3 -B scripts/check_front_review_r36_notebook.py
```

The notebook contains no embedded repository payload and only one source-loader definition. Its integration check executes both complete setup cells in order against a real temporary Git remote. It verifies source checkout, original-input hashes, clean reuse, preservation and rejection of dirty source, and rejection of the wrong source revision. The pending-field check accepts the reviewed partial box corrections and deliberately attempts forbidden edits to each pending coordinate; every attempt must fail. These checks do not run GPU training.

## Evaluate the returned pair

Keep box confidence 0.13, endpoint confidence 0.50, input size 1280, one-to-many output, NMS IoU 0.70 and matching IoU 0.50. Frozen 666 and historical 663 references remain unchanged. Report preserved R35 and revised R36 development diagnostics separately.

PR119 preserved all 15 missed identities under R35, gained one 5-pixel corner pair, and added two duplicates. Frozen 666 gained two 5-pixel pairs but lost one box match: `new_000089:43` moved from IoU approximately 0.50625 to 0.49805 across the fixed 0.50 cutoff while still producing a confident prediction. That mixed outcome supports another narrow annotation comparison, not promotion. The detailed result is in `results/evaluation/pr119-r35-20261126/`.

Prioritize missed visible fronts, including fronts with zero or one locatable corner. Report unmatched outputs, their duplicate subset, other unmatched outputs, empty-scene outputs and both-corner successes at 5/10 original-image pixels together. Retain missed eligible objects in the success denominators. Check box association before describing a mismatch as a completely absent prediction.

PR35 remains selected. This notebook prepares R36; it does not claim an R36 training result, resolve the pending annotations, or demonstrate reliability on unseen camera sessions.
