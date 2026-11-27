# R47 expanded-data, equal-epoch pilot

27 November 2026, Asia/Hong_Kong.

**Prepared experiment; no R47 GPU result has been produced by this review.**
Open [the four-code-cell notebook](../notebooks/baomai_additional_front_pilot_r47.ipynb)
in a GPU runtime and run all. The first two cells load the pinned source and
perform CPU preparation. The third runs a discarded resource trial and both
full training arms. The fourth evaluates and downloads one result ZIP.

## The question and the exact treatments

Does expanding the previously used supplementary training data from one
complete photo to all five R47-complete photos improve the fixed development
criteria under the same 30-epoch recipe?

| Training input or budget | Reference | Candidate |
| --- | ---: | ---: |
| Original effective-R37 photos | 137 | 137 |
| Supplementary photos | 1 | 5 |
| Total photos | 138 | 142 |
| Front instances | 3,827 | 3,999 |
| Locatable corner slots | 6,923 | 7,259 |
| Unavailable corner slots | 731 | 739 |
| Genuine empty photos | 13 | 13 |
| Epochs | 30 | 30 |
| Batches per epoch, batch size 4 | 35 | 36 |
| Total batches | 1,050 | 1,080 |
| Expected measured optimizer updates | 78 | 80 |
| Image presentations | 4,140 | 4,260 |

The reference supplementary photo is **`frame_0383` with the exact PR133
training labels**. The candidate supplementary set is **`frame_0146`,
`frame_0160`, `frame_0178`, `frame_0371`, `frame_0383`**, with the frozen R47
complete-candidate labels. Shared `frame_0383` labels are unchanged. The four
new photos add 172 fronts, 336 locatable corner slots and eight unavailable
slots. `frame_0178` contributes the partial-front supervision.

This reference is the **previous pilot's candidate dataset**. It is not the
137-photo arm called "reference" in the PR133 error PDF. Both new R47 arms
start afresh from the original epoch173 checkpoint; neither continues from a
PR133 final checkpoint.

The candidate receives 120 additional image presentations, 30 more batches
and two more expected optimizer updates. Additional exposure is explicitly
part of this treatment. Equal epochs do not mean equal compute, so a result
must not be described as isolating data quality from optimization effort.
This keeps the native loader and recipe simple while using every eligible
photo. The [earlier training plan](front-review-r44-training-plan.md) permits
this design when exposure is reported explicitly.

## Fixed conditions and guards

Both arms use the existing `reviewed-incremental-fronts-seed46-v1` recipe:
1280 input, batch 4, AdamW, FP32, seed 46, 30 epochs and final `last.pt`.
The original 137 effective R37 training labels and all 20 trainer-validation
photos remain unchanged. R38, frozen666 and historical663 references are
scored separately on the same development photos. Their identity and label
hashes are checked before training.

The supplementary release is bound by its committed SHA-256. Staging replays
the R47 annotation ledger from the exact R46 predecessor, checks every
original and label, and accepts only whole photos whose field holds have been
resolved. Held proposal files are never copied into either new candidate
training set. Images duplicated by content across the original and extra
sets are rejected.

The first resource trial must complete four actual batches with finite loss,
finite gradients and a nonzero parameter update. Its weights are discarded.
Each full arm records every image exposure and actual optimizer-update batch
index. Expected counts come from the pinned native warmup/accumulation
schedule, and the measured run must agree. A final receipt must cover all
thirty epochs and its saved artifacts.

The actual GPU and package versions are recorded and must agree across both
arms. A T4 completed the previous R45 recipe; this is evidence about that
previous run, not a guarantee about a new session. The current R47 GPU trial,
training and inference remain to be executed in the notebook.

## Run and return

1. Open the notebook, choose a GPU runtime, and run all four code cells.
2. The CPU preparation prints the exact photograph IDs, instance counts,
   batches, optimizer updates and image presentations before training.
3. Keep the optional Drive backup enabled when using Colab. Completed arms
   are hash-checked before restoration; incompatible completed or partial
   folders are preserved instead of being overwritten.
4. Return `baomai-additional-front-pilot-r47-results.zip`. It contains the
   treatment record, labels, source hashes, trial/full receipts, final
   checkpoints, logs and saved evaluation predictions and summaries.

No extra approval cell or model-promotion action is built into the notebook.
An interrupted partial run is preserved; the workflow does not pretend to
resume optimizer state from a weights-only checkpoint. Verified completed
arms can be reused under the same source, data and runtime identity.

The result must be checked against the current misses, total unmatched,
formal duplicate subset, other unmatched and corner-success denominators
together. An unmatched output is not necessarily physical background; it may
be a badly bounded or repeated proposal for a visible cassette. Suppression
and confidence thresholds remain fixed for this comparison.

## What has been verified before handoff

The review's code checks the exact R47 release and label plan on CPU. The
data preflight uses the real 157 original PNGs and original epoch173 bytes,
not placeholder images or weights. Negative checks exercise held-photo,
source/label and stale-artifact rejection. The first two notebook cells are
also run from the published immutable source before final handoff; the
verification record identifies that source and the scope actually executed.

The GPU cells compile and reuse the checked native training recipe, but they
are not reported as executed locally. The [pilot preflight record](../results/review/front-review-r47-20261127/pilot-preflight-verification.json)
covers real input staging, negative guards and independent code review.
The [published-notebook verification](../results/review/front-review-r47-20261127/notebook-verification.json)
records the actual first two cells completed from a fresh remote checkout,
the immutable source revision, exact notebook hash and reused input cache.

This is a development pilot on related camera photographs. It does not create
an independent capture-session test set. PR35 remains selected until the
returned model evidence warrants a separate decision.
