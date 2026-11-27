# R47 returned result: expanded supplementary training completed

## Decision and source

[PR137](https://github.com/Siuuuers/dl/pull/137), at
`dcb43d7ad061fc3a43f1392398bb8794d9d7168d`, contains a completed two-arm T4
training return. The candidate reduces unmatched outputs but increases missed
targets and reduces the strict 5px corner-success count. **Keep PR35 selected.**
This is a mixed development result, not evidence for automatic model promotion.

The prepared experiment is described in
[the R47 pilot plan](additional-front-pilot-r47.md). Its training source is
`96b05f0d2149ab71c9211c81f35be67f79344b81`; the published notebook is in
[PR136](https://github.com/Siuuuers/dl/pull/136) at
`d0136ae8b2283d38da953e8f109e542029682678`.

The 42,395,650-byte result archive has SHA-256
`fc8c40357c83831d17f88b0bb5a5ecd2e0cb1a8128be697e37e80620e4981023`.
The [CPU audit receipt](../results/evaluation/front-errors-r48-20261127/return-verification.json)
records the source closure, labels, runtime, exposure, model hashes and exact
score replay. The [R48 error review](front-error-review-r48.md) provides every
current-R38 error card and the additional visibility/filter diagnostics.

## What was actually trained

The original 137 effective-R37 training photographs were used in both arms.
The reference also used supplementary `frame_0383`. The candidate used
`frame_0146`, `frame_0160`, `frame_0178`, `frame_0371` and `frame_0383`.
Thus the supplementary frame photographs **were trained** in this run. The
candidate adds four complete photographs and 172 fronts over the reference.
Shared `frame_0383` labels are byte-identical to those used by the PR133
candidate. Held supplementary photographs were excluded by the whole-photo
eligibility gate. The original 20 development photographs remained outside
training.

| Measured quantity | Reference | Candidate |
| --- | ---: | ---: |
| Training photographs | 138 | 142 |
| Front instances | 3,827 | 3,999 |
| Locatable corner slots | 6,923 | 7,259 |
| Unavailable corner slots | 731 | 739 |
| Genuine empty photographs | 13 | 13 |
| Completed epochs | 30 | 30 |
| Batches | 1,050 | 1,080 |
| Actual optimizer updates | 78 | 80 |
| Image presentations | 4,140 | 4,260 |

Both arms started independently from the original epoch173 initializer using
the pinned seed46 recipe: 1280 input, batch 4, AdamW, FP32, 30 epochs and the
final `last.pt`. They did not continue from a PR133 final checkpoint. The
discarded resource trial completed four batches, 16 image presentations and
four optimizer updates with a recorded parameter change.

The full-arm receipts agree on Tesla T4, Python 3.13.16, torch 2.14.1+cu130,
CUDA 13.0, cuDNN 92800 and Ultralytics 8.4.182. Every recorded optimizer-update
batch index agrees with the pinned native warmup/accumulation schedule.
The candidate receives 120 more image presentations, 30 more batches and two
more updates. This is an equal-epoch experiment with additional exposure as
part of the treatment; it does not isolate extra data from extra computation.

| Final checkpoint | SHA-256 |
| --- | --- |
| Reference | `605f067d6fc43ed263f956190132371921228ac93efb0c169f22ff14941c0308` |
| Candidate | `222bf3c91a651183047a10eb1a50e16ae8fe999f15372afe005bfca2d0b157f9` |

## Fixed development results

The fixed operating point is box confidence 0.13, ordinary model NMS IoU 0.70,
extra postclip suppression at IoU strictly greater than 0.70, matching IoU at
least 0.50 and endpoint confidence at least 0.50. All counts below use the
retained postclip outputs. Corner distances are measured in the original
1280-by-720 image coordinates.

| Current R38: 654 targets | Reference | Candidate | Candidate minus reference |
| --- | ---: | ---: | ---: |
| Missed targets | 13 | 15 | +2 |
| All unmatched predictions | 90 | 84 | -6 |
| Formal duplicate subset | 18 | 14 | -4 |
| Other unmatched subset | 72 | 70 | -2 |
| Both locatable corners within 5px | 569 / 583 | 566 / 583 | -3 |
| Both locatable corners within 10px | 581 / 583 | 582 / 583 | +1 |

There are no recovered R38 targets. The two newly missed targets still have
retained overlapping predictions, so lowering confidence would not correct
their existing box geometry:

| Current R38 target | Candidate best retained box IoU | Candidate confidence |
| --- | ---: | ---: |
| `new_000008 / R001` | 0.3906 | 0.824 |
| `new_000010 / R030` | 0.3296 | 0.156 |

Both have only one locatable reference corner. Their original crops, both
model arms and exact coordinates are included in the error PDF. A failed
IoU match is a diagnostic starting point; it does not by itself establish
whether the physical front boundary, the predicted boundary or both require
review.

The three label versions remain separate. They refer to the same 20
development photographs, not three independent test sets:

| Reference version | Misses ref / cand | Unmatched ref / cand | Duplicate ref / cand | Both within 5px ref / cand |
| --- | ---: | ---: | ---: | ---: |
| Current R38, 654 fronts | 13 / 15 | 90 / 84 | 18 / 14 | 569 / 566, denominator 583 |
| Frozen666 historical primary | 33 / 34 | 98 / 91 | 22 / 17 | 518 / 511, denominator 584 |
| Historical663 diagnostic | 30 / 30 | 98 / 90 | 22 / 17 | 518 / 511, denominator 584 |

The frozen666 and historical663 two-corner 10px counts are 575 in both arms,
with denominator 584. None of the three genuine empty development scenes
(`new_000146`, `new_000147`, `new_000148`) has a retained prediction in either
arm. The frozen references are unchanged by this result audit or error renderer.

## The clearest observed gap is partial-front supervision

All 583 R38 targets with two locatable corners are matched in both arms. The
candidate's 15 misses comprise 12 of the 54 one-corner targets and three of
the 17 zero-corner targets. This is a measured visibility stratum, not proof
that occlusion alone causes every miss.

| Locatable corners per front | Training reference | Training candidate | Added fronts | R38 targets | R38 misses ref / cand |
| --- | ---: | ---: | ---: | ---: | ---: |
| Two | 3,233 | 3,397 | 164 | 583 | 0 / 0 |
| One | 457 | 465 | 8 | 54 | 10 / 12 |
| Zero | 137 | 137 | 0 | 17 | 3 / 3 |

`frame_0146`, `frame_0160` and `frame_0371` each contribute 43 added two-corner
fronts. `frame_0178` contributes 35 two-corner and eight one-corner fronts.
The shared `frame_0383` contributes 43 two-corner fronts in each arm. Therefore
164 of the 172 added fronts have two locatable corners, while every current
candidate miss lies in the one- or zero-corner stratum. This mismatch in
additional supervision is a practical reason to prioritize carefully reviewed
partial fronts in the next data collection or training experiment. It is not
causal proof of why these model outputs changed.

Box height alone also fails to explain all misses:

| Candidate target group | Targets | Minimum height | Median height | Maximum height |
| --- | ---: | ---: | ---: | ---: |
| Missed partial fronts | 15 | 10.00px | 23.00px | 45.56px |
| Matched partial fronts | 56 | 11.00px | 26.18px | 60.00px |
| Matched two-corner fronts | 583 | 9.50px | 29.00px | 67.00px |

These are axis-aligned reference box heights in original image pixels. Of the
11 targets with height from 8 to below 16px, the reference misses two and the
candidate misses three. The 411 targets from 16 to below 32px have eight
misses in both arms; the 232 targets at least 32px high have three reference
and four candidate misses. Heights are rounded to 0.001px only when assigning
bins to avoid normalized-label floating-point noise at exact boundaries.
No target is below 8px. The audit does not measure a causal brightness effect.

## Unmatched outputs and filtering

The 84 candidate unmatched outputs consist of 14 formal duplicates, 21 with
best reference overlap in `[0.40, 0.50)`, 37 with positive overlap below 0.40
and 12 with zero overlap. The latter three bins are the 70 other unmatched
outputs. They are scorer categories, not established physical-background
classes. One missed target can have several unmatched predictions, and one
physical cassette can appear in both a miss card and an unmatched card.

For 56 of the 84 unmatched outputs, the highest-overlap reference target has
one locatable corner; for four it has zero and for 12 it has two. The remaining
12 have no reference-box overlap. These are geometric associations, not
validated physical identities. Four photographs (`new_000009`, `new_000011`,
`new_000089`, `new_000090`) contain 55 of the unmatched outputs and nine of the
misses, making them useful review priorities.

The existing extra postclip suppression already removes nine candidate
outputs: 732 raw cached outputs become 723 retained outputs; unmatched drops
from 93 to 84 and formal duplicates from 22 to 14. Misses remain 15. Here
"raw" means after ordinary model NMS and before the additional postclip
suppression, not a pre-NMS model tensor.

The following higher-confidence replay is **exploratory and post hoc on the
development set**. It does not change the selected model or any default.
All matching and endpoint gates remain fixed:

| Confidence | Misses ref / cand | Unmatched ref / cand | Duplicates ref / cand | Two-corner within 5px ref / cand | One-corner within 5px ref / cand |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0.13 | 13 / 15 | 90 / 84 | 18 / 14 | 569 / 566 | 40 / 39 |
| 0.20 | 15 / 17 | 71 / 63 | 14 / 10 | 569 / 566 | 40 / 39 |
| 0.25 | 17 / 18 | 61 / 51 | 12 / 7 | 569 / 566 | 39 / 38 |
| 0.30 | 18 / 19 | 45 / 44 | 10 / 6 | 569 / 566 | 39 / 38 |
| 0.40 | 21 / 24 | 30 / 36 | 6 / 5 | 569 / 565 | 38 / 37 |
| 0.50 | 23 / 25 | 25 / 21 | 5 / 2 | 568 / 565 | 37 / 36 |
| 0.60 | 30 / 29 | 17 / 14 | 4 / 1 | 568 / 565 | 34 / 34 |

The two-corner denominator is 583 and the one-corner denominator is 54 in
each arm. Full 10px counts and zero-corner detection denominators are in the
[JSON index](../results/evaluation/front-errors-r48-20261127/error-index.json).
For example, raising candidate confidence to 0.25 removes 33 unmatched
outputs but adds three misses. Fewer outputs alone is not an improvement.
The cache starts at confidence 0.13 and already passed ordinary NMS; it
cannot test lower thresholds, pre-NMS alternatives, image enhancement,
tiling or higher-resolution inference. Those require separately recorded
inference runs under the same frozen reference and coordinate mapping.

## Prior cue identities are preserved

The user's `P###` amendment cues came from the older R47 error PDF of the
PR133 candidate. An exact comparison verifies that all 20 photographs' raw,
postclip and filter JSON values for **PR133 candidate equal PR137 reference**.
The canonical prediction SHA-256 is
`86d6c1ace58484f574ab0abde41cc94fa2f9ca1fd9ccbae9d346af922bcfe3c9`.
The [identity receipt](../results/evaluation/front-errors-r48-20261127/prior-reference-identity.json)
records the byte-pinned source archives and serialization. The checkpoint
files have different hashes; this comparison claims identical saved
predictions, not byte-identical checkpoints.

New R48 error-PDF `P###` numbers identify **PR137 candidate** outputs. They must
not silently replace the old cue numbers. Proposed R48 label amendments and
any rescoring under them are separate annotation-version work; they do not
demonstrate a new training gain.

The completed [R48 reference-sensitivity analysis](front-reference-sensitivity-r48.md)
now confirms that these amendments leave every target assignment, the 13/15
misses and the 90/84 total unmatched outputs unchanged in the two saved arms.
Only a few duplicate/other classifications and one reference-arm corner success
change. This result is kept separate from the original R38 table above.

## Reproduce the result audit

The CPU audit uses the standard library and the repository's pinned scoring
helpers. It does not instantiate a model or require a GPU:

```bash
python scripts/audit_additional_front_pilot_r47_results.py \
  --download-dir work/pr137 \
  --output work/pr137-cpu-audit.json
```

To repeat the old-cue identity comparison after obtaining the PR133 archive:

```bash
python scripts/audit_additional_front_pilot_r47_results.py \
  --archive work/pr137/baomai-additional-front-pilot-r47-results.zip \
  --output work/pr137-cpu-audit.json \
  --prior-archive work/pr133/baomai-additional-front-pilot-r45-rebuilt-results.zip \
  --identity-output work/pr133-candidate-pr137-reference.json
```

All 499 inventory entries, 81 archived source files, 40 image/role postclip
replays and 12 reference/view/arm scoring reports passed. The source and
receipts bind the returned run's actual loader-coordinate and exposure checks.
This CPU audit does not independently rerun that loader, training or inference
from the final checkpoint. It verifies a completed return and replays its
saved predictions. The development images have been repeatedly inspected;
a final thesis generalization claim needs a separately reserved capture
session and an evaluation policy fixed before that test is viewed.
