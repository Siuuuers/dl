# R51 result: fewer partial-front misses, more unmatched outputs

PR146 contains **completed paired R51 training with saved evaluation outputs**.
It is not a notebook setup repair. Independent CPU replay verifies the returned
archive, training protocol, labels, source bytes, counters and all four scoring
references. The treatment has a measurable tradeoff on these development photos:
R50 misses fall from **15 to 11**, while unmatched outputs rise from **84 to 112**.
This does not support automatic model promotion. PR35 remains the selected model.

Sources: [PR146](https://github.com/Siuuuers/dl/pull/146),
[verification receipt](../results/evaluation/pr146-r51-20261127/verification.json),
[replayed summary](../results/evaluation/pr146-r51-20261127/summary.json),
[case inventory](../results/evaluation/pr146-r51-20261127/case-inventory.json),
[audit program](../scripts/audit_partial_classification_r51_results.py).

## What was actually trained

Both arms used exactly the PR137 candidate inputs: **142 training photographs,
3,999 cassette fronts, 7,259 locatable corner slots and 739 unavailable slots**.
The photographs comprise the original 137 training PNGs and five supplementary
frames: `frame_0146`, `frame_0160`, `frame_0178`, `frame_0371`, `frame_0383`.
There are 13 empty training photos. Of the 3,999 targets, 3,397 have two locatable
corners, 465 have one, and 137 have none. These five frame photos therefore
**were used for training**. Other reviewed frames are not automatically training
examples; the whole-photo release gate and the exact staged inventory decide that.
R51 intentionally reused the PR137 candidate label bytes in both arms; it did not
introduce later supplementary review amendments into training.

The trainer's unchanged validation input is 20 photographs with 664 targets,
1,227 locatable corner slots, 101 unavailable slots and three empty photos.
The four separate evaluation references below relabel the same 20 photographs;
their differing object counts are intentional and are not four independent test sets.

| Controlled setting | Both full arms |
| --- | --- |
| Starting checkpoint | Original `cassette_lower_epoch173.pt` |
| Recipe | `reviewed-incremental-fronts-seed46-v1` |
| Seed / epochs | 46 / 30 |
| Batch / image size | 4 / 1280 |
| Optimizer / arithmetic | AdamW / FP32 |
| Completed batches | 1,080 per arm |
| Measured optimizer updates | 80 per arm |
| Image presentations | 4,260 per arm |
| Dataset and labels | Byte-identical between arms |

The discarded resource trial completed four batches, 16 image presentations and
four optimizer updates. Each full arm then started independently from epoch173.
The result archive contains a final `last.pt` and 30 finite CSV epoch records for
each arm. Controller receipts also reconcile every epoch, batch, per-image
exposure and optimizer update schedule.

The only planned treatment was **classification BCE weighting on foreground
anchors already assigned to a post-augmentation target with fewer than two
locatable keypoints**: factor 1 for the reference, factor 2 for the candidate.
The native assignment, soft target values, unweighted normalizer, box loss,
keypoint loss, visibility loss and RLE formulas remained unchanged. This is a
classification-only comparison, not a new label or data-size comparison.
The consumed target count differs from 3,999 × 30 because augmentation may remove
or clip targets; both full arms record 118,835 consumed targets, including
18,626 partial targets.

The recorded CPU guard verifies that factor 1 is native-equivalent and factor 2
changes only the intended score gradients at fixed outputs. The returned
reference prediction cache is also **exactly equal to the previous
PR137 candidate cache** on all 20 photos, including raw predictions, postclip
predictions and filtering records. This is equality of saved prediction values;
it does not claim that the serialized checkpoints have equal bytes.

## Detection results on all four references

These are postclip results at the unchanged box-confidence operating point
0.13, matching IoU 0.5 and endpoint confidence 0.5. The primary frozen benchmark
remains frozen666. R38 and R50 provide the reviewed annotation views.

| Reference | Targets | Matches, reference → candidate | Misses | Unmatched outputs | Duplicate subset |
| --- | ---: | ---: | ---: | ---: | ---: |
| Frozen666 | 666 | 632 → 635 | 34 → 31 | 91 → 120 | 17 → 23 |
| Historical663 | 663 | 633 → 634 | 30 → 29 | 90 → 121 | 17 → 23 |
| R38 | 654 | 639 → 643 | 15 → 11 | 84 → 112 | 14 → 24 |
| R50 | 654 | 639 → 643 | 15 → 11 | 84 → 112 | 13 → 23 |

No target matched by the reference becomes missed by the candidate in any of
these references. The candidate returns 755 postclip outputs compared with 723
from the reference. On R50 that difference comprises four additional matches and
28 additional unmatched outputs. It is a count difference, not an assertion that
28 specific prediction identities persisted unchanged across models.

The README's reported historical663 reference screen value is 31 misses. The
archived JSON and independent score replay instead give **633 matches + 30
misses = 663 targets**. This report uses the replayed value; it does not speculate
about the origin of the screen discrepancy.

### Unmatched outputs are not all duplicates

The original scorer defines a formal duplicate as an unmatched prediction with
IoU at least 0.5 to a same-class target that already has a match. Its remaining
unmatched predictions are recorded as `other_fp`. That second category is a
scoring category, not a visual determination that the prediction is physically
background. The review still has to distinguish poor localization, object
ownership, missing reference targets and genuine background responses.

| R50 postclip category | Reference | Candidate | Difference |
| --- | ---: | ---: | ---: |
| Formal duplicate outputs | 13 | 23 | +10 |
| Other unmatched outputs | 71 | 89 | +18 |
| All unmatched outputs | 84 | 112 | +28 |

The existing postclip filter removes nine outputs in each arm without changing
any matched/missed target identity. On R50, raw unmatched counts 93/121 become
84/112: eight formal duplicates and one other unmatched output are removed per
arm. This is an observed benefit of the existing filter, not evidence that a
stronger overlap cutoff would preserve densely adjacent true fronts.

### Original-image review of all 112 candidate unmatched outputs

The [visual ledger](../results/evaluation/pr146-r51-20261127/visual-observations.json)
records a new agent-assisted pass over every raw/overlay pair and wider original
context. It preserves the formal matcher categories above. Its primary visual
subtypes are provisional interpretations, not independent human adjudication:

| Visual subtype | Outputs | Appropriate next investigation |
| --- | ---: | --- |
| Front extent or fragment | 63 | Distinguish the target's own front from a fragment, neighbouring front or gripper; review localization supervision. |
| Formal duplicate | 23 | Inspect same-instance ranking and suppression while protecting adjacent-front recall. |
| Rack or gripper | 8 | Confirm hardware identity and the training set's verified negative coverage. |
| Keyboard or other background | 5 | Inspect background/context responses on training-eligible data. |
| Overlap with an explicit reference exclusion | 6 | Preserve the exclusion and document its effect; some boxes also cover retained material. |
| Ambiguous partial fragment | 7 | Keep uncertainty and inspect wider context before any label decision. |

The six policy-sensitive outputs are not six proven policy-only errors.
In particular, the user-directed excluded partial front in `new_000009` is not
verified background. Neither these development images nor their newly reviewed
negative regions should be quietly moved into training. The largest visual group
contains real cassette material with a poorly delimited extent; a single
confidence adjustment cannot be assumed to repair those coordinates.

IDs use `outNNN` for the one-based candidate postclip position and `PNNN` for
the retained primary prediction ID. The two values can differ. All ledger joins
use exact image/arm/output-position identities, not historical P numbers from a
different model. No labels, thresholds or predictions change during this review.

All three empty scenes (`new_000146`, `new_000147`, `new_000148`) retain zero
outputs in both arms, before and after postclip filtering. The increased
unmatched workload occurs in occupied scenes.

| Photo with increased unmatched count | Reference | Candidate | Difference |
| --- | ---: | ---: | ---: |
| `new_000008` | 4 | 10 | +6 |
| `new_000010` | 4 | 9 | +5 |
| `new_000009` | 20 | 24 | +4 |
| `new_000090` | 11 | 15 | +4 |
| `new_000007` | 4 | 7 | +3 |
| `new_000059` | 1 | 3 | +2 |
| `new_000011` | 12 | 13 | +1 |
| `new_000054` | 2 | 3 | +1 |
| `new_000086` | 0 | 1 | +1 |
| `new_000089` | 12 | 13 | +1 |

The remaining photos have unchanged unmatched counts; none has a decrease.

## Which targets changed, and how accurate were their corners?

The following are **one-based R50 reference row IDs**, not supplementary
`frame_*:source:*` identities and not prediction positions.

| Recovered target | Locatable corners | Candidate corner result |
| --- | ---: | --- |
| `new_000010 / R030` | 1 | Returned corner within 5 px; error 3.128 px |
| `new_000010 / R033` | 0 | Box matched; no corner-accuracy claim is possible |
| `new_000011 / R059` | 1 | Returned corner within 5 px; error 1.968 px |
| `new_000090 / R039` | 1 | Returned corner within 5 px; error 1.088 px |

These four recoveries are shared by R38 and R50. Frozen666 and historical663
have their own identity ledgers in `summary.json`; row numbering and target
geometry must not be carried across reference versions without a mapping.

| R50 corner stratum | Eligible targets | Matched, reference → candidate | All locatable corners within 5 px | Within 10 px |
| --- | ---: | ---: | ---: | ---: |
| Zero locatable corners | 17 | 14 → 15 | Not applicable | Not applicable |
| One locatable corner | 54 | 42 → 45 | 39 → 42 | 41 → 44 |
| Two locatable corners | 583 | 583 → 583 | 566 → 567 | 582 → 582 |

One already matched two-corner target crosses the 5 px boundary:
`new_000010 / R020`, whose largest endpoint error changes from 5.094 to 4.984 px.
This approximately 0.11 px crossing contributes the entire net gain in the
two-corner within-5-px count. It should not be described as a broad improvement in
corner placement. Under frozen666, the two-corner within-5-px count stays
511/584 and within-10-px stays 575/584.

The candidate has **19 matched R50 targets that fail the all-locatable-corners
within-5-px criterion**: 16 two-corner targets and three one-corner targets.
The machine-readable case ledger retains both arms' predictions, endpoint
errors, confidence values and exact reference geometry for these cases.

### Eleven remaining misses

| Photo | Remaining R50 rows | Corner visibility |
| --- | --- | --- |
| `new_000008` | R001, R033, R034 | One corner each |
| `new_000009` | R051, R054 | One, then zero corners |
| `new_000011` | R055, R056, R057, R060 | One, one, one, then zero corners |
| `new_000088` | R051 | One corner |
| `new_000089` | R048 | One corner |

All 11 remaining misses are partial fronts: nine have one locatable corner and
two have none. Their miss identities are unchanged between raw and postclip
evaluation, so the added postclip filter does not cause these remaining misses.
However, PR146 does not include a full decoded tensor trace for the new
candidate. Historical R49 confidence/NMS/localization diagnoses cannot simply be
relabelled as the current causes. A new stage-specific diagnosis would require
the current model's decoded outputs and the unchanged inference contract.

## What the verification proves

| Item | Verified identity or scope |
| --- | --- |
| PR146 head | `b7f64852e3de2a3ac8299783065ad56c03e0dea6` |
| Executed training source | `da5a67d4bec8171246c56c6b03b34ee20fc519e0` |
| Full returned ZIP | 43,528,788 bytes; SHA256 `9538932145e90201d407c339b7da1f73c5c905935853191ab9683b221bd5a09e` |
| Payload inventory | All 518 listed payloads, their exact bytes and CRCs |
| Training source | 85 Python files; archive, local bytes and immutable Git tree agree |
| Training labels | All 324 staged train/val files across both roles agree with PR137 candidate bytes |
| Runtime | Tesla T4; Python 3.13.16; Torch 2.14.1+cu130; Ultralytics 8.4.182 |
| Counter evidence | Trial and both complete arms; assignment totals, branch schedule, image exposure, optimizer schedule and recorded loader gates |
| Score replay | 16 aggregate tables, 320 image-level reports, 40 postclip reconstructions |

This is an independent **archive and saved-prediction replay**. It does not rerun
training, unpickle/load the returned checkpoints, infer new predictions or
independently execute a new image-loader preflight. Image identity in the result
audit is bound by the immutable image manifests and archived actual-loader
receipts. The R52 PDF workflow separately downloads and hashes the original
images used for visual review.

To reproduce the CPU result audit from downloaded pinned archives:

```bash
python -B scripts/audit_partial_classification_r51_results.py \
  --archive /path/to/baomai-partial-classification-r51-results.zip \
  --prior-archive /path/to/baomai-additional-front-pilot-r47-results.zip \
  --output /path/to/r51-audit
```

Omit the two archive arguments to download their pinned wrapped parts. The audit
needs the immutable R51 source Git tree available locally; a shallow checkout
can fetch it with `git fetch --depth=1 --filter=blob:none origin
da5a67d4bec8171246c56c6b03b34ee20fc519e0`. Existing incompatible audit files are
preserved rather than overwritten.

## Next useful experiment

R51 provides evidence that increasing classification weight can recover some
partial fronts while increasing the number of unmatched outputs. It does not
show that a broader confidence reduction, more training, different image size or
another loss change would improve the combined result. The next concrete step is
the R52 CPU annotation-quality study: reproduce the four-side geometry review,
inspect the original pixels, record supported boundary amendments and compare
before/after review workload with stable identities.

These small boundary amendments are not yet measured training improvements.
Keep annotation quality, model error counts and review time as distinct outcomes.
The same 20 photos have guided repeated development, so this result is a
within-dataset comparison. A future untouched capture session can test broader
generalization when new data becomes available.
