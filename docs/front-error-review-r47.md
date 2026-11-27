# R47: complete missed, unmatched and corner-error review

## Result identity

[PR134](https://github.com/Siuuuers/dl/pull/134) at
`d49a55d7c2240bbb18565e643cf01e691d39a7a3` publishes R46 annotation and CPU
replay work. It does not contain another model-training return. This R47 report
expands the error review of the completed
[PR133](https://github.com/Siuuuers/dl/pull/133) result at
`22a49e59adb2aeb9007c3fdac4c70ede2fb67802`.

The two R45 arms trained the original 137 photographs. The candidate also
trained `frame_0383`: 43 additional fronts and 86 locatable corner slots. Both
arms completed 30 epochs, 1,050 batches and 78 optimizer updates. The original
20 development photographs remained outside training. See
[the verified R45 result](additional-front-pilot-r45-results.md) for the full
training lineage and historical-reference comparison.

This R47 review performs **no fresh inference or training** and makes **no
development-reference label changes**. PR35 remains selected. The PDF uses
current R38 labels consistently; the alternative historical references remain
separate summaries of the same development photographs.

## Complete current-R38 inventory

The fixed operating point is box confidence 0.13, ordinary NMS IoU 0.70,
extra postclip suppression at IoU strictly greater than 0.70, matching IoU
at least 0.50, and endpoint confidence at least 0.50.

| Metric | Reference | Candidate |
| --- | ---: | ---: |
| Current R38 fronts | 654 | 654 |
| Missed targets | 11 | 13 |
| All unmatched predictions | 86 | 90 |
| Formal duplicate subset | 22 | 18 |
| Other unmatched subset | 64 | 72 |
| Both locatable corners within 5px | 571 / 583 | 569 / 583 |
| Both locatable corners within 10px | 582 / 583 | 581 / 583 |

The candidate PDF contains 123 indexed cards:

- All 13 missed R38 targets.
- The one recovered R38 target and one fragile match with IoU in `[0.50, 0.55)`.
- All 18 matched fronts that fail the 5px check over their locatable corners:
  14 two-corner fronts and four one-corner fronts. Seventeen have an accepted
  coordinate more than 5px from its reference; one has a rejected endpoint.
- All 72 other unmatched outputs.
- All 18 formal duplicate outputs, with the assigned output drawn as a dotted
  box for comparison.

Misses count reference objects; unmatched counts model outputs. A miss and an
unmatched output can refer to the same physical cassette. These totals must
not be added as independent physical-object errors.

## What “other unmatched” contains

The following are reproducible overlap bins, not manually established physical
classes or new evaluation thresholds:

| Candidate output category | Count | Exact definition |
| --- | ---: | --- |
| Formal duplicate | 18 | An unmatched output overlaps an already assigned R38 target at IoU >= 0.50 |
| Other, near gate | 17 | Highest IoU with any R38 target is in `[0.40, 0.50)` |
| Other, positive overlap | 44 | Highest IoU is in `(0, 0.40)` |
| Other, zero overlap | 11 | IoU is zero with every R38 target box |

The 72 “other” outputs are not automatically background. A second output for
an already matched cassette can fall below the formal duplicate gate because
its box is distorted. A real visible fragment can also be unmatched because
the prediction includes an occluder or because the annotation definition needs
review. Zero reference-box overlap alone establishes neither a physical
background error nor an omitted annotation.

Of the 17 near-gate outputs, nine have a best-overlap target that the candidate
missed; the remaining eight overlap a target already matched by the candidate.
Across all “other” bins, 21 outputs have a missed best-overlap target. These are
21 predictions, not 21 distinct missed cassettes.

## Focused next steps from the evidence

The three newly missed targets still have retained, overlapping predictions:

| Current R38 target | Candidate best box IoU | Candidate box confidence |
| --- | ---: | ---: |
| `new_000009 / R051` | 0.4916 | 0.723 |
| `new_000011 / R059` | 0.4702 | 0.292 |
| `new_000089 / R048` | 0.4973 | 0.834 |

The original crops show partial fronts next to a gripper. The candidate boxes
extend across material outside the named R38 visible-front extent. Lowering
confidence would not repair those existing boxes. The useful review is
physical front ownership, box extent and consistency of occlusion policy.
These observations are review issues, not permission to rewrite the frozen
evaluation labels to fit the model.

`new_000009`, `new_000011`, `new_000089` and `new_000090` contain 54 of the
90 candidate unmatched outputs and nine of the 13 misses. Review those
regions before selecting a global suppression change. Suppression can reduce
extra outputs while also removing a neighboring true cassette; the current
fixed reference and corner criteria must remain in the comparison.

The rejected endpoint case is `new_000008 / R001`: the front is matched, but
its locatable right endpoint has confidence 0.471, below the unchanged 0.50
endpoint gate. The other 17 matched corner-failure cards include the numerical
left/right distances. Missing reference corners are not guessed or scored.

The additional fragile match is `new_000056 / R014`, with candidate box
IoU 0.5048. It is still a match. Showing it prevents a threshold-adjacent match
from being described as robust solely because it passes the gate.

## Reading the PDF and index

The 44-page report uses the following order:

| Pages | Contents |
| --- | --- |
| 1-3 | Result status, definitions, complete 20-photo metric table |
| 4-11 | All 13 misses, one recovery and one fragile match |
| 12-20 | All 18 matched corner failures |
| 21-38 | All 72 other unmatched outputs |
| 39-43 | All 18 formal duplicates |
| 44 | Sources, historical-reference comparison and limits |

`R###` is a one-based line in the **R38 development label file**. It is not a
supplementary `frame_*` source-row identity. `P###` is the one-based model-output
index in that photograph's **postclip candidate list**. “Primary #” identifies
the cached output before the extra postclip suppression; ordinary model NMS
has already occurred. The index preserves both identities.

Cyan boxes and circles show the named R38 target and locatable corners. Amber
boxes and crosses show a model output and its accepted endpoints. On duplicate
cards, the dotted amber box is the output that actually received the target
assignment. Each card also retains an unannotated original crop.

For an unmatched target, the comparison panel selects the highest-overlap
retained output only as a diagnostic aid. Actual match status comes from the
unchanged full-image, confidence-ranked greedy scorer. When all overlaps are
zero, no unrelated “closest row” is assigned to the output.

The JSON index stores exact boxes, corners, confidence, per-image metrics,
classification, image SHA-256 values and page membership. It contains only
read-only diagnostic records. The current R38 and historical label files are
never written by the renderer.

## Reproduction

Install Pillow 13.0.0 and ReportLab 4.4.9; no ML framework or GPU is required.
The renderer uses embedded ReportLab Vera fonts for portable, deterministic
PDF output.

```bash
python scripts/audit_additional_front_pilot_r45_results.py \
  --download-dir work/pr133 \
  --output work/pr133-cpu-audit.json

python scripts/export_front_error_review_r47.py \
  --archive work/pr133/baomai-additional-front-pilot-r45-rebuilt-results.zip \
  --images-dir work/pr133/originals --download-images \
  --output output/pdf/r47-missed-errors-unmatched.pdf \
  --index output/review/r47-error-index.json
```

The first command downloads and reassembles the exact two PR133 chunks. The
renderer reruns the independent archive/receipt/scoring audit and downloads
15 original PNGs needed for its cards. Every downloaded original must match
the recorded source SHA-256 before it can appear in the PDF. Existing
incompatible files are rejected rather than silently replaced.

The audit passed all 334 inventory entries, 40 image/role postclip replays and
12 reference/view/arm score reports. All 44 PDF pages were rendered and
visually inspected, and exact case coverage was checked against the archive.
Recorded checkpoint hashes establish file consistency, not independent proof
of inference execution. No new performance claim follows from this CPU review.

Machine-readable evidence lives in
[`results/evaluation/front-errors-r47-20261127`](../results/evaluation/front-errors-r47-20261127/).
