# R48: complete PR137 missed, unmatched and corner-error review

## Scope

This report audits the completed [PR137](https://github.com/Siuuuers/dl/pull/137)
return at `dcb43d7ad061fc3a43f1392398bb8794d9d7168d` against the unchanged
current-R38 development reference. It performs no new inference or training
and does not write any reference labels. The
[result analysis](additional-front-pilot-r47-results.md) explains the exact
training exposure, separate historical-reference scores, visibility strata
and exploratory confidence replay. **PR35 remains selected.**

The fixed gates are box confidence 0.13, ordinary NMS IoU 0.70, extra
postclip suppression at IoU strictly greater than 0.70, matching IoU at least
0.50 and endpoint confidence at least 0.50. The renderer reruns the full
archive and score audit before producing the report.

## Complete current-R38 case inventory

| Candidate category | Cards | Definition |
| --- | ---: | --- |
| Missed targets | 15 | An R38 target has no assigned candidate output at IoU >= 0.50 |
| Matched corner failures | 20 | A matched front has an eligible corner rejected or farther than 5px from reference |
| Formal duplicates | 14 | An unmatched output overlaps an already assigned R38 target at IoU >= 0.50 |
| Other, near gate | 21 | An unmatched nonduplicate output has best reference IoU in `[0.40, 0.50)` |
| Other, positive overlap | 37 | An unmatched nonduplicate output has best reference IoU in `(0, 0.40)` |
| Other, zero overlap | 12 | An unmatched output has zero IoU with every R38 box |
| Recovered targets | 0 | Previously missed R38 targets matched by the candidate |
| Fragile matches | 0 | Candidate matches with diagnostic IoU in `[0.50, 0.55)` |

There are **119 indexed cards**: all 15 misses, all 20 matched corner failures
and all 84 unmatched outputs. The 20 corner cases comprise 17 two-corner
fronts and three one-corner fronts. All have an accepted coordinate farther
than 5px from its reference; there are no rejected-endpoint cases in this
matched-failure set. There are no omitted recoveries or fragile matches.

Misses count reference objects; unmatched counts predictions. A miss and an
unmatched prediction can describe the same physical cassette, and several
unmatched outputs can overlap one target. The 119 cards therefore are not
119 independent physical-object errors. "Other unmatched" is an assignment
category; it must not be relabeled as background without visual adjudication.

All 583 two-corner R38 targets match. The 15 misses are 12 one-corner and
three zero-corner targets. The two newly missed targets are
`new_000008 / R001` (best retained IoU 0.3906, confidence 0.824) and
`new_000010 / R030` (IoU 0.3296, confidence 0.156). Their saved predictions
already exist above the confidence floor; lowering that floor would not
repair those particular box extents.

## Read the PDF

The 46-page PDF has this structure:

| Pages | Contents |
| --- | --- |
| 1 | Completed training and the mixed current-R38 result |
| 2 | Unmatched bins, coordinate convention and identity legend |
| 3 | Complete 20-photograph metrics for both arms |
| 4 | Training visibility versus failure strata; original-image target heights |
| 5 | Both-arm confidence replay and corner-success denominators |
| 6-13 | All 15 candidate misses, with original pixels and both model arms |
| 14-23 | All 20 matched corner failures |
| 24-41 | All 70 other unmatched outputs |
| 42-45 | All 14 formal duplicates, including the assigned output |
| 46 | Evidence, separate historical-reference summaries and limits |

`R###` is a one-based line in the **R38 development label file**. It is not
a supplementary `frame_* / source row` identity. `P###` is the one-based
position in that photograph's **PR137 candidate postclip output list**.
"Primary #" preserves the cached output identity before the extra postclip
suppression; ordinary model NMS has already happened.

The old R47 PDF used **PR133 candidate** `P###` numbers. The user's amendment
cues retain that old meaning. The
[identity receipt](../results/evaluation/front-errors-r48-20261127/prior-reference-identity.json)
verifies that PR133 candidate saved predictions equal PR137 reference saved
predictions for every photograph and every raw/postclip/filter record. The
new candidate's `P###` numbers must not be substituted into those old cues.

Cyan boxes and circles show the named R38 reference and its locatable
corners. Amber boxes and crosses show the model output and accepted
endpoints. On duplicate cards, the dotted amber box is the output that
actually received that target's assignment. Each card preserves an
unannotated original crop. Original-image x increases right and y increases
down; stored `ymin` is the top edge and stored `ymax` is the bottom edge.

For an unmatched target, the comparison selects the highest-overlap retained
output only as a diagnostic aid. Actual match status comes from the unchanged
full-image confidence-ranked greedy scorer. Zero-overlap outputs are not
assigned an unrelated closest row. The original-source images are pinned by
SHA-256, checked for 1280-by-720 dimensions, and never geometrically distorted
to fit their panels.

## What the added analysis can and cannot establish

The four added training photographs contribute 164 two-corner fronts,
eight one-corner fronts and no zero-corner fronts. The PDF contrasts this
with the exclusively partial-front miss stratum. That is a measured
supervision gap; it does not establish a causal explanation for the model
change. Missed partial-front heights span 10 to 45.56 original pixels, while
some two-corner targets match at 9.5px. A universal size or darkness
explanation is unsupported by this audit.

The confidence table replays both arms at 0.13, 0.20, 0.25, 0.30, 0.40,
0.50 and 0.60 with unchanged matching and endpoint gates. The table reports
misses, unmatched outputs, duplicates and 5px corner success with fixed
denominators of 583 two-corner and 54 one-corner targets. Full 10px and
zero-corner detection counts are in the JSON index. This is post-hoc
development exploration, not a selected threshold or unbiased test result.

The cache already passed ordinary NMS and starts at confidence 0.13. It
cannot evaluate lower confidence, alternative ordinary NMS, a learned
refinement filter, image enhancement, tiling or higher-resolution inference.
New annotation proposals are reviewed separately; a rescore under a new
label version must not be described as a model-training improvement.

## Reproduction and verification

Install Pillow 13.0.0 and ReportLab 4.4.9. The PDF uses bundled ReportLab
Vera fonts. No ML framework or GPU is required:

```bash
python scripts/audit_additional_front_pilot_r47_results.py \
  --download-dir work/pr137 \
  --output work/pr137-cpu-audit.json

python scripts/export_front_error_review_r48.py \
  --archive work/pr137/baomai-additional-front-pilot-r47-results.zip \
  --images-dir work/pr137/originals --download-images \
  --output output/pdf/r48-missed-errors-unmatched.pdf \
  --index output/review/r48-error-index.json
```

The downloader verifies the three PR137 wrapper hashes and exact reassembled
archive. The renderer downloads only the 15 original PNGs used by its cards,
checking every hash before rendering. Existing incompatible files are
rejected instead of replaced. Its outputs are the PDF, the full JSON case
index and a PDF hash/count receipt. The source references are read-only.

All 499 inventory entries, 81 archived source files, 40 image/role filter
replays and 12 reference/view/arm score reports passed. All 46 PDF pages
were rendered with Poppler and inspected as contact sheets; selected full
pages were inspected for summary tables, both new misses, corner errors,
unmatched cases, duplicates and evidence. Case coverage and page membership
were checked against the JSON index; no text bounds extend outside a page.
The [verification record](../results/evaluation/front-errors-r48-20261127/verification.json)
records hashes and exact QA scope. The complete machine-readable evidence is
in [`results/evaluation/front-errors-r48-20261127`](../results/evaluation/front-errors-r48-20261127/).
