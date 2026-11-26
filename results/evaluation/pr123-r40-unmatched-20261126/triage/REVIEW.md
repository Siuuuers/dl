# R40 read-only triage of the 64 other-unmatched R39 outputs

All 64 outputs were reviewed against all 14 relevant original full-frame contexts and exact-coordinate crops. All 14 image hashes match the accepted R38 manifest. This is a diagnostic review, not a label amendment or a new model result. The 21 formal duplicates are outside this file's scope.

The accepted starting point remains 728 predictions, 643 matched targets, 11 misses and 85 unmatched outputs (21 formal duplicates plus these 64 others). Every identifier below is the original primary pID, mapped through `primary_prediction_ids`, not the index in the shorter clipped cache.

## What the 64 outputs represent

| Review group | Predictions | Interpretation |
|---|---:|---|
| Around already matched accepted targets | 35 | Extent, seam or occlusion variants around 23 best-associated reference targets. Two (008 p38, 058 p19) are specifically mixed front/jaw responses. This is not a claim that all 35 are independently verified physical duplicates. |
| Around currently missed accepted targets | 14 | Alternative geometries for six missed reference targets, detailed below. Fourteen proposals do not mean fourteen different cassettes. |
| Explicitly excluded target regions | 4 | Three outputs on excluded 009 R37 row52; one on excluded 010 R37 row30. Exclusions remain unchanged. |
| Clear background/fixture regions | 8 | Two on the 054 base support, two on the 089 white top fixture, four on one lower-right blue/gray peripheral in090. |
| Uncertain | 3 | 008 p40 jaw/rack with weak front overlap; 053 p22 top sliver overlapping both accepted and excluded regions; 055 p28 possible clipped front/fixture. |

Mechanical overlap alone places 51/64 at IoU at least 0.10 with an accepted target, including 33 at 0.30–0.50. Ten have zero accepted-reference overlap. These measurements are association clues; physical identity and surface ownership require the images.

### The six missed targets that already have output proposals

| Image | Accepted R38 row | Other-unmatched primary pIDs | Best IoU | Best pID |
|---|---:|---|---:|---:|
| new_000010 | 30 | 35, 36 | 0.374125 | 36 |
| new_000011 | 55 | 56, 58, 63 | 0.487830 | 56 |
| new_000011 | 56 | 57, 64, 67 | 0.406508 | 64 |
| new_000011 | 57 | 61, 62, 66 | 0.338829 | 61 |
| new_000088 | 51 | 56, 57 | 0.456572 | 57 |
| new_000090 | 39 | 48 | 0.489707 | 48 |

All remain misses under the fixed 0.50 gate. In particular, 011 p56 and090 p48 are high-confidence proposals near the correct visible front whose accepted-reference IoU is just below the gate. They should not be described as a complete failure to notice an object. Conversely, changing the IoU threshold would change the scoring contract, not improve the model.

### Explicit exclusions

The removed R37 `new_000009:52` is the small bottom front immediately left of the jaw. The original pixels support a partial front; the user-directed exclusion changes the target population. Current p62, p69 andp71 cover that region with differing box widths, with no overlap to accepted R38 rows. `new_000010` p29 overlaps the excluded R37 row30 at IoU0.779704. None should trigger an automatic re-addition.

The earlier R34 `new_000009:56/57` deletions in the R35 ledger were also checked; no additional clear association was found among these 64 proposals. `new_000053` p22 has partial overlap with excluded R37 row1 but also with current row1, so it stays uncertain rather than being counted as a clean exclusion match.

### Background and contextual evidence

`new_000054` p23 andp24 cover the rack/base support strip below the cassette stack; a binder clip and uninterrupted structural strip are visible in the original context. `new_000089` p52 andp62 cover the white horizontal fixture above the mesh. Adjacent original frame090 shows the continuous white rail and its two mounting bolts, resolving an otherwise cassette-like pale strip. `new_000090` p51,p56,p59,p62 are four extents of the same blue/gray background item outside the rack.

These eight outputs concern three broad scene regions, not eight separate hallucinated cassettes. No observed additional unlabeled cassette was established by this audit; the three uncertain cases retain uncertainty, especially the 055 top-clipped strip.

## Why “labels or model?” is not a binary choice

The accepted R33 standard already defines the box as an envelope of **observed frontal material** including its owned rim/lip, and excludes all-jaw or rack-only boundary strips. It separately permits predicted hidden corners where the accepted annotation has no observed corner. Those are two distinct contracts.

Many outputs on009,011,088 and089 begin at a real visible front and then extend under the white jaw. A wider inferred box may plausibly cover an occluded cassette, but it fails the accepted observed-front box convention; the hidden portion cannot be verified from these pixels. That does not justify changing accepted labels or treating hidden-corner output as forbidden.

`docs/annotation-boundaries-r33.md` explicitly illustrates this for088 row51: the accepted left boundary moved from749 to802 to remove a jaw-only strip, while the hidden left point stayed unavailable. It explicitly warns that broad historical predictions can then fail IoU without a model failing to propose the cassette. Current p56/p57 show that same type of mismatch. There are also ordinary geometry errors, repeated proposals, and clear background responses. All four mechanisms coexist.

## Implication for the smallest next intervention

Keep the accepted clipped-NMS result as the baseline. Do not adopt a broad confidence increase or lower NMS threshold simply because it reduces the unmatched count: six other-unmatched outputs are more confident than the currently matched prediction for their associated target. Five of those six pairs overlap each other at prediction IoU at least0.50, so greedy suppression can prefer the worse box and remove a useful match:

| Image | More-confident other output | Currently matched output |
|---|---:|---:|
| new_000008 | p29 | p34 |
| new_000009 | p52 | p58 |
| new_000011 | p60 | p65 |
| new_000056 | p20 | p21 |
| new_000089 | p51 | p53 |
| new_000089 | p55 | p59 |

The easiest supported work without training is a frozen-reference replay of any proposed output filter, retaining prediction-identity and missed-target checks. The associations in this audit must never be runtime filter inputs; they use the reference. A rack-region filter could address the outside-rack090 background if a general rack locator is independently available, but fixed masks drawn on these validation images would overfit this batch and would not address the fixture responses inside the rack area. No such ROI implementation is established by this audit.

The next training question, if postprocessing cannot pass those checks, is visible-front localization and grouping under the jaw, using existing eligible training examples and the accepted convention. The14 proposal-to-six-miss associations make that a more focused objective than blindly boosting recall. This audit does not prove a new training recipe will improve the model.

## Artifacts and provenance

- `review.json`: all64 individual cases, original pIDs, exact box/crop coordinates, source hashes, accepted-reference associations, exclusion overlap, physical confidence and uncertainty, and matched-output confidence comparisons.
- `mechanical.json`: reproducible scoring and nearest-reference geometry before manual classification.
- `context_01.png`–`context_04.png`: source-frame context for every image reviewed.
- `sheet_01.png`–`sheet_11.png`: all64 paired unmarked/annotated exact-coordinate crops. Crops use nearest-neighbor enlargement; they do not generate or alter scene evidence.
- `representative_cases.png`: six selected cards for quick review.
- `top_sequence.png`: original source pixels from087–090 showing the top fixture.
- `top_sequence_053_056.png`: original top-boundary context for the unresolved clipped cases.
- `prepare_triage.py`, `finish_triage.py`, `render_representative.py`: read-only assembly, audit and six-card rendering scripts. Each takes explicit `--root`, `--image-root` and `--output` arguments. Run them in that order. `finish_triage.py` checks the fixed cache/reference hashes before attaching the recorded manual judgments.
- `representative_cases.layout.json`: the six card bounds, identities, original crop coordinates and rendered sizes.

Example invocation, with paths supplied for the current checkout and original photographs:

```sh
python prepare_triage.py --root /path/to/dl --image-root /path/to/original/val --output /path/to/review
python finish_triage.py --root /path/to/dl --image-root /path/to/original/val --output /path/to/review
python render_representative.py --root /path/to/dl --image-root /path/to/original/val --output /path/to/review
```

Working cache SHA256: `d4b404997b891192839c6551fa4513fea49f2c51bbc7cfb9212185ace45ff0f2`.

R38 manifest SHA256: `0921081930f49340ff34290d5752be2d669f084f4d1518e2019060a5600c43b5`.

No repo, label, scoring threshold or model was modified. No training, inference or deployment experiment was performed. This is a source-pixel review of a development batch, not independent deployment validation.
