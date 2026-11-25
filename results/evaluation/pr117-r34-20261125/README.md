# PR117: fixed R34 comparison, one confidence probe, and R35 reference replay

The returned R34 candidate does not improve the primary frozen-666 recall. At the fixed confidence 0.13, it also loses one match on the current R34 review and has one fewer eligible corner pair within five pixels. PR35 remains selected. The later R35 scores in this directory use amended references and **unchanged model outputs**; they are not R35 training results.

## Verified provenance

- [PR117](https://github.com/Siuuuers/dl/pull/117) artifact revision: `bc018376733a0c17ea126a1cb19112314ca5eb33`.
- Exact published training source: `84b568f16f539b2189802d13726d528a0f21e237`.
- Protocol: `experiments/front-review-2026-11-25-r34.json`, SHA256 `05a23cbef0fab6a2b5be1e1b7c8aafc4a261ab32f13cde26d13d68bbff3e3277`.
- Reconstructed archive: `baomai-front-review-r34-results.zip`, 39,795,736 bytes, SHA256 `1dc96bbcc798773ae848307911858382054a6768cf9105a545e501cb9218462f`.
- Reference final checkpoint SHA256: `f472cdb8d239a3dedc314d646a841b9876f99f440b497d42f0f72ec2d1123fb9`.
- Candidate final checkpoint SHA256: `d1580dab164de3db58839dc0dbc54b903c5abba804e24322fdd51072f842d6c0`.

Both wrapper hashes, both raw-part hashes, the full archive hash and size, ZIP integrity, safe extraction paths, and all 15 archive members were verified. Both native run receipts and the paired comparison passed the repository verifier in an isolated checkout of the exact training source. The mutable R35 development checkout was not substituted for that source verification.

Both roles completed 30 epochs, 1,050 batches, 4,110 image presentations, and 78 optimizer steps, with final-epoch selection, exit code zero, and independently initialized epoch-173 weights. The recorded GPU runtime is NVIDIA L4, PyTorch 2.14.1+cu130, and Ultralytics 8.4.182. The arms share the recorded runtime, numerical recipe, and initial observed batch image order. Their accepted training difference is seven rows in six photographs: six box changes, one right-corner change, and one left-corner/visibility restoration. `archive-audit.json` contains the detailed identities and verification receipt.

Fresh inference for both PR117 checkpoints used the existing common CPU environment: Python 3.12.14, PyTorch 2.10.0+cpu, Ultralytics 8.4.182, FP32, and four threads. All 20 validation images, including the three empty scenes, are present. Original image hashes are checked. Fixed inference is size 1280, rectangular preprocessing, batch 1, native one-to-many output, NMS IoU 0.70, box confidence 0.13, max detections 300, and the repository's quantize=32 convention. Point confidence is 0.50; box matching IoU is 0.50.

The saved PR114 and PR35 caches have exactly the same recorded inference runtime and settings. Despite different checkpoint file hashes, **the PR117 R33-label reference produces exactly identical returned prediction records on all 20 images to the PR114 R33-label candidate**. This is an output-reproduction check, not a claim that the serialized checkpoint files are identical.

## Fixed confidence 0.13 results

Duplicate outputs are a subset of unmatched outputs. Pair counts below use all eligible two-locatable-corner objects as their denominator, including missed objects.

| Reference | Model | Misses | Unmatched | Duplicate subset | Pairs <=5 px | Pairs <=10 px |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Frozen 666 / 584 pairs | PR117 R33-label reference | 35 | 105 | 30 | 516 | 573 |
| Frozen 666 / 584 pairs | PR117 R34-label candidate | 35 | 105 | 29 | 515 | 573 |
| Frozen 666 / 584 pairs | PR114 R33-label candidate | 35 | 105 | 30 | 516 | 573 |
| Frozen 666 / 584 pairs | PR35 selected | 18 | 80 | 32 | 533 | 582 |
| Historical 663 / 584 pairs | PR117 R33-label reference | 32 | 105 | 30 | 516 | 573 |
| Historical 663 / 584 pairs | PR117 R34-label candidate | 31 | 104 | 29 | 515 | 573 |
| Historical 663 / 584 pairs | PR35 selected | 15 | 80 | 32 | 533 | 582 |
| R34 review / 661 fronts / 583 pairs | PR117 R33-label reference | 16 | 91 | 30 | 570 | 582 |
| R34 review / 661 fronts / 583 pairs | PR117 R34-label candidate | 17 | 92 | 30 | 569 | 582 |
| R34 review / 661 fronts / 583 pairs | PR35 selected | 45 | 112 | 15 | 546 | 569 |
| R35 review / 659 fronts / 583 pairs | PR117 R33-label reference | 14 | 91 | 29 | 570 | 582 |
| R35 review / 659 fronts / 583 pairs | PR117 R34-label candidate | 15 | 92 | 29 | 569 | 582 |
| R35 review / 659 fronts / 583 pairs | PR35 selected | 43 | 112 | 15 | 546 | 569 |

`comparison.json` additionally includes the PR114 R31-label control and the preserved R33 reference, keeping every reference version separate.

### Exact paired changes

On R34, the candidate newly misses `new_000010:31` and recovers no reference row. The control's assigned output was p37, confidence **0.1310166866**, IoU **0.5923092809**. The candidate's nearest surviving output at confidence 0.13 is p36, IoU **0.3723993426**. The small distance of the control confidence from the cutoff makes this a useful calibration diagnostic, but the separately tested 0.10 threshold does not recover this row.

All 17 candidate R34 misses are partial fronts: four have no locatable corner and thirteen have one. All 583 two-corner fronts match; only 61 of 78 partial fronts match. Nine missed fronts have no output with IoU >=0.10; eight have some overlap below the matching threshold. These are geometric triage categories, not physical diagnoses.

The matched corner-error catalogue has 20 rows, up from the control's 18. The added rows are `new_000010:29` and `new_000056:19`. Each has an available reference corner more than five pixels from its returned point. Five and ten pixels are review radii, not calibrated manipulation tolerances.

Frozen 666 retains exactly the same missed-row identities in both PR117 arms. Historical 663 recovers `new_000010:14`; this historical diagnostic gain does not replace the unchanged primary frozen-666 result.

## One bounded exploratory confidence probe

The same PR117 candidate was inferred once more on all 20 images at box confidence **0.10**, retaining NMS 0.70 and every other inference setting. No NMS sweep, training change, or threshold selection was performed. The primary setting remains 0.13.

Filtering the 0.10 outputs back to confidence >=0.13 preserves **all 736 original output records exactly**, in their original order. The probe adds 16 outputs, for 752 total. All three empty scenes remain at zero outputs.

| Reference | Misses, 0.13 -> 0.10 | Unmatched, 0.13 -> 0.10 | Duplicates, 0.13 -> 0.10 | Corner pairs |
| --- | ---: | ---: | ---: | --- |
| Frozen 666 | 35 -> 34 | 105 -> 120 | 29 -> 32 | 515/573 unchanged |
| Historical 663 | 31 -> 31 | 104 -> 120 | 29 -> 32 | 515/573 unchanged |
| R34 review | 17 -> 15 | 92 -> 106 | 30 -> 31 | 569/582 unchanged |
| R35 review | 15 -> 13 | 92 -> 106 | 29 -> 30 | 569/582 unchanged |

The two additional R34/R35 matches are:

- `new_000010:34`: probe p39, confidence `0.1053079367`, IoU `0.7936313112`.
- `new_000090:39`: probe p64, confidence `0.1030991822`, IoU `0.8948391187`.

The two diagnostic matches cost 14 additional unmatched outputs: one duplicate and thirteen other unmatched. Frozen 666 gains one match for fifteen extra unmatched outputs; historical 663 gains none for sixteen extra unmatched outputs. `new_000010:31` remains missed. These measurements expose the tradeoff; they do not establish a deployment threshold or promote the candidate. Probe pIDs refer to their own cache. The PDF appendix includes both recovered crops and all 16 added outputs.

## R35 reference changes are accounted for separately

R35 review manifest SHA256: `1ec7e4826d3e14c61d83e5267fdb798255348858c834e64ec2d66347076f9019`.

`new_000009` R34 rows 56 and 57 map to **null** under the explicitly requested instance exclusions. Rows 1–55 retain their identities. Both removed rows were missed by every compared cache. Their removal lowers the candidate miss count from 17 to 15 while true positives remain 644: no retained object is newly detected. The reference has 659 fronts instead of 661, 1,224 visible-corner targets instead of 1,225, and the same 583 eligible two-corner pairs. Earlier interpretations of the deleted regions as cassette fragments remain preserved in R34. These are user-directed exclusions, not newly established physical duplicates.

One retained assignment changes after the `new_000009:51` box amendment:

- p52 IoU changes `0.6015687555 -> 0.4763225872`; it changes from matched to other-unmatched.
- p58 IoU changes `0.6165424616 -> 0.5459121278`; it changes from duplicate to matched.

The row remains matched and total unmatched stays 92. Its composition changes from 30 duplicates + 62 other unmatched to **29 duplicates + 63 other unmatched**. The scorer remains the same confidence-ordered greedy matcher.

The amended `new_000010:24` right corner moves from x911 to x909 at y491. Its unchanged assigned p26 error changes `1.5538093392 -> 3.5345348695` pixels, remaining within five pixels. The assigned box IoU changes `0.6740062249 -> 0.6673328959`. The other two development box amendments retain their assigned predictions. The 20 corner-error identities and 569/582 corner-pair counts remain unchanged.

`r35-reference-delta.json` records row lineage, all six development reference operations, old/new assignments and distances, deleted misses, and output category changes for all compared models and the confidence probe. This preserves the distinction between a changed reference and a changed prediction.

## Definitions and limits

The preserved scorer processes predictions in descending confidence, assigning each to an unmatched same-class reference with IoU >=0.50. A missed front is an unassigned reference. An unmatched output is an unassigned prediction. A duplicate is an unmatched prediction overlapping an already matched reference at IoU >=0.50. Thus the R34 candidate's **92 unmatched = 30 duplicate + 62 other unmatched**. Duplicate counts must not be added a second time.

Other-unmatched outputs are not automatically hallucinations. Poor localization, excluded surfaces, omitted or uncertain reference instances, and other physical interpretations can produce this category. When all IoUs are zero, the display does not invent a nearest row/output from an arbitrary tie.

A maximum-cardinality diagnostic uses the same class/IoU eligibility graph. It finds no recoverable additional matches for the preserved model/reference combinations or the confidence probe. This is a count diagnostic; it does not prove that every equally sized assignment names the correct physical instance. The reported greedy scoring policy is unchanged.

Reference visibility 2 supplies a locatable coordinate target. Visibility 0 does not; predicted points on unavailable corners are permitted but cannot be scored against the placeholder origin. Stored image x grows right and y grows down. Top is ymin, bottom is ymax. Row identities are specific to their stated reference version; pIDs are one-based indices in their stated prediction cache.

These references have been repeatedly reviewed during development. They do not establish reliability on new acquisition sessions. Frozen 666 remains the primary historical comparison; historical 663 and the successive development reviews remain separate. The R35 notebook's accepted training delta and fixed native recipe should be evaluated without folding this exploratory confidence change into the same training experiment.

## Files and reproduction

- `predictions/reference.jsonl` and `candidate.jsonl`: fresh fixed-confidence PR117 caches.
- `predictions/candidate-conf010.jsonl`: the single exploratory probe cache.
- `comparison.json`: five checkpoint records across separately preserved references, per-image counts, partial-front groups, exact miss identities, candidate review catalogues, and matching diagnostics.
- `confidence-probe.json`: primary/probe scores, recovered identities, all-image output consistency, and empty-scene accounting.
- `r35-reference-delta.json`: stable-row reference-only effects and exclusions.
- `archive-audit.json`: byte validation and exact-source native-run verification.
- `pdf-inventory.json` and `pdf-qa.json`: complete crop/page inventory and rendered layout checks.

The PR114/PR35 caches are reused by their tracked paths under `results/evaluation/pr114-r33-20261125/predictions/`, with hashes and runtime equality checks. They are not silently replaced by a differently configured inference.

Replay all saved caches, including the probe and R35 reference changes, without loading a model:

```bash
python3 scripts/score_pr117_review.py
```

Render the review PDF after materializing the original images through Git LFS:

```bash
python3 scripts/render_pr117_failures.py --output /tmp/pr117-r34-misses-and-errors.pdf
```

The PDF has 38 primary R34 pages, five separately numbered exploratory-confidence pages, and three separately numbered R35 reference pages. The primary catalogue includes every one of the 17 missed fronts, 20 matched corner-error rows, and 92 unmatched candidate outputs. All 20 image counts, including the three empty scenes, are shown. PDFs, original images, and model weights are not committed in this directory.
