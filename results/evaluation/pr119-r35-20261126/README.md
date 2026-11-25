# PR119: R35 fixed comparison and separate R36 reference replay

The R35 candidate retains the same 15 missed fronts on the preserved R35 review, adds one unmatched output, and has one fewer matched corner-error row. On the primary frozen-666 reference it adds one missed front and two unmatched outputs. PR35 remains selected. The R36 records here rescore unchanged predictions against two amended development boxes; they are **not R36 training results**.

## Verified identities for reuse

| Item | Identity |
| --- | --- |
| Results PR | [PR119](https://github.com/Siuuuers/dl/pull/119) |
| Artifact revision | `3950115ba518c8c2b4e35e5a8511f0feb9c3ed29` |
| Exact immutable training source | `bfe57e5ceb95ffa1d94ee56880010052ed84c86d` |
| Notebook delivery revision | `635e6a75d573f87571eae99b3631b95762ec4567` |
| Protocol | `experiments/front-review-2026-11-25-r35.json` |
| Protocol SHA256 | `b74cf472f42380f7f8c373d9ad9fda1e88560d16ca2fb64d079741d403c5a80d` |
| Full archive | `baomai-front-review-r35-results.zip`, **39,795,858 bytes** |
| Archive SHA256 | `2a90c3c038caa42ebbc331376671dcf4b3022d690707bd6f7906505703df1c30` |
| GPU results manifest SHA256 | `c616f7472e428965235b379e4b0110a8b9ec1433a15046ccbe2291f73fbdc6a9` |
| R34-label reference checkpoint | `5acba132cc4e003c728ea8786be7dae807b991f5205170ca32c70e3a006f32f5` |
| R35-label candidate checkpoint | `57b368387281c8ef499004f1ee49f01cefd2f4cbf1cbaa5444bee63b1115faeb` |
| Original epoch-173 initializer | `e0aab1b377491700318e0b14cd4c558ec3c1684087a13b7dde2298ebdbf2847e` |
| R35 review manifest SHA256 | `1ec7e4826d3e14c61d83e5267fdb798255348858c834e64ec2d66347076f9019` |
| R36 review manifest SHA256 | `d0cdb9d56c043175ca12e538951cb28b1a8cd897039cce7a971e8f046407d416` |

The two stored ZIP wrappers are under `results/gpu/baomai-front-review-r35-l4-001/` at the artifact revision. Verify `chunks.sha256`, unzip each wrapper to its raw part, verify `parts.sha256`, then concatenate part00 followed by part01. Check the reconstructed archive against its declared size/hash before extracting it. Check ZIP integrity, safe relative paths, and every member listed by `gpu-results-manifest.json`. This replay completed all those checks: **15 members**, consisting of the manifest plus 14 declared files.

Both native runs were verified in an isolated checkout of the exact source above, before inference. The mutable R36 checkout was not substituted for receipt/source verification. Both roles completed **30 epochs, 1,050 batches, 4,110 image presentations, and 78 optimizer steps**, with exit code zero, final-epoch checkpoint selection, the same recorded runtime and numerical recipe, and the same initial observed batch image order. Both independently start from the original epoch-173 initializer.

The recipe remains `reviewed-incremental-fronts-seed46-v1`: seed 46, native Pose26 loss, partial-box factor 1, batch 4, size 1280, FP32, AdamW lr0=0.0001. The accepted training difference is **five box-only rows in three photographs**, with classes and visibility unchanged. The GPU receipt records NVIDIA L4, PyTorch 2.14.1+cu130, and Ultralytics 8.4.182. Development-label amendments and the earlier R35 reference exclusions do not enter training or trainer validation.

`archive-audit.json` preserves wrapper/part/member identities, checkpoint manifests, runtime/source fingerprints, and the exact-source native-verification result. No GPU training was performed for this evaluation.

## Common inference and prior-cache comparison

Fresh inference for both PR119 checkpoints uses Python 3.12.14, PyTorch 2.10.0+cpu, torchvision 0.25.0+cpu, Ultralytics 8.4.182, NumPy 2.3.5, Pillow 13.0.0, and OpenCV 5.0.0.93. The restored environment has the same recorded runtime headers as the reused PR117 and PR35 caches. All 157 original images were materialized for source verification; all 20 validation images, including three empty scenes, are present in each prediction cache.

Fixed inference remains rectangular size 1280, batch 1, FP32, four CPU threads, native one-to-many output, NMS IoU **0.70**, box confidence **0.13**, max detections 300, and the repository's quantize=32 convention. Scoring uses box IoU **0.50**, point confidence **0.50**, and five-/ten-pixel review radii. There was no new confidence or NMS sweep.

The new PR119 R34-label reference reproduces the prior PR117 R34-label candidate's aggregate results on every preserved reference. Prediction records on the 17 nonempty images are not bit-identical: maximum indexwise box-coordinate difference is **0.000244140625 px**, point-coordinate difference **0.0001220703125 px**, box-confidence difference about **1.10e-6**, and point-confidence difference about **2.38e-7**. Every image has the same output count. The three empty records are exactly equal. These are measured numerical differences; identical aggregates are not described as identical serialized outputs.

## Fixed-confidence results

Duplicate outputs are already included in unmatched outputs. Corner-pair denominators include all reference objects with two locatable corners, including eligible missed objects.

| Reference | Model | Misses | Unmatched | Duplicate subset | Pairs <=5 px | Pairs <=10 px |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Frozen 666 / 584 pairs | PR119 R34-label reference | 35 | 105 | 29 | 515 | 573 |
| Frozen 666 / 584 pairs | PR119 R35-label candidate | 36 | 107 | 29 | 517 | 573 |
| Frozen 666 / 584 pairs | PR117 R34-label candidate | 35 | 105 | 29 | 515 | 573 |
| Frozen 666 / 584 pairs | PR35 selected | 18 | 80 | 32 | 533 | 582 |
| Historical 663 / 584 pairs | PR119 R34-label reference | 31 | 104 | 29 | 515 | 573 |
| Historical 663 / 584 pairs | PR119 R35-label candidate | 32 | 106 | 29 | 517 | 573 |
| Historical 663 / 584 pairs | PR35 selected | 15 | 80 | 32 | 533 | 582 |
| R35 review / 659 fronts / 583 pairs | PR119 R34-label reference | 15 | 92 | 29 | 569 | 582 |
| R35 review / 659 fronts / 583 pairs | PR119 R35-label candidate | 15 | 93 | 31 | 570 | 582 |
| R35 review / 659 fronts / 583 pairs | PR117 R34-label candidate | 15 | 92 | 29 | 569 | 582 |
| R35 review / 659 fronts / 583 pairs | PR35 selected | 43 | 112 | 15 | 546 | 569 |

The full JSON additionally includes the PR117 R33-label reference and the preserved R34 development review. Every reference is scored separately; the current reviewed reference does not replace frozen 666 or historical 663.

### Current missed and corner-error identities

Both PR119 arms miss exactly the same 15 R35 fronts:

- `new_000008`: rows 33, 34.
- `new_000009`: rows 52, 55.
- `new_000010`: rows 31, 34.
- `new_000011`: rows 55, 56, 57, 60.
- `new_000053`: row 1.
- `new_000086`: row 57.
- `new_000088`: rows 51, 57.
- `new_000090`: row 39.

All 15 are partial fronts: three have no locatable corner and twelve have one. The candidate matches all **583/583 two-corner fronts**, but only **61/76 partial fronts**. Seven missed fronts have no output with IoU >=0.10; eight have some overlap below the matching threshold. These are geometric triage groups, not physical adjudications.

The candidate has **19 matched corner-error rows**, compared with the reference's 20. The sole row removed from this catalogue is `new_000056:19`: its returned left-corner distance changes **5.0818494252 -> 4.1045640871 px**, crossing inside the unchanged five-pixel radius. No new corner-error identity appears. Five pixels is a review radius, not a calibrated physical manipulation tolerance.

On frozen 666 and historical 663, the candidate additionally misses `new_000089:43`. The reference assigns p55 at IoU **0.5062509074**. The candidate's closest overlapping output is p54 at IoU **0.4980481750**, with confidence **0.5587420464**. This is a crossing of the fixed IoU 0.50 matching gate; the candidate still returns an overlapping detection. The PDF shows this case alongside the corner-radius crossing, using original crops and unchanged cutoffs.

## Separate R36 reference replay

R36 changes two development boxes and no point targets:

| Reference row | Amendment | Same candidate output | R35 IoU | R36 IoU |
| --- | --- | --- | ---: | ---: |
| `new_000010:24` | xmax 910 -> 913 | p26 | 0.6669966728 | 0.6870065729 |
| `new_000057:1` | xmin 777.00011968 -> 782 | p3 | 0.8949169767 | 0.9125143709 |

All five compared model records preserve their assigned pairs, output categories, detection counts, corner metrics, and partial-front groups under R36. The candidate remains at **15 misses, 93 unmatched = 31 duplicates + 62 other unmatched, 19 corner-error rows, and 570/582 successful corner pairs**. R36 still contains 659 validation fronts, 1,224 visible-corner targets, and 583 two-corner pairs.

The `new_000010:24` right point remains `(909,491)`; the unavailable left point remains unlabelled. Its candidate right-corner error stays `3.5459647499` pixels. Both `new_000057:1` points are unchanged, with candidate errors `2.2683707451` and `1.9617296247` pixels. Higher box IoU here comes from revised reference boundaries, not model learning. The earlier R35 user-directed exclusions remain excluded and are not reported as newly recovered detections.

`r36-reference-delta.json` records old/new references, row lineage, assignments, IoUs, and point distances for every compared cache. The final PDF page provides both before/after crops. No R36 model has been trained in this evaluation.

## Category definitions and limits

The preserved scorer processes predictions in descending confidence, assigning each to an unmatched same-class reference at IoU >=0.50. A missed front is an unassigned reference. An unmatched output is an unassigned prediction. A duplicate is an unmatched prediction overlapping an already matched reference at IoU >=0.50. Therefore **93 unmatched = 31 duplicate + 62 other unmatched** for the R35/R36 candidate. Duplicate counts must not be added a second time.

Other-unmatched outputs are not automatically hallucinations: misplaced boxes, excluded surfaces, and omitted or uncertain reference instances can also produce this category. When all IoUs are zero, the report leaves the nearest row/output unset rather than choosing an arbitrary zero-overlap tie.

A maximum-cardinality check of the same class/IoU graph finds **zero recoverable additional matches** for every model/image/reference combination. It does not establish the physical correctness of every equally sized assignment. The official confidence-ordered scorer remains unchanged.

Reference visibility 2 supplies a locatable coordinate target. Visibility 0 does not; a predicted unavailable corner is permitted but is not scored against the placeholder origin. Stored image x grows right and y grows down. Rows are tied to the stated reference snapshot; pIDs are one-based indices in the stated checkpoint cache.

These repeatedly reviewed development images do not establish reliability on new acquisition sessions. The small changes in counts are useful diagnostics, and the threshold illustrations show why discrete score changes need their underlying geometry inspected. The current result does not justify changing thresholds, the training recipe, or model selection.

## Reproduction and deliverables

- `predictions/reference.jsonl` and `candidate.jsonl`: fresh PR119 fixed-confidence caches, with original output order, checkpoint identities, runtime headers, and image hashes.
- `comparison.json`: five checkpoints across separately preserved references, all per-image counts, miss identities, candidate error catalogues, corner-identity changes, control numerical comparison, and matching diagnostics.
- `r36-reference-delta.json`: reference-only row effects and unchanged assignments/categories.
- `archive-audit.json`: archive checks, exact-source native verification, checkpoint manifests, and source/runtime fingerprints.
- `pdf-inventory.json`: every primary crop, source-image hash, row/pID, category, page, threshold illustration, and R36 reference crop.
- `pdf-qa.json`: final PDF hash, page count, exhaustive catalogue counts, and rendered layout checks.

PR117 caches are reused from `results/evaluation/pr117-r34-20261125/predictions/`; the selected PR35 cache is reused from `results/evaluation/pr114-r33-20261125/predictions/pr35.jsonl`. Their hashes and runtime headers are checked. No prior cache is overwritten.

Replay scoring without loading a model:

```bash
python3 scripts/score_pr119_review.py
```

Render the cropped report after materializing original images through Git LFS:

```bash
python3 scripts/render_pr119_failures.py --output /tmp/pr119-r35-misses-and-errors.pdf
```

The report has **38 pages**: 36 primary R35 pages, one paired-threshold illustration, and one R36 reference-only page. Its primary catalogue covers all 15 misses, all 19 corner-error rows, and all 93 unmatched outputs. All 20 validation image counts, including the three empty scenes, are shown. PDFs, images, and weights are not committed in this evaluation directory.
