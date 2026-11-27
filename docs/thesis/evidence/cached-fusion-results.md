# R53 fixed cached output fusion diagnostic

This post-hoc replay tests complementarity and fixed, score-ordered NMS fusion of the returned views. It performs no inference or training, changes no defaults, and does not establish a deployment result.

The raw records already passed each view's native NMS and were clipped/mapped into original image coordinates. They are pooled before the extra postclip pass, with confidence >0.13, fixed IoU >0.70 suppression, maximum 300 and no coordinate averaging. A final original postclip pass is verified idempotent. Removed native proposals cannot be recovered.

## Immutable inputs and reproduction

The existing R53 scoring and input protocol is pinned to [source `7b36e23dbbc24769f92a88cb93ae44acfc0406c5`](https://github.com/Siuuuers/dl/tree/7b36e23dbbc24769f92a88cb93ae44acfc0406c5). The replay verifies source-helper hashes, all four reference manifests, all 80 reference label files, both archive hashes and their member inventories. The JSON records every input reference path and SHA-256.

- [PR149 clahe archive](https://github.com/Siuuuers/dl/blob/343f8d759ccaba671fa38ad4e9674d75a1bb5ba7/results/l4/baomai-clahe-inference-r53-l4-001/baomai-clahe-inference-r53-results.zip), commit `343f8d759ccaba671fa38ad4e9674d75a1bb5ba7`, 2,343,077 bytes, SHA-256 `1e1dde4528729deced36a7943f1a3fc68e4e4993e05c0289e10fcf03f2cfa56d`; [download](https://raw.githubusercontent.com/Siuuuers/dl/343f8d759ccaba671fa38ad4e9674d75a1bb5ba7/results/l4/baomai-clahe-inference-r53-l4-001/baomai-clahe-inference-r53-results.zip).
- [PR150 scale archive](https://github.com/Siuuuers/dl/blob/e8a70f67cba1bf99e2d8a7bce9c93b5d4d7996b2/results/a100/baomai-scale-inference-r53-a100-001/baomai-scale-inference-r53-results.zip), commit `e8a70f67cba1bf99e2d8a7bce9c93b5d4d7996b2`, 2,328,146 bytes, SHA-256 `b1638b1eaf5e3877a77efb840c342bd1dfe7783ca5951041ed37b692df59abca`; [download](https://raw.githubusercontent.com/Siuuuers/dl/e8a70f67cba1bf99e2d8a7bce9c93b5d4d7996b2/results/a100/baomai-scale-inference-r53-a100-001/baomai-scale-inference-r53-results.zip).

Checkpoint identity: `222bf3c91a651183047a10eb1a50e16ae8fe999f15372afe005bfca2d0b157f9` (the PR137 candidate used by both R53 runs). The saved R53 returns contain the control records required here; PR146, weights, source photographs, a GPU, and model inference are not needed for this CPU replay. This diagnostic does not independently rerun the archived model.

From this repository checkout with Python 3, after saving the two linked ZIPs outside the checkout:

```bash
python -B scripts/analyze_r53_cached_fusion.py \
  --clahe-archive /path/to/baomai-clahe-inference-r53-results.zip \
  --scale-archive /path/to/baomai-scale-inference-r53-results.zip \
  --output /path/to/replay-results
```

The default `--root` is the checkout containing this script. If its old scoring helpers or reference files have changed, pass `--root /path/to/pinned-r53-source` from the immutable source commit above; do not substitute scoring code. This path uses the Python standard library and project helpers. Do not run with `python -O`, which disables validation assertions.

The default results are compact. Add `--include-image-details` to include all per-image count rows, and `--write-fused-predictions` to regenerate the complete fused arrays with original source-view and source-index provenance. The compact JSON retains all gained/lost identities and empty-scene counts. No fused arrays need to be committed.

## frozen666

| Output view | Matched | Missed | Unmatched | Duplicate subset | Pair ≤5 px | Pair ≤10 px |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| identity | 632 | 34 | 91 | 17 | 511/584 | 575/584 |
| clahe | 633 | 33 | 92 | 19 | 506/584 | 573/584 |
| scale1600 | 633 | 33 | 82 | 20 | 503/584 | 566/584 |
| identity_plus_clahe | 633 | 33 | 111 | 25 | 510/584 | 575/584 |
| identity_plus_scale1600 | 633 | 33 | 135 | 34 | 506/584 | 570/584 |

### Retrospective success unions

These use reference knowledge and separate per-view assignments. They are oracle ceilings for choosing among those already successful rows, not attainable output rules or universal bounds on every possible fusion assignment.

- identity+clahe: matched union 635; pair ≤5 union 518; pair ≤10 union 577.
- identity+scale1600: matched union 636; pair ≤5 union 528; pair ≤10 union 576.
- identity+clahe+scale1600: matched union 639; pair ≤5 union 534; pair ≤10 union 578.

### Pair screen on standalone scale outputs

| View | Endpoint cutoff | Offered | Supported ≤5 | Supported fraction | Eligible recall | Unmatched | Incomplete reference pair | Measured >5 px |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| identity | 0.50 | 712 | 511 | 71.7697% | 87.5000% | 84 | 51 | 66 |
| identity | 0.80 | 654 | 511 | 78.1346% | 87.5000% | 36 | 41 | 66 |
| scale1600 | 0.50 | 704 | 503 | 71.4489% | 86.1301% | 74 | 53 | 74 |
| scale1600 | 0.80 | 672 | 503 | 74.8512% | 86.1301% | 50 | 45 | 74 |

## historical663

| Output view | Matched | Missed | Unmatched | Duplicate subset | Pair ≤5 px | Pair ≤10 px |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| identity | 633 | 30 | 90 | 17 | 511/584 | 575/584 |
| clahe | 632 | 31 | 93 | 18 | 506/584 | 573/584 |
| scale1600 | 634 | 29 | 81 | 20 | 503/584 | 566/584 |
| identity_plus_clahe | 633 | 30 | 111 | 24 | 510/584 | 575/584 |
| identity_plus_scale1600 | 634 | 29 | 134 | 34 | 506/584 | 570/584 |

### Retrospective success unions

These use reference knowledge and separate per-view assignments. They are oracle ceilings for choosing among those already successful rows, not attainable output rules or universal bounds on every possible fusion assignment.

- identity+clahe: matched union 635; pair ≤5 union 518; pair ≤10 union 577.
- identity+scale1600: matched union 637; pair ≤5 union 528; pair ≤10 union 576.
- identity+clahe+scale1600: matched union 639; pair ≤5 union 534; pair ≤10 union 578.

### Pair screen on standalone scale outputs

| View | Endpoint cutoff | Offered | Supported ≤5 | Supported fraction | Eligible recall | Unmatched | Incomplete reference pair | Measured >5 px |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| identity | 0.50 | 712 | 511 | 71.7697% | 87.5000% | 83 | 52 | 66 |
| identity | 0.80 | 654 | 511 | 78.1346% | 87.5000% | 36 | 41 | 66 |
| scale1600 | 0.50 | 704 | 503 | 71.4489% | 86.1301% | 73 | 54 | 74 |
| scale1600 | 0.80 | 672 | 503 | 74.8512% | 86.1301% | 50 | 45 | 74 |

## r38

| Output view | Matched | Missed | Unmatched | Duplicate subset | Pair ≤5 px | Pair ≤10 px |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| identity | 639 | 15 | 84 | 14 | 566/583 | 582/583 |
| clahe | 640 | 14 | 85 | 23 | 564/583 | 582/583 |
| scale1600 | 639 | 15 | 76 | 18 | 557/583 | 581/583 |
| identity_plus_clahe | 642 | 12 | 102 | 24 | 565/583 | 582/583 |
| identity_plus_scale1600 | 644 | 10 | 124 | 30 | 557/583 | 582/583 |

### Retrospective success unions

These use reference knowledge and separate per-view assignments. They are oracle ceilings for choosing among those already successful rows, not attainable output rules or universal bounds on every possible fusion assignment.

- identity+clahe: matched union 643; pair ≤5 union 570; pair ≤10 union 582.
- identity+scale1600: matched union 644; pair ≤5 union 574; pair ≤10 union 582.
- identity+clahe+scale1600: matched union 647; pair ≤5 union 576; pair ≤10 union 582.

### Pair screen on standalone scale outputs

| View | Endpoint cutoff | Offered | Supported ≤5 | Supported fraction | Eligible recall | Unmatched | Incomplete reference pair | Measured >5 px |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| identity | 0.50 | 712 | 566 | 79.4944% | 97.0840% | 78 | 51 | 17 |
| identity | 0.80 | 654 | 566 | 86.5443% | 97.0840% | 35 | 36 | 17 |
| scale1600 | 0.50 | 704 | 557 | 79.1193% | 95.5403% | 69 | 52 | 26 |
| scale1600 | 0.80 | 672 | 557 | 82.8869% | 95.5403% | 43 | 46 | 26 |

## r50

| Output view | Matched | Missed | Unmatched | Duplicate subset | Pair ≤5 px | Pair ≤10 px |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| identity | 639 | 15 | 84 | 13 | 566/583 | 582/583 |
| clahe | 640 | 14 | 85 | 22 | 564/583 | 582/583 |
| scale1600 | 638 | 16 | 77 | 17 | 557/583 | 581/583 |
| identity_plus_clahe | 642 | 12 | 102 | 23 | 565/583 | 582/583 |
| identity_plus_scale1600 | 644 | 10 | 124 | 28 | 557/583 | 582/583 |

### Retrospective success unions

These use reference knowledge and separate per-view assignments. They are oracle ceilings for choosing among those already successful rows, not attainable output rules or universal bounds on every possible fusion assignment.

- identity+clahe: matched union 643; pair ≤5 union 570; pair ≤10 union 582.
- identity+scale1600: matched union 644; pair ≤5 union 574; pair ≤10 union 582.
- identity+clahe+scale1600: matched union 647; pair ≤5 union 576; pair ≤10 union 582.

### Pair screen on standalone scale outputs

| View | Endpoint cutoff | Offered | Supported ≤5 | Supported fraction | Eligible recall | Unmatched | Incomplete reference pair | Measured >5 px |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| identity | 0.50 | 712 | 566 | 79.4944% | 97.0840% | 78 | 51 | 17 |
| identity | 0.80 | 654 | 566 | 86.5443% | 97.0840% | 36 | 35 | 17 |
| scale1600 | 0.50 | 704 | 557 | 79.1193% | 95.5403% | 70 | 51 | 26 |
| scale1600 | 0.80 | 672 | 557 | 82.8869% | 95.5403% | 45 | 44 | 26 |

## Verification and limits

All 32 original raw/postclip model-reference tables reproduce exactly. Identity controls have identical coordinates and discrete outcomes; maximum confidence drift is recorded in JSON. All new views were directly rescored, source pairs were preserved exactly, and pair-support categories partition each image. The four references share the same 20 repeatedly used development photographs. Thresholds were fixed for this diagnostic and were not optimized. Unmatched and unavailable-reference pairs are not all proven physical errors. Full gains/losses and empty-scene counts are in the JSON.
