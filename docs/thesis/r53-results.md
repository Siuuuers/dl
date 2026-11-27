# Completed R53 contrast and scale results

27 November 2026, Asia/Hong_Kong.

**Both returns are reproducible; neither supports adopting the treatment as the default.** PR149 tests fixed Lab-L CLAHE at input size 1280. PR150 tests the unchanged image at size 1600 against size 1280. Both use the identical PR137 research checkpoint, not the retained PR35 checkpoint or the R51 candidate.

## Main comparison

All rows use postclip predictions, box confidence 0.13, endpoint confidence 0.50, native and postclip NMS IoU 0.70, and one-to-one matching IoU 0.50. Duplicates are included within unmatched outputs. Missing eligible fronts remain in the pair denominator.

| Reference and arm | Matched | Missed | Unmatched | Duplicates | Both corners within 5 px | Both within 10 px |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Frozen666, unchanged | 632 | 34 | 91 | 17 | 511/584 | 575/584 |
| Frozen666, CLAHE | 633 | 33 | 92 | 19 | 506/584 | 573/584 |
| Frozen666, 1600 | 633 | 33 | 82 | 20 | 503/584 | 566/584 |
| Historical663, unchanged | 633 | 30 | 90 | 17 | 511/584 | 575/584 |
| Historical663, CLAHE | 632 | 31 | 93 | 18 | 506/584 | 573/584 |
| Historical663, 1600 | 634 | 29 | 81 | 20 | 503/584 | 566/584 |
| R38, unchanged | 639 | 15 | 84 | 14 | 566/583 | 582/583 |
| R38, CLAHE | 640 | 14 | 85 | 23 | 564/583 | 582/583 |
| R38, 1600 | 639 | 15 | 76 | 18 | 557/583 | 581/583 |
| R50, unchanged | 639 | 15 | 84 | 13 | 566/583 | 582/583 |
| R50, CLAHE | 640 | 14 | 85 | 22 | 564/583 | 582/583 |
| R50, 1600 | 638 | 16 | 77 | 17 | 557/583 | 581/583 |

The controls in the two returns have identical coordinates and discrete outcomes; their largest score difference is approximately 2.38e−7. Each treatment is nevertheless evaluated against its own control. All four references describe the same 20 development photographs.

## Case-level interpretation

CLAHE recovers three primary targets and newly misses two. Size 1600 recovers four and newly misses three. Each newly misses one primary two-corner target. Five-pixel pair transitions are seven gains/twelve losses for CLAHE and seventeen gains/twenty-five losses for size 1600. All three reference-empty scenes remain empty.

The scale miss `Frozen666/new_000057/R012` retains a high-score prediction, but its best IoU falls from approximately 0.50516 to 0.48961. This is a matching-threshold crossing, not a complete absence of a network response. It illustrates why small front boxes need boundary review and why a confidence change cannot repair displaced geometry.

The larger input removes nine unmatched outputs on the primary reference but increases the duplicate subset by three and loses eight five-pixel pair successes. It is a tradeoff, not a joint quality improvement. These results concern the exact recipes and checkpoint tested; they do not establish that all contrast processing or every scale strategy is ineffective.

## Verification

The original R53 verifier independently reconstructed the archives and checked member CRC/hashes, pinned source, checkpoint, 20 original-image hashes, 80 label files for the four scored references, 20 additional R48 lineage labels, all raw/postclip reports, stable row identities, repeated outputs, and timing declarations. Both controls preserve every required discrete outcome. Maximum drift from the older archived baseline is approximately 0.000122 pixel per coordinate component and 0.00000116 per score. All three within-run repeats are identical.

The CLAHE run used an L4; the scale run used an A100. The control tensor is 1×3×736×1280. The scale tensor is 1×3×928×1600 after a 1.25 interpolation gain and vertical padding. All reported errors use the original 1280×720 coordinate frame. Software-path timing excludes acquisition and is not camera frame rate.

## Immutable sources

| Evidence | Identifier |
| --- | --- |
| [PR149](https://github.com/Siuuuers/dl/pull/149) | `343f8d759ccaba671fa38ad4e9674d75a1bb5ba7` |
| PR149 archive SHA-256 | `1e1dde4528729deced36a7943f1a3fc68e4e4993e05c0289e10fcf03f2cfa56d` |
| [PR150](https://github.com/Siuuuers/dl/pull/150) | `e8a70f67cba1bf99e2d8a7bce9c93b5d4d7996b2` |
| PR150 archive SHA-256 | `b1638b1eaf5e3877a77efb840c342bd1dfe7783ca5951041ed37b692df59abca` |
| R53 source | `7b36e23dbbc24769f92a88cb93ae44acfc0406c5` |
| Research checkpoint SHA-256 | `222bf3c91a651183047a10eb1a50e16ae8fe999f15372afe005bfca2d0b157f9` |
| R53 driver SHA-256 | `8c8a37c3dcaccde5c568e9df6f44651cdb05965ca5f5ed5b0f2978852387dcc1` |

The [original protocol](../inference-ablation-r53.md) records recipe and baseline tolerances. The [cached-fusion replay](evidence/cached-fusion-results.md) provides portable immutable archive URLs and the subsequent fixed combination analysis. The next proposed comparison is [explicit horizontal reflection](flip-experiment-r56.md), with a verified notebook bootstrap but no model result yet.
