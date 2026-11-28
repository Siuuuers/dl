# Box geometry diagnostic

Saved predictions and all three annotation references are unchanged. These are descriptive development-data measurements.

| Reference | Method | Missed | Near-overlap misses | Candidate width >1.5x | Candidate height >1.5x |
| --- | --- | ---: | ---: | ---: | ---: |
| supplied664 | original | 39 | 20 | 1 | 12 |
| supplied664 | proposed | 46 | 29 | 1 | 16 |
| frozen666 | original | 26 | 6 | 2 | 3 |
| frozen666 | proposed | 34 | 17 | 0 | 7 |
| r50 | original | 34 | 19 | 15 | 4 |
| r50 | proposed | 18 | 10 | 8 | 1 |

## Reviewed-reference geometry of the retained method

| Group | Fronts | Median width (px) | Median height (px) | Median square-root area (px) |
| --- | ---: | ---: | ---: | ---: |
| all | 654 | 92.000 | 28.995 | 50.927 |
| complete | 583 | 93.120 | 29.000 | 51.846 |
| incomplete_matched | 53 | 81.000 | 24.000 | 42.790 |
| incomplete_missed | 18 | 35.500 | 23.005 | 28.853 |
| incomplete_near_overlap_missed | 10 | 31.500 | 18.500 | 25.229 |

## The ten near-overlap misses under the reviewed reference

| Case | Reference width x height (px) | Max IoU | Width ratio | Height ratio | Signed left / top / right / bottom errors (px) | W2 distance (px) |
| --- | --- | ---: | ---: | ---: | --- | ---: |
| r50/new_000008/R001 | 217.50 x 10.00 | 0.3906 | 0.992 | 2.490 | +5.48 / +0.00 / +3.76 / +14.90 | 11.538 |
| r50/new_000009/R051 | 21.00 x 16.00 | 0.4906 | 1.308 | 0.930 | -5.59 / +4.33 / +0.87 / +3.21 | 5.526 |
| r50/new_000011/R055 | 31.00 x 18.00 | 0.4333 | 2.080 | 1.109 | -1.51 / -0.64 / +31.99 / +1.33 | 22.666 |
| r50/new_000011/R056 | 20.00 x 16.00 | 0.3723 | 2.192 | 1.220 | -1.95 / +0.05 / +21.89 / +3.57 | 15.744 |
| r50/new_000011/R057 | 15.50 x 12.00 | 0.3065 | 2.300 | 1.320 | -5.18 / +0.64 / +14.96 / +4.48 | 11.644 |
| r50/new_000011/R059 | 32.00 x 24.00 | 0.4726 | 1.777 | 1.183 | -25.01 / -1.37 / -0.15 / +3.02 | 17.839 |
| r50/new_000088/R051 | 28.50 x 23.01 | 0.4293 | 2.126 | 1.094 | -29.15 / +0.02 / +2.95 / +2.19 | 20.775 |
| r50/new_000089/R043 | 32.50 x 19.00 | 0.4689 | 1.813 | 1.176 | -0.76 / -0.70 / +25.68 / +2.65 | 18.269 |
| r50/new_000089/R048 | 46.00 x 32.00 | 0.4951 | 1.838 | 1.099 | -36.14 / -1.28 / +2.43 / +1.88 | 25.662 |
| r50/new_000090/R039 | 51.00 x 21.00 | 0.4937 | 1.517 | 1.335 | -0.57 / -5.05 / +25.79 / +1.99 | 18.642 |

## Interpretation

A near-overlap miss is an unmatched reference whose highest retained-prediction IoU is at least 0.30 and below 0.50. The diagnostic does not change the official one-to-one assignment. It may reuse a prediction or select a box already assigned to a different reference; cases.json records that status.

The overlap-selected candidate is not verified as the same physical front. All-zero-overlap cases have no selected candidate. Width/height ratios and signed edge errors indicate what to inspect, not which label or prediction should be changed.

The pixel distance is the 2-Wasserstein distance between Gaussian box representations with mean (center_x, center_y) and diagonal covariance (width^2/4, height^2/4). No normalization constant is selected and no distance replaces IoU in evaluation. A narrow visible fragment is not necessarily a tiny full object.

These measurements motivate an audit of visible extent, occlusion ownership and training augmentation before another loss-function experiment. They do not establish the training-time cause or a model improvement.
