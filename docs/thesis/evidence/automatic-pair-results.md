# R55 automatic pair-output diagnostic

Automatic consumption makes the quality of each emitted coordinate pair important. This analysis asks whether the existing endpoint-score screen alone establishes a well-supported two-corner output. It is not a new model, deployed gate or independent evaluation.

Only saved postclip predictions are filtered. Matching is recomputed among accepted predictions against every reference front. A reference-supported usable pair must match a reference with both corners locatable and have both errors within the stated original-pixel tolerance. Unmatched accepted predictions and matches to unavailable pairs stay in the acceptance denominator; their lack of support is not a claim of proven physical error.

The original eight model/reference baseline tables reproduce exactly before this new diagnostic.

## frozen666

| Model / box cutoff | Corner cutoff | Accepted pairs | Reference-supported ≤5px | Reference-supported ≤10px | Unmatched accepted | Unavailable reference pair | Matched localization >5px | Supported-pair fraction ≤5px | Eligible-pair recall ≤5px |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| reference_box0.13 | 0.50 | 712 | 511 | 575 | 84 | 51 | 66 | 71.77% | 87.50% |
| reference_box0.13 | 0.70 | 683 | 511 | 575 | 61 | 45 | 66 | 74.82% | 87.50% |
| reference_box0.13 | 0.80 | 654 | 511 | 575 | 36 | 41 | 66 | 78.13% | 87.50% |
| reference_box0.13 | 0.90 | 512 | 422 | 477 | 13 | 20 | 57 | 82.42% | 72.26% |
| reference_box0.13 | 0.95 | 110 | 87 | 104 | 5 | 1 | 17 | 79.09% | 14.90% |
| reference_box0.13 | 0.99 | 0 | 0 | 0 | 0 | 0 | 0 | N/A | 0.00% |
| candidate_box0.13 | 0.50 | 738 | 511 | 575 | 110 | 51 | 66 | 69.24% | 87.50% |
| candidate_box0.13 | 0.70 | 696 | 511 | 575 | 74 | 45 | 66 | 73.42% | 87.50% |
| candidate_box0.13 | 0.80 | 662 | 511 | 575 | 43 | 42 | 66 | 77.19% | 87.50% |
| candidate_box0.13 | 0.90 | 515 | 424 | 480 | 14 | 19 | 58 | 82.33% | 72.60% |
| candidate_box0.13 | 0.95 | 116 | 92 | 110 | 5 | 1 | 18 | 79.31% | 15.75% |
| candidate_box0.13 | 0.99 | 0 | 0 | 0 | 0 | 0 | 0 | N/A | 0.00% |
| candidate_box0.2 | 0.50 | 709 | 511 | 575 | 81 | 51 | 66 | 72.07% | 87.50% |
| candidate_box0.2 | 0.70 | 678 | 511 | 575 | 56 | 45 | 66 | 75.37% | 87.50% |
| candidate_box0.2 | 0.80 | 650 | 511 | 575 | 32 | 41 | 66 | 78.62% | 87.50% |
| candidate_box0.2 | 0.90 | 514 | 424 | 480 | 13 | 19 | 58 | 82.49% | 72.60% |
| candidate_box0.2 | 0.95 | 116 | 92 | 110 | 5 | 1 | 18 | 79.31% | 15.75% |
| candidate_box0.2 | 0.99 | 0 | 0 | 0 | 0 | 0 | 0 | N/A | 0.00% |

## r50

| Model / box cutoff | Corner cutoff | Accepted pairs | Reference-supported ≤5px | Reference-supported ≤10px | Unmatched accepted | Unavailable reference pair | Matched localization >5px | Supported-pair fraction ≤5px | Eligible-pair recall ≤5px |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| reference_box0.13 | 0.50 | 712 | 566 | 582 | 78 | 51 | 17 | 79.49% | 97.08% |
| reference_box0.13 | 0.70 | 683 | 566 | 582 | 56 | 44 | 17 | 82.87% | 97.08% |
| reference_box0.13 | 0.80 | 654 | 566 | 582 | 36 | 35 | 17 | 86.54% | 97.08% |
| reference_box0.13 | 0.90 | 512 | 468 | 483 | 9 | 19 | 16 | 91.41% | 80.27% |
| reference_box0.13 | 0.95 | 110 | 104 | 109 | 0 | 1 | 5 | 94.55% | 17.84% |
| reference_box0.13 | 0.99 | 0 | 0 | 0 | 0 | 0 | 0 | N/A | 0.00% |
| candidate_box0.13 | 0.50 | 738 | 567 | 582 | 101 | 54 | 16 | 76.83% | 97.26% |
| candidate_box0.13 | 0.70 | 696 | 567 | 582 | 68 | 45 | 16 | 81.47% | 97.26% |
| candidate_box0.13 | 0.80 | 662 | 567 | 582 | 43 | 36 | 16 | 85.65% | 97.26% |
| candidate_box0.13 | 0.90 | 515 | 472 | 486 | 10 | 18 | 15 | 91.65% | 80.96% |
| candidate_box0.13 | 0.95 | 116 | 111 | 115 | 0 | 1 | 4 | 95.69% | 19.04% |
| candidate_box0.13 | 0.99 | 0 | 0 | 0 | 0 | 0 | 0 | N/A | 0.00% |
| candidate_box0.2 | 0.50 | 709 | 567 | 582 | 73 | 53 | 16 | 79.97% | 97.26% |
| candidate_box0.2 | 0.70 | 678 | 567 | 582 | 50 | 45 | 16 | 83.63% | 97.26% |
| candidate_box0.2 | 0.80 | 650 | 567 | 582 | 32 | 35 | 16 | 87.23% | 97.26% |
| candidate_box0.2 | 0.90 | 514 | 472 | 486 | 9 | 18 | 15 | 91.83% | 80.96% |
| candidate_box0.2 | 0.95 | 116 | 111 | 115 | 0 | 1 | 4 | 95.69% | 19.04% |
| candidate_box0.2 | 0.99 | 0 | 0 | 0 | 0 | 0 | 0 | N/A | 0.00% |

## Interpretation limits

- Endpoint confidence is being tested as an acceptance signal, not assumed to encode physical visibility or error.
- A reference-unavailable pair has no two-point ground truth for localization; it is unsupported by the chosen reference, not proven physically wrong.
- Supported-pair fractions combine reference-unverified outputs and measured localization failures in their complement. They are not calibrated precision or true automated-use accuracy.
- The four reference versions reuse the same 20 development photographs. The 5/10px tolerances are diagnostic, not a downstream acceptance requirement.
- A cutoff accepting zero pairs has no defined accepted-pair precision. It must not be described as 100% reliable.
- The candidate .20 box cutoff is inherited from the R54 exploratory analysis, not newly validated for automatic consumption.
- No runtime, label, model or deployment setting changes in this analysis.

## Reproduce

Extract `r55-automatic-output-audit.zip` beside this file, then use the capsule's verified inputs and scorer from its `r55-analysis` folder:

    python -B analyze_automatic_pairs.py --inputs ../r54-analysis/inputs.json --scorer ../r54-analysis/pose_scoring.py --output replay
