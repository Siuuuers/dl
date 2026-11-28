# Independent verification of corner averaging

The independent implementation in [`verify_corner_average_independent.py`](../../../../scripts/verify_corner_average_independent.py) reproduces the saved coordinate predictions and aggregate results. Its output is [`independent-check.json`](independent-check.json), including individual gains, losses, remaining failures, and a subsequent reference-assignment audit.

## Procedure and result

The verifier constructs separate sorted lists of same-class box-overlap candidates in each direction. It accepts mutually best pairs at an inclusive overlap of 0.50, breaking ties by the lower partner index. It averages corresponding coordinates only when both predictions contain two finite, in-image corners with scores of at least 0.50. This calculation is completed before loading the reference annotations. The implementation does not import the experiment's averaging function.

Comparison with the saved experiment evidence passed for every one of the **723 output records**, with **666 coordinate pairs averaged** and **56 averaged associations having multiple eligible overlap candidates**. All original box records and corner scores remain unchanged. The original scorer reproduces identical box assignments, missed targets, extra outputs, and duplicate identities before and after averaging.

| Reference | Both corners within 5 px: original → average | Gains / losses | Both corners within 10 px: original → average | Gains / losses |
|---|---:|---:|---:|---:|
| Fixed 666 | 511 → 525 / 584 | 19 / 5 | 575 → 574 / 584 | 0 / 1 |
| Historical 663 | 511 → 525 / 584 | 19 / 5 | 575 → 574 / 584 | 0 / 1 |
| Reviewed R38 | 566 → 569 / 583 | 6 / 3 | 582 → 583 / 583 | 1 / 0 |
| Reviewed R50 | 566 → 569 / 583 | 6 / 3 | 582 → 583 / 583 | 1 / 0 |

The subsequent reference-assignment audit found no accepted association whose two predictions were assigned to different reference targets. All 583 reviewed complete-target associations retained the same target. This is a check on this development sample, not proof that the association rule is always correct.

The one fixed-reference 10 px loss is `new_000008/R002`: its maximum original corner error was 8.9965 px and its maximum averaged error was 10.0388 px. Averaging therefore improves the stricter aggregate count while losing one success under the looser criterion. The 14 remaining reviewed 5 px failures occur in five images, with nine in `new_000008`.

## Reproduce the independent comparison

From the repository root:

```bash
python -B scripts/verify_corner_average_independent.py \
  --archive /path/to/baomai-flip-inference-r56-results.zip \
  --output /path/to/independent-check.json \
  --compare-summary docs/thesis/evidence/corner-average/summary.json \
  --compare-predictions docs/thesis/evidence/corner-average/averaged-predictions.json.gz
```

The original reflection archive is verified against its exact expected SHA-256 before use. Reference files and scoring dependencies are checked against the returned source contract. The result JSON SHA-256 is `7ce23108ab39bfa860566a71f2c79d8da816f6bf9ab4a83de6fae6895711a052`; the portable independent script SHA-256 is `1fcf0da8d98c170c7831e2a4f4f067230d9544da6b8f0ea409d8f7052f43ef8d`.

This verification concerns the fixed equal-weight rule and its saved numerical results. The comparison uses the same 20 development images as the reflection experiment. Original point scores retain the existing comparison cohorts and do not establish calibrated confidence in the averaged coordinates.
