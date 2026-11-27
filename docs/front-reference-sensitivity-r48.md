# R48 reference sensitivity with fixed PR137 predictions

The [R48 development amendments](front-review-r48.md) improve the recorded
physical target definition. Their effect on a metric must be distinguished
from a change in the model. This CPU replay changes only the reference labels,
then reruns the unchanged scorer on both saved PR137 arms.

All twenty generated R48 label files equal an independent replay of the
exact R38 before-lines and numeric operations. The complete saved R38 score
dictionaries reproduce exactly for both arms, in both raw and postclip views.
The [machine-readable evidence](../results/evaluation/front-errors-r48-20261127/reference-sensitivity.json)
contains all 654 target assignments per arm/view, prediction-category
transitions, reviewed-target endpoint errors, and isolated-operation plus
leave-one-out attribution.

## Same predictions, two reference versions

| Postclip metric | Reference / R38 | Reference / R48 proposal | Candidate / R38 | Candidate / R48 proposal |
| --- | ---: | ---: | ---: | ---: |
| Matched targets | 641 | 641 | 639 | 639 |
| Missed targets | 13 | 13 | 15 | 15 |
| Total unmatched outputs | 90 | 90 | 84 | 84 |
| Formal duplicates | 18 | 16 | 14 | 13 |
| Other unmatched outputs | 72 | 74 | 70 | 71 |
| Both corners within 5px | 569/583 | 570/583 | 566/583 | 566/583 |
| Both corners within 10px | 581/583 | 581/583 | 582/583 | 582/583 |
| Returned eligible endpoints | 1209/1220 | 1209/1220 | 1208/1220 | 1208/1220 |

Every target keeps its matched/missed state and, where matched, its assigned
prediction identity. No cassette is newly detected or lost in this replay.
The nine amended rows preserve 654 objects: 583 with two locatable corners,
54 with one, and 17 with none. Both explicit development holds remain unchanged.

## Why the duplicate counts change

These identities belong to the stated PR137 arm's postclip list:

- Reference `new_000089 / P058`: changing R049 top to 473 moves best IoU
  from 0.5040958 to 0.4937644. It becomes other unmatched instead of a formal
  duplicate.
- Both arms `new_000090 / P050`: changing R040 top to 469 moves reference
  IoU from 0.5847431 to 0.4978972 and candidate IoU from 0.5792473 to 0.4939786.
  Both predictions become other unmatched.
- Candidate `new_000087`: changing R043 top to 433 moves P059 from duplicate
  to other unmatched (0.6487203 to 0.4184423), while P063 moves in the opposite
  direction (0.4679478 to 0.5767810). The aggregate duplicate count is unchanged
  for this photograph. No target assignment changes.

Applying each amendment alone and reverting it from the complete proposal
reproduces these crossings. Full confidence-ranked greedy matching is rerun;
the analysis does not assume additive effects from independent rows.

For reference `new_000090 / R047`, the revised physical corner positions
change the fixed right prediction's distance from 5.1881 px to 0.6837 px, and the
left distance from 2.8098 px to 0.7718 px. That accounts for one additional strict
corner success. The model did not learn anything during this replay.

## Consequence for the project

These particular amendments do not reduce the candidate's 15 misses or 84
unmatched outputs. A decrease in the formal-duplicate subset alone would be
misleading evidence of a better detector: some outputs simply crossed the
reference-dependent IoU boundary between two unmatched categories.

Keep the original R38 result and separately name the R48 proposal result.
The latter remains post-hoc development analysis, including the held R029
bottom and R041 right boundary. It is neither an independent test set nor
human sign-off on all 654 targets.

## Reproduce

After exporting the R48 labels with the review builder:

```bash
python scripts/evaluate_front_review_r48_reference.py \
  --archive /absolute/path/to/baomai-additional-front-pilot-r47-results.zip
```

Final mode requires all twenty generated labels to equal the independent
operation replay. The script writes only the JSON report. No inference,
training, checkpoint mutation or label mutation occurs.
