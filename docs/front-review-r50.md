# R50: one clarified boundary and a more explanatory error review

27 November 2026, Asia/Hong_Kong.

**The physical bottom of `new_000010 / R38 row 029` is now nominal `ymax=714`,
previously 720.** The user's new clarification identifies the lower band as
another cassette. Original-image review places the lower-left front transition
near the retained visible left point `(325,714)`. This resolves the material-owner
hold from [R48](front-review-r48.md), while retaining the missing right corner.

The latest completed model experiment is still [PR137](additional-front-pilot-r47-results.md).
[PR140](https://github.com/Siuuuers/dl/pull/140) repairs a missing sparse-checkout
dependency in the R49 notebook; it does not contain a completed diagnosis return.
The [R50 notebook](diagnosis-r50.md) repairs resumed checkouts too and verifies
source/data closure before model inference.

## Exact annotation decision

| Field | R48 predecessor | R50 proposal |
| --- | ---: | ---: |
| Left bound, `xmin` | 316.5845 | unchanged |
| Top bound, `ymin` | 666 | unchanged |
| Right bound, `xmax` | 457.0427 | unchanged |
| Bottom bound, `ymax` | 720 | 714 |
| Left lower corner | `(325,714)`, visible | exact tokens preserved |
| Right lower corner | `0 0 0`, unavailable | exact tokens preserved |

The original image is 1280 by 720, with Y increasing downward. Physical bottom
means stored `ymax`. The chosen box follows the identified visible front; there
is no general rule to snap a box edge to a corner or to a model prediction. The
nominal coordinate has about 2 px visual uncertainty, an explanatory estimate
rather than a calibrated confidence interval.

The [versioned development reference](../data/reviews/fronts-2026-11-27-r50/README.md)
contains all 20 TXT files: 654 fronts, 1,220 visible and 88 missing corner slots.
Only one field in one row changes. No instance, corner, availability token or
training label changes. Rows inherited from R48 are not newly certified. One
development field remains held: `new_000089:R38:041 / xmax`, whose thin rim beside
the gripper has unresolved cassette ownership. The earlier six complete
supplementary photographs and six held photographs are unchanged.

## What the fixed-prediction replay establishes

The same PR137 postclip prediction arrays were scored against R48 and R50.
Every assigned prediction identity and matched/missed target state is preserved.
The [full sensitivity report](../results/review/front-review-r50-20261127/reference-sensitivity.json)
contains both arms and every per-image score.

| Fixed PR137 output / metric | Reference on R48 | Reference on R50 | Candidate on R48 | Candidate on R50 |
| --- | ---: | ---: | ---: | ---: |
| Missed targets | 13 | 13 | 15 | 15 |
| All unmatched outputs | 90 | 90 | 84 | 84 |
| Formal duplicate subset | 16 | 16 | 13 | 13 |
| Other unmatched outputs | 74 | 74 | 71 | 71 |
| Both corners within 5 px | 570/583 | 570/583 | 566/583 | 566/583 |

These R48/R50 duplicate counts differ from the original R38 counts, 18/14 for
reference/candidate, because the reference changed in R48. They do not represent
a new prediction filter or a newly trained model. The R50 boundary clarification
itself does not change any aggregate score. PR35 remains selected under the
existing frozen-reference gate.

## Two review documents

`r50-annotation-amendment.pdf` is a focused two-page document with unmarked
original pixels, R48/R50 overlays, exact preserved fields, the unchanged scores,
and the remaining ownership hold.

`r50-missed-errors-unmatched.pdf` retains all 119 original R38 cases for traceable
comparison: 15 missed targets, 20 matched corner failures, and 84 unmatched
outputs. Each case adds an observed type, evidence, conclusion, next action and
uncertainty, with current R50 status shown separately. These are review-card
counts, not 119 distinct physical cassettes. See the [error analysis](front-error-review-r50.md).

Both PDFs are delivered separately. Source code, label versions and text/JSON
evidence are stored in Git; original images, weights, archives and PDFs are not
added by this change. The existing HTML stays as its historical review snapshot.

## Reproduce the amendment

```bash
python scripts/audit_additional_front_pilot_r47_results.py \
  --download-dir /tmp/r50-pr137 --output /tmp/r50-pr137-audit.json
python scripts/build_front_review_r50.py \
  --archive /tmp/r50-pr137/baomai-additional-front-pilot-r47-results.zip \
  --originals /path/to/verified-originals \
  --output /tmp/r50-amendment \
  --pdf /tmp/r50-annotation-amendment.pdf
```

The builder validates all 20 exported labels against the exact parent, source
hashes, preserved point tokens, unchanged membership and original archive. The
PDF renderer checks the original images it uses. The committed
[verification receipt](../results/review/front-review-r50-20261127/verification.json)
and [PDF QA record](../results/review/front-review-r50-20261127/pdf-qa.json) describe
the checks actually completed.

The professor's lecture and current annotation/training assignment are discussed
in [the revised project direction](project-direction-r50.md). A thesis topic is
still open; the lecture is background material, not an approved thesis proposal.
