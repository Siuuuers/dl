# Frozen score and assignment audit

Scope: all142_training_photos. These are frozen, unaugmented training-image diagnostics, not historical training replay, a model improvement, or independent-session evidence.

Native rules and box-only counterfactuals use identical frozen decoded predictions within each checkpoint.

| Model | Branch | Overlap | Known corners | Targets | Unassigned | Mean p | Mean q | Mean p−q |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| candidate | one2many | box_only_counterfactual | 0 | 137 | 0 | 0.19295 | 0.28880 | -0.09585 |
| candidate | one2many | box_only_counterfactual | 1 | 465 | 0 | 0.52955 | 0.47714 | 0.05241 |
| candidate | one2many | box_only_counterfactual | 2 | 3397 | 0 | 0.73971 | 0.68481 | 0.05490 |
| candidate | one2many | native | 0 | 137 | 0 | 0.19262 | 0.28897 | -0.09635 |
| candidate | one2many | native | 1 | 465 | 0 | 0.55248 | 0.64367 | -0.09119 |
| candidate | one2many | native | 2 | 3397 | 0 | 0.74636 | 0.80645 | -0.06009 |
| candidate | one2one | box_only_counterfactual | 0 | 137 | 0 | 0.24691 | 0.64025 | -0.39333 |
| candidate | one2one | box_only_counterfactual | 1 | 465 | 0 | 0.60240 | 0.74216 | -0.13977 |
| candidate | one2one | box_only_counterfactual | 2 | 3397 | 0 | 0.81634 | 0.83718 | -0.02084 |
| candidate | one2one | native | 0 | 137 | 0 | 0.24694 | 0.63879 | -0.39185 |
| candidate | one2one | native | 1 | 465 | 0 | 0.63991 | 0.85852 | -0.21861 |
| candidate | one2one | native | 2 | 3397 | 0 | 0.82321 | 0.91649 | -0.09328 |
| reference | one2many | box_only_counterfactual | 0 | 137 | 0 | 0.16725 | 0.28308 | -0.11583 |
| reference | one2many | box_only_counterfactual | 1 | 465 | 0 | 0.54322 | 0.46579 | 0.07743 |
| reference | one2many | box_only_counterfactual | 2 | 3397 | 0 | 0.78515 | 0.67886 | 0.10629 |
| reference | one2many | native | 0 | 137 | 0 | 0.16726 | 0.28338 | -0.11612 |
| reference | one2many | native | 1 | 465 | 0 | 0.55980 | 0.62618 | -0.06639 |
| reference | one2many | native | 2 | 3397 | 0 | 0.79091 | 0.80317 | -0.01226 |
| reference | one2one | box_only_counterfactual | 0 | 137 | 0 | 0.22517 | 0.62538 | -0.40021 |
| reference | one2one | box_only_counterfactual | 1 | 465 | 0 | 0.64759 | 0.73230 | -0.08472 |
| reference | one2one | box_only_counterfactual | 2 | 3397 | 0 | 0.89307 | 0.83499 | 0.05808 |
| reference | one2one | native | 0 | 137 | 0 | 0.22423 | 0.62571 | -0.40148 |
| reference | one2one | native | 1 | 465 | 0 | 0.67439 | 0.85463 | -0.18024 |
| reference | one2one | native | 2 | 3397 | 0 | 0.89700 | 0.91579 | -0.01879 |

Means above are over surviving anchors. The detailed file preserves per-target eligibility, selection, ownership and quality components. p−q is computed on the same assigned anchor; it is not calibration against observed correctness.

Changing overlap changes assignment and targets together in this descriptive replay. It does not demonstrate what would happen after retraining. No confidence threshold or training objective is selected from these values.
