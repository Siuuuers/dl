# R48 proposed development reference

This directory contains **twenty development label files**, not a full training
dataset. It carries forward all 654 R38 instances and all corner availability
states. The original R38 reference remains unchanged.

The [ledger](audit-ledger.json) maps fifteen user cues to ten existing targets
in seven original photographs. Nine rows change: eight box fields and two
visible corner positions. No object is added or deleted. The bottom of
`new_000010 / R029` and the right bound of `new_000089 / R041` remain explicit
ownership holds; their prior coordinates are retained.

Old P numbers identify PR133 candidate postclip predictions, reproduced exactly
in the PR137 reference cache. They are not the new PR137 candidate P numbers.
R numbers identify one-based lines of the original R38 development files.
Each cue object, original hash, exact before-line, nominal update and reason is
recorded in the ledger.

The proposed reference is assistant-reviewed, development-only, and not
independently human-certified. Unreviewed rows are inherited. It must not be
silently substituted into the published R38 model comparison or used as a new
training dataset. A [fixed-cache sensitivity replay](../../../results/evaluation/front-errors-r48-20261127/reference-sensitivity.json)
compares both unchanged model arms under both label versions and identifies
score changes caused by the reference itself.

See [the R48 amendment report](../../../docs/front-review-r48.md) and
[annotation policy](../../../docs/annotation-policy.md). The builder validates
every label against the pinned parent and numeric edit replay before exporting
the manifest and TXT files.
