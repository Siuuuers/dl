# Completed corner-conditioned suppression comparison

The [four-cell notebook](../../../../notebooks/baomai_current_proposals_r69.ipynb) completed this saved-output experiment alongside a separate current-checkpoint proposal trace. It executes published implementation `41d0e6937c695377373cc55071ffb780345870ee`. The [execution receipt](../current-proposals-r69/notebook-execution.json) binds the actual cells, logs and complete result package.

## Question and fixed rule

Does agreement between the two predicted lower corners help additional box suppression remove extras while retaining matched fronts?

The [protocol](protocol.json) was recorded before the comparison. All five unchanged R68 output sets are evaluated under four references. The R68 reflected-output union is excluded. The three conditions are unchanged outputs, additional hard suppression above IoU 0.60, and that suppression with a corner-agreement condition.

The qualified rule suppresses a lower-ranked output only when it shares the higher-ranked output's class, box IoU is strictly above 0.60, both have two accepted corners, and both corresponding corner distances are at most five pixels. Accepted corners have finite coordinates inside the 1280 by 720 image and confidence at least 0.50. Output confidence orders candidates, with original output index breaking ties. Survivors keep their original records and relative order. Reference annotations enter only fresh evaluation matching.

## Main result

The table uses the PR158 reviewed-label checkpoint and the extended r67 reference with 658 fronts.

| Rule | Misses | Extras | Duplicate subset | Pairs within 5 pixels | Pairs within 10 pixels |
| --- | ---: | ---: | ---: | ---: | ---: |
| Unchanged | 18 | 50 | 13 | 572/583 | 583/583 |
| Additional suppression at 0.60 | 19 | 31 | 2 | 572/583 | 583/583 |
| Suppression at 0.60 with corner agreement | 18 | 44 | 11 | 572/583 | 583/583 |

The corner condition removes six extras, including two duplicates. It preserves every reference target's matched output and known-corner outcome under all four references. Incomplete-front coverage stays at 57/75 under r67. The [target-change ledger](target-changes.json.gz) confirms this preservation at target level.

The same rule loses one previously matched one-corner front for the supplied-label checkpoint, `new_000009/R012`. Its known right corner previously had 1.622 pixels error. This counterexample keeps the benefit specific to the reviewed-label model's predictions.

The companion fresh CPU trace applies the identical selector to the same reviewed-label checkpoint. It removes six outputs and preserves target outcomes there too. The archived runtime changes duplicates from 13 to 11, while the fresh runtime changes them from 14 to 12. Both change extras from 50 to 44 and retain 18 misses. The [current trace](../current-proposals-r69/results.md) explains the separate runtime category change.

## Evidence

- [Complete 60-row matrix](matrix.csv) and [readable results](results.md).
- [Per-image scores](per-image.json.gz), covering 1,200 image/reference/method/condition records.
- [Selected indices](selected-indices.json.gz) and exact [surviving predictions](predictions.json.gz).
- [Removed-output ledger](removed-outputs.json.gz), [overlapping-pair comparisons](overlapping-pairs.json.gz) and [target changes](target-changes.json.gz).
- [Independent verification](independent-verification.json), including separate IoU, selector and scoring reconstruction.
- [Result inventory](inventory.json) and [summary](summary.json).

The complete notebook ZIP retains these same files under its `corner-suppression/` prefix. No training, annotation editing or threshold search occurs in this experiment. The rule is an optional measured candidate and does not alter live operating defaults.
