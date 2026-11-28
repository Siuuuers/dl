# Reflected detection boxes with equal corner averaging

This completed CPU experiment keeps the reflected view's boxes, scores, order and identities and applies the existing mutual-best association and equal corner averaging with the original view. The [source](../../../../scripts/probe_reflected_anchor_r62.py) makes one fixed exploratory change. It performs no inference, training, threshold search, label editing or automatic promotion.

All common settings remain fixed: object confidence strictly above 0.25, NMS overlap 0.70, matching and association IoU at least 0.50, and corner confidence at least 0.50. The [interpretation](../../detection-errors.md) retains the adverse results.

| Reviewed reference | Retained two-view method | Reflected-box method |
| --- | ---: | ---: |
| Missed fronts | 18 | 15 |
| Extra predictions | 51 | 43 |
| Duplicates included in extras | 5 | 10 |
| Complete pairs within 5 pixels | 569/583 | 569/583 |
| Complete pairs within 10 pixels | 583/583 | 583/583 |

The eight recovered and five newly missed fronts all have incomplete reference corner pairs. On the untouched supplied reference, five-/ten-pixel successes fall from 514/565 to 512/563 out of 584. On the earlier fixed reference they fall from 525/574 to 524/573. The supplied original model remains ahead under those older references.

The input is the same 972,658-byte annotation-reference archive used by the detection audit, SHA-256 `ce8df4bcf223b6bee49fa67e829a443acb572e4a6aad75fa0f92657a16cc2601`. Its inventory, source helper hashes, canonical views, scores, suppression records and associations reproduce exactly. An independent averaging implementation reproduces all coordinates and links. All 648 averaged links are symmetric under anchor reversal; the 583 reviewed complete-reference pairs keep exactly the same averaged coordinates.

No association is assigned to two different reference targets in this sample. Some links remain unassigned on one or both sides, so this is not proof of every link's physical identity. The summary retains those categories.

| File | Purpose |
| --- | --- |
| `summary.json` | All twelve comparison rows, paired changes, accepted-pair accounting and verification |
| `reflected-anchor-predictions.json.gz` | Every resulting output |
| `reflected-anchor-associations.json.gz` | Prediction-only association and fallback decisions |
| `association-identities.json.gz` | Reference-based diagnostic of association identity |
| `paired-cases.json.gz` | Every reference target and four-method comparison |
| `per-image.json.gz` | All detailed score tables |
| `inventory.json` | Result-member byte counts and hashes |

Run from a source checkout, keeping results outside it:

```sh
python -B scripts/probe_reflected_anchor_r62.py --work /tmp/cassette-reflected-boxes
```

The [CPU notebook](../../../../notebooks/baomai_detection_audit_r62.ipynb) reproduces both this experiment and the error audit. Fresh GPU model results remain pending in their separate notebooks.
