# R51: one confidence probe from the returned PR143 tensors

This is a CPU replay of one declared change, box confidence **0.13 to 0.05**.
It does not run a model, train weights, change labels, or select a new production
threshold. The original return is pinned by SHA-256:

`892c8ee79ff470990bb47dc59f947ad6bd2577a86bc9827e9dd57f81c56213f3`

The [replay implementation](../../../../scripts/replay_front_confidence_r51.py)
uses NumPy float32 ordinary NMS and the existing repository postclip filter.
Before allowing the lower-confidence probe, it exactly reproduces all 20 native
0.13 anchor-index sequences, full prediction records and postclip outcomes from
the returned archive. It rejects score ties or candidate/output cap situations
that would require a different replay argument. The fixed native geometry is
736x1280 input, 720x1280 original, unit gain and eight vertical padding pixels.

The [summary](summary.json) records baseline checks, tensor identities, helper
hashes, runtime, all four reference identities and aggregate/corner-group scores.
The [mechanism table](mechanisms.json) records all 15 current misses, separate
best-IoU and highest-confidence eligible proposals, ordinary NMS suppressors,
linear Soft-NMS scores and the recovered proposal identities at 0.05. Anchor
indices are zero-based; reference rows and displayed prediction numbers are
one-based. Row geometry is identical in R38 and R50 for all 15 misses.

| Reference | Misses at 0.13 / 0.05 | Unmatched at 0.13 / 0.05 | Duplicates at 0.13 / 0.05 |
| --- | ---: | ---: | ---: |
| Frozen666 | 34 / 27 | 91 / 164 | 17 / 36 |
| Historical663 | 30 / 25 | 90 / 165 | 17 / 35 |
| R38 | 15 / 9 | 84 / 158 | 14 / 36 |
| R50 | 15 / 9 | 84 / 158 | 13 / 35 |

R38/R50 recover `new_000008:R001,R034`, `new_000010:R030,R033`,
`new_000011:R059` and `new_000090:R039`. There are no newly missed targets.
Both-corner success remains 566/583 within 5px and 582/583 within 10px.
The exchange is six recoveries and 74 additional unmatched outputs. The empty
scenes `new_000146` and `new_000147` each gain one output. The default remains
0.13; this evidence motivates investigating training rather than declaring the
lower threshold an improvement.

For `new_000090:R039`, the recovery is a different lower-score proposal
(anchor 9352, score about 0.11097). Lowering the threshold does not unsuppress
the original high-score eligible proposals. Reference-based proposal choices
in the mechanism table explain the result and are never used as deployed policy.

## Reproduce

Run from the repository root with Python and NumPy. Use a fresh output directory
because an existing output with different bytes is deliberately preserved.
Runtime/version metadata may differ across environments even when the numerical
reports agree.

```bash
python -B scripts/replay_front_confidence_r51.py \
  --archive /path/to/baomai-r49-diagnosis-results.zip \
  --output /path/to/new-r51-confidence-probe
```

Add `--include-predictions` only when full replay predictions are needed. The
compact committed receipts contain the assessment; the full predictions can be
regenerated from the immutable returned tensors. The successful terminal check
ends with `Native 0.13 reconstruction exact on all 20 images`.

These are reused development scenes under four annotation versions, not four
independent test sets. The lower-confidence replay is post hoc. Assignment,
soft-target training quality and generalization need a separate controlled
training comparison; the replay alone does not identify their causes.
