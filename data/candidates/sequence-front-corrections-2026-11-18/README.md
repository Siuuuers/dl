# Bounded sequence-front training candidate

This is the **v2 training treatment**, composed from the [complete r2 review](../sequence-refined-fronts-2026-11-18-r2/README.md). It is ready for its separate controlled experiment; the full review remains unreleased. **No new model has been trained or selected by this preparation. PR35 remains selected.**

Use [the new notebook and short run guide](../../../docs/sequence-front-training.md). The historical v1 notebook still selects its older candidate.

| Training supervision | Photos | Fronts | Locatable endpoints | Missing endpoints | Empty photos |
| --- | ---: | ---: | ---: | ---: | ---: |
| Exact PR35 control | 137 | 3,771 | 6,841 | 701 | 13 |
| This v2 treatment | 137 | 3,791 | 6,859 | 723 | 13 |
| Complete r2 review, separate | 137 | 3,797 | 6,860 | 734 | 13 |

**84 training label files differ from PR35.** On photos without a remaining case, this candidate contains the exact r2 reviewed bytes. On these nine whole photos, both arms use exact PR35 bytes: **005, 030, 034, 041, 094, 115, 117, 121 and 128**, each with the `new_000` prefix. Their ten unresolved case groups remain bound to the [r2 status ledger](../../../results/review/front-refinement-20261118-r2/review-status.json).

The hold applies to the whole photo. New034's supported endpoint corrections are excluded while row15 remains unresolved. New030/new041's supported lower-front corrections are also excluded while their upper-front ownership remains unresolved. No photo, uncertain region or target is erased to obtain readiness. `full_dataset_certified` remains `false`.

All original training-validation bytes remain unchanged: 20 photos, 664 fronts, 1,227 locatable endpoints, 101 missing endpoints and three empty photos. The separate 18 November reference (666 fronts) is the prospective primary comparison; the historical 14 November completion reference (663 fronts) is a required separate diagnostic. Neither replaces the in-training validation labels.

The [manifest](manifest.json) binds the exact complete review, status ledger, per-image selections, PR35 control and original source inventory. The [v2 protocol](../../../experiments/sequence-front-corrections-2026-11-18.json) fixes independent epoch173 starts, matched 30-epoch runs, seed46, the unchanged numerical recipe, final-epoch selection and both evaluation references. The selected task is currently visible front detection with locatable actual lower-front corners; it does not add hidden-object tracking targets.

From the repository root:

```bash
python -B scripts/compose_sequence_front_training.py --verify-only
python -B scripts/check_sequence_front_training.py
```

The checks use original PNGs but need no weights, GPU or network. The notebook separately downloads and verifies epoch173 before training. Dataset checks establish exact composition and provenance, not physical perfection or improved model performance.
