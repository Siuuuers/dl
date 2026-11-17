# Run the sequence-front comparison

Use the new [baomai_sequence_fronts.ipynb](../notebooks/baomai_sequence_fronts.ipynb) for one controlled comparison of the 18 November r2 labels. **PR35 remains the selected model.** This preparation does not establish a model improvement; the returned models still need evaluation.

## Run outside this workspace

1. **[Open the new notebook in Colab](https://colab.research.google.com/github/Siuuuers/dl/blob/codex/video-continuity-review-20261114/notebooks/baomai_sequence_fronts.ipynb).** Choose **Runtime → Change runtime type → GPU**, then **Runtime → Run all**. Keep the published source and protocol pins unchanged.
2. Mount Drive when Colab asks. The notebook verifies the inputs, runs a four-batch resource trial, then trains the reference and candidate in the same runtime. Both runs independently start from original epoch173: **seed 46, 30 epochs, batch 4, input 1280, FP32, AdamW at 0.0001**. Each final `last.pt` is preselected.
3. Return **`baomai-sequence-fronts-seed46-v2-results.zip`**. Its verified backup is in `MyDrive/baomai-sequence-fronts-seed46-v2-20261118`. Keep the ZIP and both completed arms until evaluation is finished, including unfavorable results.

The notebook restores compatible completed artifacts and runs only missing work. It preserves incomplete or incompatible runs for inspection. Neither the resource trial, PR35 weights nor the other arm initializes a full run. No model is automatically promoted.

## Which labels will train

The task is detection of **currently visible cassette front faces**, with actual lower-front corners labeled when locatable. Video neighbors support annotation identity; they do not create entirely hidden-object targets or a tracking state.

All variants keep **137 training photos and 13 empty scenes**. The [complete r2 review](../data/candidates/sequence-refined-fronts-2026-11-18-r2/README.md) and the [bounded training candidate](../data/candidates/sequence-front-corrections-2026-11-18/README.md) have different scopes:

| Labels | Fronts | Locatable endpoints | Missing endpoints |
| --- | ---: | ---: | ---: |
| Exact PR35 control | 3,771 | 6,841 | 701 |
| Bounded v2 treatment | 3,791 | 6,859 | 723 |
| Complete r2 review, kept separately | 3,797 | 6,860 | 734 |

**84 photos differ between the training arms.** On eligible photos the treatment uses the exact r2 reviewed label bytes. On these **nine whole photos**, both arms retain exact PR35 labels because ten case groups remain unresolved:

`new_000005`, `new_000030`, `new_000034`, `new_000041`, `new_000094`, `new_000115`, `new_000117`, `new_000121`, `new_000128`.

This whole-photo rule also excludes supported corrections elsewhere in those photos. In particular, new034's endpoint corrections stay out while its row15 question remains open; new030/new041's lower-front corrections stay out while their upper-front ownership remains unresolved. No uncertain region is erased or turned into a negative. The full review remains unreleased and `full_dataset_certified` stays `false`.

The [v2 protocol](../experiments/sequence-front-corrections-2026-11-18.json) is separately versioned. The old `baomai_accepted_fronts.ipynb` still selects its historical 3,788-object v1 experiment and is preserved for reproduction. The numerical recipe is unchanged; the new supervision and experiment identity are explicit.

## Evaluate the returned pair

Training uses the unchanged original validation labels: **20 photos, 664 fronts, 1,227 locatable endpoints, 101 missing endpoints and three empty scenes**. Its displayed validation metrics use those original labels.

The final comparison evaluates both new models and retained PR35 on both separately pinned development references:

| Reference | Role | Fronts | Remaining uncertain rows |
| --- | --- | ---: | ---: |
| [18 November sequence reference](../data/references/sequence-refined-reference-2026-11-18/README.md) | Primary | 666 | 8 |
| [14 November completion reference](../data/references/visible-front-completion-2026-11-14/README.md) | Historical diagnostic, required | 663 | 13 |

Report each reference separately, using confidence **0.13**, endpoint cutoff **0.50**, NMS IoU **0.70**, matching IoU **0.50**, and **5/10 original-pixel** corner radii. Include misses, duplicate/other extra outputs, empty-scene outputs and corner availability groups; retain eligible misses in corner-success denominators. Inspect gained and lost object identities. A reversal between references leaves the outcome inconclusive. There is no epoch, seed or threshold search in this experiment.

These reused video frames are development data. A favorable comparison would support this bounded label change, while reliability on unseen camera scenes remains unestablished.

## Local preparation check

With the original PNGs present, these commands verify the actual composition, stage both complete datasets, and check held-photo equality, original validation preservation, source-code identity and both separate references. They need no GPU, checkpoint download or network access:

```bash
python -B scripts/compose_sequence_front_training.py --verify-only
python -B scripts/check_sequence_front_training.py
```

The Colab notebook additionally downloads and verifies the actual epoch173 weights before any GPU work. Local label/staging checks do not substitute for its resource trial, completed training or post-training evaluation.
