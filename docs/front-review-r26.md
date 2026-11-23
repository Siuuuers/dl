# R26: targeted training corrections and the next comparison

23 November 2026, Asia/Hong_Kong.

**Use the R26 notebook for the next controlled comparison. PR35 remains the selected model; R26 has not been trained.** The [completed PR101 evaluation](../results/evaluation/pr101-r25-20261123/README.md) found no basis to replace PR35 with R25. This round adds a small, supported training intervention alongside the requested development-label amendments.

## Run the notebook

Open [baomai_front_review_r26.ipynb](../notebooks/baomai_front_review_r26.ipynb).

1. Select a GPU runtime.
2. Choose **Run all**. The existing Drive option preserves completed runs.
3. Return **`baomai-front-review-r26-results.zip`** after both arms finish.

The notebook pins the published source and [R26 protocol](../experiments/front-review-2026-11-23-r26.json). Its input preflight checks the original images and initializer, replays the saved changes against R25, verifies row identities and accepted standards, and checks the actual training inputs before running the resource trial. The remaining six runtime cells preserve the proven R25 behavior apart from round names and pins.

## What enters training

The new training intervention affects only **`new_000013`, rows 29–31**, compared with the R25 candidate. These are existing training examples analogous to the reviewed partly visible fronts beside the rack in photo 009. The lower-left points are moved to the observed front-face turn; two left box edges are tightened where they include the rack.

| Photo 013 row | R25 left point, px | R26 left point, px | Left box boundary |
| --- | --- | --- | --- |
| 29 | (682, 450) | (684, 446) | 681 → 682 px |
| 30 | (682, 469) | (683, 465) | 681 → 682 px |
| 31 | (682, 488) | (683, 485.5) | Unchanged |

Coordinates above are rounded for reading; exact normalized tokens remain in the [row ledger](../results/review/front-review-20261123-r26/row-review.json). Each row's right corner remains unavailable with its original zero tokens. Classes, visibility and training object counts remain unchanged. These are **three left-corner corrections and two box-boundary corrections**, with no new images or instances.

The separately versioned development review contains **16 row replacements and one deletion** across photos 007, 008, 009, 010, 053, 089 and 090. It now has **661 fronts**. Those changes refine the exploratory scoring reference; they do not enter either training arm. The review covers 22 named development rows, including five retained rows, plus the three training analogues. All **11 accepted standard rows** remain byte-identical. The saved row maps preserve the relationship to R25 after the deletion.

## What the comparison can establish

Both arms independently start from original epoch173 with **seed 46, 30 epochs, batch 4, input 1280, FP32 and AdamW at initial learning rate 0.0001**. The resource trial does not initialize either full run. The final epoch is selected in advance. Both arms run in the same recorded environment; an old completed control is not substituted into the new pair.

The control uses PR35 supervision, and the candidate uses accumulated eligible R26 supervision. Both retain 137 training images, including 13 empty scenes, original trainer validation of 20 images/664 fronts, exact PR35 labels on held photo 015, and the 31 old-series treatment exclusions. This within-run comparison therefore tests the accumulated eligible corrections. A direct R25-to-R26 comparison can examine the small new training intervention only after its input/runtime receipts are checked and both models are scored against identical reference bytes.

Retain the **666-front primary**, **663-front historical** and **661-front R26 exploratory** references as separate versions. A development-label correction can change a score while model predictions remain unchanged. An improvement on one reference alone does not establish a universal replacement or reliability in a new camera session.

The measured R25 result provides context: on the frozen 666-front reference, PR35/R24/R25 had **18/35/37 misses** and **80/105/108 unmatched outputs**. On the same accepted R25 review, R24/R25 had **16/17 misses**, **90/92 unmatched outputs**, and identical successful two-corner identities at 5 and 10 pixels. The new three-row training correction is a specific hypothesis about supervision, not evidence that R26 will improve those results or resolve every partial-front miss.

## Saved identities

- [Candidate manifest](../data/candidates/front-review-2026-11-23-r26/manifest.json): SHA-256 `4f3a75054b130a3627a523a9d0f1479834cc98fc3dfc66253efd01794dee109f`.
- [Training protocol](../experiments/front-review-2026-11-23-r26.json): SHA-256 `232f81cdf6d367fc9ece902e01b4ab5a03da5ae0898f5f75abae25ef5001372f`.
- [Review manifest](../data/reviews/fronts-2026-11-23-r26/manifest.json), [operations](../results/review/front-review-20261123-r26/operations.json) and [row maps](../results/review/front-review-20261123-r26/row-maps.json) retain the annotation evidence and lineage.

The final notebook builder requires a real published source commit. It refuses an unpinned source or a protocol/builder that differs from that commit. Preparation performs no GPU training and makes no model-improvement claim.
