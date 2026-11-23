# PR97 / r23 returned-model evaluation

**Recommendation: retain PR35. The r23 candidate does not support promotion.** On the frozen 666-front primary reference it has 18 fewer matched targets, 30 more unmatched outputs and fewer both-corner successes than PR35. Its ranking reverses on the separately pinned pre-return r23 review. This is evidence of reference dependence, not a basis for choosing the more favorable reference.

Measured on 23 November 2026, Asia/Hong_Kong. The two returned final checkpoints were inferred once. Exact compatible PR35 predictions and the completed PR96 evidence were reused. No training, tuning or model-selection change was performed; no r24 annotations were loaded.

## Fixed measurements

All three models use the same CPU runtime and fixed operating point: input 1280, box confidence 0.13, endpoint confidence 0.50, NMS IoU 0.70, matching IoU 0.50, FP32, maximum 300 outputs and four threads. Both-corner success means both predicted points lie within the radius in original-image pixels and pass endpoint confidence. Its denominator is all **584 fronts with two locatable reference corners**, including eligible missed objects. Unmatched outputs are matching outcomes; they are not all independently adjudicated background errors. Duplicates are a subset of unmatched outputs.

| Reference | Model | Matched | Unmatched targets | Unmatched outputs | Duplicates | Both corners ≤5 px | Both corners ≤10 px |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Frozen 666 primary | Retained PR35 | 648 | 18 | 80 | 32 | 533/584 | 582/584 |
| Frozen 666 primary | Fresh r23 control | 646 | 20 | 83 | 29 | 531/584 | 583/584 |
| Frozen 666 primary | Prior r21 candidate | 631 | 35 | 100 | 27 | 518/584 | 574/584 |
| Frozen 666 primary | r23 candidate | 630 | 36 | 110 | 32 | 521/584 | 575/584 |
| Historical 663 | Retained PR35 | 648 | 15 | 80 | 32 | 533/584 | 582/584 |
| Historical 663 | Fresh r23 control | 646 | 17 | 83 | 29 | 531/584 | 583/584 |
| Historical 663 | Prior r21 candidate | 632 | 31 | 99 | 27 | 518/584 | 574/584 |
| Historical 663 | r23 candidate | 631 | 32 | 109 | 32 | 521/584 | 575/584 |
| Pre-return r23 / 662 exploratory | Retained PR35 | 625 | 37 | 103 | 21 | 545/584 | 573/584 |
| Pre-return r23 / 662 exploratory | Fresh r23 control | 624 | 38 | 105 | 16 | 541/584 | 573/584 |
| Pre-return r23 / 662 exploratory | r23 candidate | 645 | 17 | 95 | 33 | 556/584 | 583/584 |

All three current comparison models return **zero outputs on all three empty scenes**. On the primary reference, the candidate's unmatched targets comprise 13 zero-corner, 16 one-corner and 7 two-corner fronts. On the exploratory reference, they comprise 7 zero-corner and 10 one-corner fronts; all 584 two-corner fronts match. Candidate miss-free nonempty scenes are 4/17 on the primary reference and 9/17 on the exploratory review. Full group counts, per-scene scores and gained/lost identities are retained in [fixed-comparison.json](fixed-comparison.json); [summary.json](summary.json) is the compact machine-readable result.

Against the fresh control on the primary reference, the candidate gains 3 and loses 19 matched identities. Against PR35 it gains 3 and loses 21. The fixed primary and historical references retain 8 and 13 unresolved rows respectively; these remain included. The pre-return r23 ledger has no unresolved development rows, which is not a guarantee that every physical boundary is correct.

### Fair comparison with the preceding returned model

On either unchanged frozen reference, r23 versus the prior r21 candidate gives **one fewer match, 10 more unmatched outputs, 5 more duplicates, 3 more both-corner successes within 5 px, and 1 more within 10 px**. This is a mixed result. The r23 control reproduces the prior r21 control's prediction arrays exactly on all 20 images despite different checkpoint-file hashes. The candidate changes predictions on all 17 nonempty images; the three empty-image arrays remain identical.

The prior r21 rows above come from the completed [PR96 evidence](../pr96-r21-20261122/README.md), using identical frozen-reference hashes and inference settings. Its exploratory endpoint reference differs from r23, so exploratory corner-score differences are not presented as an isolated training effect. The r23 experiment also includes the intervening r22 annotation amendments; it does not isolate the four latest r23 training-point edits. Both fresh arms compare accumulated eligible supervision against PR35 supervision. PR35 originally trained on an A100, while these fresh arms used an L4; PR35-versus-control differences cannot be attributed only to labels.

## Why the reference ranking changes

[reference-sensitivity.json](reference-sensitivity.json) traces identities through the immutable recovery and successive review row maps, replaying their hash-bound operations. It does not assign identities by proximity or assume unchanged row numbers. Of the 666 primary identities, 660 persist and 6 are removed; two added identities yield 662 exploratory targets.

The candidate's 36 primary matching misses divide into **16 reference-box crossings**, **6 removed identities**, and **14 retained identities still unmatched**. For each crossing, the exact same returned prediction changes from IoU below 0.5 against the primary box to at least 0.5 against the r23 box. Three previously matched identities become unmatched, and both added identities match: `630 + 16 − 3 + 2 = 645` matches without changing model output.

This diagnoses matching sensitivity to reference geometry. It does not independently prove either box definition correct. Accordingly, “36 unmatched primary targets” must not be described as “36 physical cassettes with no detector output.” No new visual adjudication of these model errors is claimed here; the preserved PR96 analysis remains separate.

## Archive, initialization and resources

The audit uses [PR97](https://github.com/Siuuuers/dl/pull/97) at head `a58b8247e104e2fc700a49ac5c651464a570eefb`. Its seven changed files return the archive, two chunk ZIPs, checksums and documentation. The complete archive was available directly as a Git blob: **39,783,521 bytes**, SHA256 `3f12115b2c15dec87a7e7404115ff933c54fadbea7eef415a098533933059be9`. CRC, safe unique paths, the exact 13-member inventory and all 12 indexed payload hashes and sizes passed. Chunk reconstruction was unnecessary.

Training source is `ffe4d4ecb6699180e7b25ee10fc9eb31647fec38`; protocol SHA256 is `a4183607c6737a8466aef4edd625ca52450a486dd030456e4d10e3cd98736a54`. The notebook delivered at `ee2ca5f260af96621aed4bf5803680b9b6403349` has SHA256 `c49f188c9d74c4413d6fed92e9ba878a8d4ad769fe553281a665b5941788bb36` and pins that exact source and protocol. The pre-return r23 exploratory manifest is `d93be83707ee3090d05808333410c924c8a32df82df87f4579f0c3aead4176d0`.

Both arms independently initialize from original epoch 173, SHA256 `e0aab1b377491700318e0b14cd4c558ec3c1684087a13b7dde2298ebdbf2847e`. They complete 30 epochs, 1,050 batches, 4,110 image presentations and 78 optimizer steps, with exit code 0, no stop reason and preselected final-epoch checkpoints. Both use recipe `reviewed-incremental-fronts-seed46-v1`, seed 46, batch 4, input 1280, FP32 and AdamW with initial learning rate 0.0001. The entire recorded training runtime and fixed settings agree between arms.

| Arm | Wall time | Sampled peak process-tree RSS | Worker peak RSS |
| --- | ---: | ---: | ---: |
| Fresh control | 456.851 s | 2,984.953 MiB | 2,965.594 MiB |
| r23 candidate | 455.462 s | 2,942.719 MiB | 2,957.160 MiB |

Both remain below their recorded 21,600-second and 12,000-MiB limits. RSS sampling is not an exhaustive peak or GPU VRAM measurement; no GPU-memory peak receipt was supplied. Training used Python 3.13.16, Torch 2.14.1+cu130, Ultralytics 8.4.182 and the same NVIDIA L4 fingerprint for both arms.

All 157 original images, staged source bytes, initializer bytes and source-script hashes were verified. Exactly 92 training-label files differ between arms. Both contain 137 training images, including 13 empty scenes; control has 3,771 fronts / 6,841 locatable endpoints, candidate 3,784 fronts / 6,836 locatable endpoints. Photo 015 and the 31 excluded old-series photos retain exact PR35 labels in both arms. Original trainer validation remains identical at 664 fronts, distinct from the 666/663 scoring references and the 662-front exploratory review.

| Checkpoint | SHA256 |
| --- | --- |
| Retained PR35 | `420c00f73eeba24e587342400878b7d96371d3e745772113809aa705ec8163b1` |
| Fresh r23 control | `dae9539320c954677f6874d9ede087ba98955d816f49ecf711da12187bf7c890` |
| r23 candidate | `e1d9104b9add3e39bbd9b0d728c58d258e42cf1501ac9615a14e61e5e1c8716e` |

CPU inference uses Python 3.12.14, Torch 2.14.1+cpu, torchvision 0.29.1+cpu, Ultralytics 8.4.182, NumPy 2.4.6, Pillow 11.3.0 and OpenCV 5.0.0.93. Exact PR35 cache reuse passed runtime, checkpoint, scorer, original-image identity and inference-contract checks; its frozen-reference scores reproduce PR96. The detached r23 checkout's hydrated LFS inputs are byte-verified against both Git LFS metadata and the original manifest. [Inference provenance](inference-provenance.json) preserves these checks and resource receipts.

## Reproduction and limits

Use a hydrated checkout of the pinned r23 source, with the delivered notebook commit also available as a Git object. The scripts accept explicit paths, avoiding dependence on the saved machine-specific commands:

- [audit_return.py](audit_return.py): `--root`, `--archive`, extracted `--bundle`, fresh `--output`. Verifies the exact return and its source, staging and completion receipts without training.
- [run_pr97_inference.py](run_pr97_inference.py): `--root`, pinned-runtime `--python`, fresh `--output`, `--reference-manifest`, `--candidate-manifest`. Runs only the two returned finals at the fixed operating point.
- [measure_pr97.py](measure_pr97.py): `--root`, three prediction paths (`--pr35`, `--reference`, `--candidate`), their three `--*-manifest` paths and fresh `--output`. Reproduces all three reference scores from predictions and text receipts without model weights. PR35 predictions and receipts remain in the sibling PR96 evidence directory.
- [reference_sensitivity.py](reference_sensitivity.py): `--root`, this `--evidence` directory and fresh `--output`; optional `--pr35` overrides the sibling PR96 cache path. Replays identity lineage and exact IoU crossings.

The [adaptation patch](measurement-adaptation.patch) records the limited PR96-to-PR97 changes, retaining the scoring logic. Training logs are byte-preserved as `train-log.txt`; replay restores their original names in temporary directories for strict receipt validation. [Replay verification](replay-verification.json) confirms the preserved evidence reproduces the fixed comparison byte-for-byte. Raw return archives and model-weight binaries are not duplicated here.

These repeatedly reviewed development frames do not estimate performance in unseen capture sessions. Pixel accuracy of unlocatable corners cannot be measured; confidence is not a calibrated probability of physical visibility or corner accuracy. The frozen-reference regressions and reference-dependent ranking leave PR35 as the retained selection.
