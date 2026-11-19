# PR79 r10 result audit

**Keep PR35 as the selected model.** At the unchanged operating point, the returned r10 candidate has **13 more scored misses and 15 more unmatched outputs** than its fresh control on the frozen 666-front primary reference. It gains two both-corner successes within five pixels and loses three within ten pixels. The historical 663-front diagnostic also has more misses and unmatched outputs. These results do not support promotion.

This audit reran inference and scoring on 19 November 2026. It did not repeat training, change either frozen reference, tune thresholds, or alter model selection. [Full comparison](fixed-comparison.json) preserves every prediction identity, match, gain and loss; [compact summary](summary.json) contains the main numbers.

## Verified return and actual training inputs

The [PR79 return](https://github.com/Siuuuers/dl/pull/79) is pinned to head `f9a67b6df66fb3227b6b0197f49b6514c227a150`. Its archive has 39,782,413 bytes and SHA-256 `6948372aa37c4a33b4168e6699f5819997ba4bf64f05bb4d258892e37ac3276b`. ZIP CRC, safe unique member names, exact inventory, and the hashes and sizes of all 12 indexed members passed.

Training used source `bae1ed46bf3337956dbd60ac85aa5bcc759bbbcd` and [r10 protocol](../../../experiments/front-review-2026-11-19-r10.json), SHA-256 `0b73d423fcaff8f7b7d72da0e6bc33acb5814f2d8435fffa0e163b3efd1bd48f`. The audit checked recorded source files against that Git revision and the preserved local source, validated both completed checkpoint bundles, checked staging code identities, and rehashed every staged image and label source. All 157 original PNGs and original labels passed their frozen source hashes. The original epoch173 initialization checkpoint was also downloaded and rehashed successfully.

| Actual r10 training input | Fresh control | Candidate |
| --- | ---: | ---: |
| Training photographs | 137 | 137 |
| Empty training scenes | 13 | 13 |
| Labelled fronts | 3,771 | 3,778 |
| Locatable endpoint slots | 6,841 | 6,846 |
| Missing endpoint slots | 701 | 710 |
| Original validation photographs / fronts | 20 / 664 | 20 / 664 |

The arms differ on **77 training label files**. All **15 held training photographs** and all **31 excluded old photographs** use exact PR35 supervision in both arms. The 157 original PNGs and the 20 original validation label files are identical across arms. Equality of held supervision controls the experimental difference; it does not certify those inherited labels as physically correct.

Both runs completed 30 epochs, 1,050 batches, 4,110 image presentations and 78 recorded optimizer steps. Both used the fixed seed-46 recipe, independently started from epoch173, and selected final epoch 30. Both ran on NVIDIA A100-SXM4-40GB. The recorded training wall times were 384.89 seconds for control and 379.81 seconds for candidate. These single-run timings are receipts, not a controlled speed comparison.

Each run staged both datasets under its own absolute directory. This changes the corresponding YAML hashes; reconstruction from each recorded directory reproduced all four YAML hashes. The actual image and label inventories, paths, contents and counts agree across the two staging receipts. [Verified return](verified-return.json) records the checks and [checkpoint receipts](checkpoint-receipts/) preserve the original manifests, logs, CSVs and receipts without copying the large weights into this audit directory.

## Fresh fixed inference

All three checkpoints were freshly inferred in the same CPU environment: PyTorch 2.14.1+cpu, torchvision 0.29.1+cpu, Ultralytics 8.4.182, NumPy 2.4.6, Pillow 11.3.0 and OpenCV 5.0.0.93. [Package pins](inference-pip-freeze.txt) and [actual commands](inference/commands.json) record that environment.

The operating point remains box confidence 0.13, endpoint confidence 0.50, ordinary NMS IoU 0.70, matching IoU 0.50, FP32, input size 1280, maximum 300 detections and four CPU threads. No confidence, NMS, seed, epoch or reference search was performed. The scorer code is unchanged; [measurement adaptation](measurement-adaptation.patch) changes only the experiment, protocol and model identities and description strings from the prior audit.

The fresh r10 control's **ordered decoded prediction arrays exactly match retained PR35 on all 20 photographs**. Fresh PR35 predictions also exactly reproduce its prior PR76 audit on all 20 photographs. This is an observed equality for this replay, not an assumption that every future control will be identical. [Prediction consistency](prediction-consistency.json) records the comparisons.

### Primary reference: 666 fronts

The frozen primary reference contains 20 photographs, 17 nonempty scenes, three empty scenes, 1,227 locatable endpoints and 105 missing endpoint slots. Its manifest SHA-256 remains `0de64be3797dce2e43d4cb98d09620eab969bb95725ed9e75682dcfcd53c7f96`.

| Model | Matched fronts | Scored misses | Unmatched outputs | Duplicate subset | Both corners <=5 px | Both <=10 px |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Retained PR35 | 648/666 | 18 | 80 | 32 | 533/584 | 582/584 |
| Fresh r10 control | 648/666 | 18 | 80 | 32 | 533/584 | 582/584 |
| PR79 r10 candidate | 635/666 | 31 | 95 | 29 | 535/584 | 579/584 |

The corner-pair denominator is the **584 reference fronts with two locatable corners**. Missed eligible fronts remain failures in that denominator. This is not complete-scene accuracy, and it does not remove partial fronts from the 666-front detection denominator.

| Locatable reference corners | Fronts | PR35 / control misses | Candidate misses |
| --- | ---: | ---: | ---: |
| Zero | 23 | 10 | 13 |
| One | 59 | 7 | 15 |
| Two | 584 | 1 | 3 |

All three models return zero outputs on the three empty scenes. PR35 and control have six miss-free nonempty scenes out of 17; candidate has four. None has a fully error-free nonempty scene under this fixed matcher. Candidate returns 730 outputs compared with 728 from control; 13 fewer matches and two more outputs account for its 15 additional unmatched outputs.

### Historical diagnostic: 663 fronts

The separately versioned historical reference retains manifest SHA-256 `bbc1dca71f7a3e0fe2f54a4391962e562c3353a62533eb233b697819edcc64b1`.

| Model | Matched fronts | Scored misses | Unmatched outputs |
| --- | ---: | ---: | ---: |
| Retained PR35 | 648/663 | 15 | 80 |
| Fresh r10 control | 648/663 | 15 | 80 |
| PR79 r10 candidate | 636/663 | 27 | 94 |

Both-corner success counts remain those in the primary table. Candidate has 12 additional misses and 14 additional unmatched outputs relative to its fresh control. The references share inherited labels and are not independent physical certification. The current r10/r11 development review was not substituted into either frozen comparison.

## Where matches changed

Against control, the primary reference gains `new_000008:1`, `new_000009:53` and `new_000087:15`, and loses 16 identities:

- `new_000008`: rows 33, 34 and 35.
- `new_000009`: row 56.
- `new_000010`: rows 15, 31 and 32.
- `new_000011`: row 61.
- `new_000056`: rows 14 and 15.
- `new_000057`: rows 7 and 11.
- `new_000089`: row 43.
- `new_000090`: rows 41, 42 and 43.

These row numbers belong to the frozen primary reference. They should not be assumed to match a later annotation PDF after additions or deletions.

A scored miss means the fixed one-to-one matcher found no qualifying box match. It does not necessarily mean the model produced no overlapping proposal. Examples from the saved outputs:

| Primary identity | Candidate best box IoU | Interpretation limited to the saved output |
| --- | ---: | --- |
| `new_000057:7` | 0.49919 | Overlapping proposal lies just below the unchanged 0.50 matching cutoff. |
| `new_000008:35` | 0.49038 | Overlapping proposal is below the cutoff. |
| `new_000056:14` | 0.48411 | Overlapping proposal is below the cutoff. |
| `new_000009:56` | 0 | No retained candidate output overlaps the frozen box at this operating point. |

These overlap measurements do not decide which physical boundary is correct. Likewise, unmatched outputs include duplicate detections and extent/reference disagreements; they have not all been individually adjudicated as background errors. New annotation corrections should continue to follow source pixels and the agreed front-face definition, while model decisions remain tied to a stable comparison.

## Returned epoch-30 trainer metrics

These trainer metrics use the original 664-front validation labels and aggregate metric procedure. They are separate from the fixed operating point above.

| Metric | Fresh r10 control | r10 candidate |
| --- | ---: | ---: |
| Box precision | 0.96820 | 0.95405 |
| Box recall | 0.91699 | 0.92169 |
| Box mAP50 | 0.93920 | 0.93997 |
| Box mAP50-95 | 0.57519 | 0.51529 |
| Pose precision | 0.98584 | 0.98127 |
| Pose recall | 0.93825 | 0.93373 |
| Pose mAP50 | 0.96043 | 0.96093 |
| Pose mAP50-95 | 0.96036 | 0.96068 |

Sources: [control CSV](checkpoint-receipts/reference/fit/results.csv) and [candidate CSV](checkpoint-receipts/candidate/fit/results.csv). Slight mAP50 gains coexist with more fixed-point misses and a lower box mAP50-95, so aggregate mAP50 alone is insufficient for promotion.

## Reproduce from the saved evidence

In a repository checkout with the preserved original PNGs and r10 source files, replay the saved predictions without loading any model weights:

```bash
pr79_root="$(git rev-parse --show-toplevel)"
pr79_audit="$pr79_root/results/evaluation/pr79-r10-20261119"
python3 -B "$pr79_audit/measure_pr79.py" \
  --root "$pr79_root" \
  --pr35 "$pr79_audit/inference/eval-pr35/predictions.jsonl" \
  --pr35-manifest "$pr79_audit/checkpoint-receipts/pr35/checkpoint-manifest.json" \
  --reference "$pr79_audit/inference/eval-reference/predictions.jsonl" \
  --reference-manifest "$pr79_audit/checkpoint-receipts/reference/checkpoint-manifest.json" \
  --candidate "$pr79_audit/inference/eval-candidate/predictions.jsonl" \
  --candidate-manifest "$pr79_audit/checkpoint-receipts/candidate/checkpoint-manifest.json" \
  --output /path/to/new-pr79-comparison.json
cmp "$pr79_audit/fixed-comparison.json" /path/to/new-pr79-comparison.json
```

[Replay verification](replay-verification.json) confirms the saved prediction and receipt files reproduce the full comparison byte for byte. `run_pr79_inference.py` accepts the three original full checkpoint bundles and a fresh output directory for new inference. `audit_return.py` accepts the PR79 archive and its extracted bundle; for its exact training-source checks, use preserved source revision `bae1ed46bf3337956dbd60ac85aa5bcc759bbbcd` (or its unchanged source at delivery commit `8230b8d592c792002bf97405d950264d22292a47`) rather than a later checkout whose training scripts have evolved.

This is one seed and recipe on repeatedly used development photographs. It does not establish performance on unseen capture sessions, millimetre positioning, or robot grasp success. Model confidence is not a calibrated probability of physical visibility or pixel accuracy. No model promotion follows from this audit.
