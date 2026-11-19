# PR76 r8 result audit

**Keep PR35 as the selected model.** The returned r8 candidate has six more scored misses and six more unmatched outputs than its fresh control on the unchanged 666-front primary reference. It gains four both-corner successes within five pixels but loses three within ten pixels. Against retained PR35 it adds eight misses and nine unmatched outputs. These results do not support promotion.

This audit was recovered on 19 November 2026. It reruns inference and scoring, not training. It does not change annotations, model selection, frozen references, or the existing notebook.

## What was actually trained

The [PR76 return](https://github.com/Siuuuers/dl/pull/76) is pinned to head `b5545d2bfbfb908a10d82196ffe690dfb7fb3039`. The returned ZIP is 39,782,118 bytes with SHA-256 `02442e197cf619768a3d14d933ff69841aa99387d744686bde0f6e1dfe5cddc3`. Its ZIP integrity check, exact member inventory, and all 12 indexed member hashes and sizes passed.

The return records training source `f870296f13f7c40f03786bc421aeefb9866be6de` and [r8 protocol](../../../experiments/front-review-2026-11-19-r8.json), SHA-256 `96426b9d0ba7a07b902698e320a82449c051c6790c4bac25dda709d2f181b7d3`. Every recorded training-source file hash matches the preserved source used here. The completed-checkpoint validator verified both bundles, including model bytes, CSV, training receipts, staging manifests and training logs. Every staged image/label source hash was checked against local preserved input bytes.

Both arms completed 30 epochs, 1,050 batches, 4,110 image presentations and 78 recorded optimizer steps on an NVIDIA L4. They independently started from the recorded original epoch173 checkpoint, used the fixed seed-46 recipe, and selected final epoch 30. The original initialization checkpoint bytes are not contained in this return and were not downloaded again; its recorded identity and recipe bindings were checked.

| Actual PR76 training input | Control | Candidate |
| --- | ---: | ---: |
| Training photographs | 137 | 137 |
| Empty training scenes | 13 | 13 |
| Labelled fronts | 3,771 | 3,770 |
| Locatable endpoint slots | 6,841 | 6,847 |
| Original validation photographs / fronts | 20 / 664 | 20 / 664 |

Labels differ on **59 training photographs**. All **64 held training photographs** use identical PR35 supervision in both arms. All 157 original PNGs and original validation labels remain identical between arms. The later interrupted review's reported 35-photo hold is a different, subsequent review state; it must not be attributed to the completed PR76 experiment.

[Verified return](verified-return.json) records these checks. Holding inherited labels controls the treatment difference; it does not establish that those labels are physically correct.

## Fresh inference and fixed references

All three checkpoints were freshly inferred in one CPU environment with PyTorch 2.14.1+cpu and Ultralytics 8.4.182. [Package pins](inference-pip-freeze.txt), [commands](inference/commands.json) and the saved prediction headers record the actual runtime. The operating point is unchanged: box confidence 0.13, endpoint confidence 0.50, ordinary NMS at IoU 0.70, matching IoU 0.50, FP32, input size 1280 and four CPU threads. No threshold, seed, epoch or NMS search was performed.

The scorer code is unchanged. [Measurement adaptation](measurement-adaptation.patch) changes only experiment/checkpoint identities and corresponding description strings. [Full fixed comparison](fixed-comparison.json) contains denominators, per-image scores, match traces, gained/lost identities and both independently versioned references. [Compact summary](summary.json) contains the main table values.

### Primary reference: 666 fronts

The frozen reference has 20 photographs, 17 nonempty scenes, three empty scenes, 1,227 locatable endpoints and 105 unavailable endpoint slots. Its manifest SHA-256 is `0de64be3797dce2e43d4cb98d09620eab969bb95725ed9e75682dcfcd53c7f96`.

| Model | Matched fronts | Scored misses | Unmatched outputs | Duplicate subset | Both corners <=5 px | Both <=10 px |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Retained PR35 | 648/666 | 18 | 80 | 32 | 533/584 | 582/584 |
| Fresh r8 control | 646/666 | 20 | 83 | 29 | 531/584 | 583/584 |
| PR76 r8 candidate | 640/666 | 26 | 89 | 28 | 535/584 | 580/584 |

The pair denominator is the **584 reference fronts with two locatable corners**. Eligible fronts that were missed remain failures in that denominator. It is not complete-scene accuracy. The full detection denominator remains 666, including partial fronts and fronts with unavailable corners.

| Locatable reference corners | Fronts | PR35 misses | Control misses | Candidate misses |
| --- | ---: | ---: | ---: | ---: |
| Zero | 23 | 10 | 10 | 13 |
| One | 59 | 7 | 9 | 10 |
| Two | 584 | 1 | 1 | 3 |

All three models return zero outputs on the three empty scenes. Miss-free nonempty scenes are 6/17 for PR35 and 4/17 for each fresh r8 arm. No model produces a fully error-free nonempty scene under the fixed matcher. Candidate and control each return 729 outputs; six fewer matches account for the candidate's six additional unmatched outputs.

Retained PR35's ordered decoded prediction arrays exactly match its earlier PR74 audit on all 20 photographs, as recorded in [consistency check](pr35-replay-consistency.json). The fresh r8 control differs from retained PR35. Use the fresh control for the within-experiment annotation comparison rather than assuming the control and old checkpoint are identical.

### Historical reference: 663 fronts

The separate historical reference retains manifest SHA-256 `bbc1dca71f7a3e0fe2f54a4391962e562c3353a62533eb233b697819edcc64b1`.

| Model | Matches | Scored misses | Unmatched outputs |
| --- | ---: | ---: | ---: |
| Retained PR35 | 648/663 | 15 | 80 |
| Fresh r8 control | 646/663 | 17 | 83 |
| PR76 r8 candidate | 641/663 | 22 | 88 |

Both-corner scores are the same as in the primary table. The candidate adds five misses and five unmatched outputs relative to its fresh control. These references share inherited labels and are not independent physical certification.

## What the changed matches mean

Compared with the fresh control, the primary comparison gains `new_000008:1` and `new_000009:53`, and loses `new_000008:33`, `new_000009:56`, `new_000010:15`, `new_000010:32`, `new_000011:61`, `new_000056:15`, `new_000057:11` and `new_000090:42`. These row numbers belong to the frozen primary reference, not necessarily the latest review PDF.

A scored miss means no qualifying one-to-one box match. It does not always mean the model returned no overlapping proposal. For example:

| Primary-reference identity | Candidate best IoU | Candidate box confidence | Saved-output observation |
| --- | ---: | ---: | --- |
| `new_000010:15` | 0.49833 | 0.88409 | Strong overlapping proposal, just below the unchanged 0.50 matching cutoff. |
| `new_000057:11` | 0.49296 | 0.86068 | Strong overlapping proposal below the cutoff. |
| `new_000010:32` | 0.48630 | 0.21479 | Overlapping proposal below the cutoff. |
| `new_000009:56` | 0 | Not applicable | No retained candidate output overlaps this frozen box at this operating point. |

These are measured overlap observations. They do not decide which physical boundary is correct. Likewise, unmatched outputs include duplicates and reference/extent disagreements and have not all been independently adjudicated as background errors. Correcting review labels and assessing a model against a frozen answer key are separate tasks.

## Returned epoch-30 validation metrics

These metrics use the original 664-front training validation labels and the trainer's aggregate metric procedure. They are distinct from the fixed operating point above.

| Metric | Fresh r8 control | r8 candidate |
| --- | ---: | ---: |
| Box precision | 0.96808 | 0.95684 |
| Box recall | 0.91365 | 0.92018 |
| Box mAP50 | 0.93881 | 0.93898 |
| Box mAP50-95 | 0.57249 | 0.56500 |
| Pose precision | 0.98807 | 0.98002 |
| Pose recall | 0.93825 | 0.93675 |
| Pose mAP50 | 0.96042 | 0.96056 |
| Pose mAP50-95 | 0.96021 | 0.96042 |

Sources: [control CSV](checkpoint-receipts/reference/fit/results.csv), [candidate CSV](checkpoint-receipts/candidate/fit/results.csv). A small aggregate mAP gain does not establish fewer operational misses or more reliable positioning.

## Reproduce without loading weights

In the repository checkout with the preserved original PNGs, run the saved predictions through the unchanged scorer:

```bash
pr76_root="$(git rev-parse --show-toplevel)"
pr76_audit="$pr76_root/results/evaluation/pr76-recovery-20261119"
python3 -B "$pr76_audit/measure_pr76.py" \
  --root "$pr76_root" \
  --pr35 "$pr76_audit/inference/eval-pr35/predictions.jsonl" \
  --pr35-manifest "$pr76_audit/checkpoint-receipts/pr35/checkpoint-manifest.json" \
  --reference "$pr76_audit/inference/eval-reference/predictions.jsonl" \
  --reference-manifest "$pr76_audit/checkpoint-receipts/reference/checkpoint-manifest.json" \
  --candidate "$pr76_audit/inference/eval-candidate/predictions.jsonl" \
  --candidate-manifest "$pr76_audit/checkpoint-receipts/candidate/checkpoint-manifest.json" \
  --output /path/to/new-pr76-comparison.json
cmp "$pr76_audit/fixed-comparison.json" /path/to/new-pr76-comparison.json
```

The bundled receipts include the exact training logs required by the checkpoint validator. [Replay verification](replay-verification.json) confirms this saved evidence reproduces the comparison byte for byte. For new inference, `run_pr76_inference.py` accepts the original full checkpoint bundles and a fresh output directory.

This is one seed and training recipe on repeatedly used development photographs. It does not establish performance on independent capture sessions, millimetre accuracy, or robot grasp success. A model's confidence is not a calibrated probability of physical visibility or pixel accuracy. No model promotion follows from this audit.
