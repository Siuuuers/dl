# PR72 r5 result audit

**Retain PR35 as the selected model.** The returned r5 candidate gains six both-corner successes at five pixels versus its fresh control, but loses ten matched fronts and produces 22 more unmatched outputs. Its ten-pixel both-corner score also decreases. The fixed comparison does not support promotion. No training, threshold search, or model replacement was performed during this audit.

A **scored miss** means that a reference front has no one-to-one box match at the preregistered IoU threshold of 0.50. It does not necessarily mean that the network produced no proposal for the visible material. Likewise, unmatched outputs are not all independently verified background detections. The unchanged references and saved matching traces make those distinctions reviewable.

## Return provenance

[PR72](https://github.com/Siuuuers/dl/pull/72), head `1a97010569e20bfb46bc060e3b00483b489ee775`, contains the returned r5 training archive. The audit downloaded its exact committed bytes and verified:

- Archive size: **39,779,326 bytes**.
- SHA-256: `cb89cee476ec45ac37eb974ee7746e8fb190395af72735e3449e3d4c73e6027a`.
- ZIP integrity, exact member inventory, and all 12 indexed member hashes and sizes.
- Both strict completed checkpoint bundles, their receipts, CSVs and logs.
- Matching training source, runtime, data proposal and shared recipe across arms.
- The exact training source [`7685802bc76d6f76f5afd5067833ed19284222dd`](https://github.com/Siuuuers/dl/tree/7685802bc76d6f76f5afd5067833ed19284222dd), checked out independently with clean tracked state.
- Every recorded training-source hash, the r5 candidate reconstruction, all 157 original image and label hashes, original epoch173 checkpoint bytes, and retained PR35 checkpoint bytes.
- Unchanged primary and historical reference manifests.

[verified-return.json](verified-return.json) records the checks. [returned-gpu-results-manifest.json](returned-gpu-results-manifest.json) preserves the original return index. Original receipts and logs are under [checkpoint-receipts](checkpoint-receipts). The recorded GPU execution was checked for consistency with the files and frozen code; training was not repeated.

Each arm completed **30 epochs, 1,050 batches, 4,110 image presentations and 78 recorded optimizer updates** on NVIDIA L4. Both independently start from the original epoch173 checkpoint, with seed 46, batch 4, image size 1280, FP32, AdamW and the fixed learning-rate schedule. Final epoch 30 is selected; no best-epoch search is used.

Both arms retain 137 training photos, including 13 empty scenes. Control supervision has **3,771 fronts and 6,841 locatable corners**; r5 supervision has **3,788 fronts and 6,849 locatable corners**. Labels differ on 94 training photos. The whole unresolved `new_000023` photo uses exact PR35 labels in both arms. This controlled hold does not certify those inherited labels.

## Original training-validation metrics

These are the returned **epoch-30 CSV metrics on the original 20-photo, 664-front validation labels**, not the fixed operating-point measurements in the next section.

| Metric | Fresh control | r5 candidate |
| --- | ---: | ---: |
| Box precision | 0.96808 | 0.96618 |
| Box recall | 0.91365 | 0.90365 |
| Box mAP50 | 0.93881 | 0.93103 |
| Box mAP50-95 | 0.57249 | 0.55286 |
| Pose precision | 0.98807 | 0.97757 |
| Pose recall | 0.93825 | 0.93373 |
| Pose mAP50 | 0.96042 | 0.96057 |
| Pose mAP50-95 | 0.96021 | 0.96050 |

Sources: [control CSV](checkpoint-receipts/reference/fit/results.csv) and [candidate CSV](checkpoint-receipts/candidate/fit/results.csv). A small pose mAP increase alone does not establish better complete-image cassette detection.

## Fresh fixed comparison

All three checkpoints were freshly inferred on the same 20 original PNGs in the same CPU environment: Python 3.12.14, PyTorch 2.14.1+cpu, torchvision 0.29.1+cpu, Ultralytics 8.4.182, NumPy 2.4.6, Pillow 11.3.0 and OpenCV 5.0.0.93. Full package versions are in [inference-pip-freeze.txt](inference-pip-freeze.txt).

The operating point is unchanged: **box confidence 0.13, endpoint confidence 0.50, matching IoU 0.50, NMS IoU 0.70, input size 1280, ordinary NMS, FP32 and four CPU threads**. Distances use original-image pixels. This is a model comparison, not a GPU or production-latency benchmark.

The pinned [r5 protocol](https://github.com/Siuuuers/dl/blob/7685802bc76d6f76f5afd5067833ed19284222dd/experiments/front-review-2026-11-19-r5.json) has SHA-256 `25ed684fd677ba4863f5cd605f48e3b9e1bc78893ae7c4267616c6779be1cd1c`. The scorer is unchanged from the PR70 audit; [measurement-adaptation.patch](measurement-adaptation.patch) shows the protocol/checkpoint identity changes only. [fixed-comparison.json](fixed-comparison.json) contains complete scores, per-image traces and gained/lost identities for both references.

### Primary reference: 666 fronts

The unchanged manifest is `0de64be3797dce2e43d4cb98d09620eab969bb95725ed9e75682dcfcd53c7f96`: 20 photos, 17 nonempty scenes, three empty scenes, 1,227 locatable endpoints and 105 missing endpoint slots.

| Model | Matched fronts | Scored misses | Unmatched outputs | Duplicate subset | Both corners <=5 px | Both <=10 px | Miss-free nonempty scenes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Retained PR35 | 648/666 | 18 | 80 | 32 | 533/584 | 582/584 | 6/17 |
| Fresh r5 control | 646/666 | 20 | 83 | 29 | 531/584 | 583/584 | 4/17 |
| r5 candidate | 636/666 | 30 | 105 | 27 | 537/584 | 580/584 | 3/17 |

Candidate versus fresh control: **10 more scored misses, 22 more unmatched outputs, six more five-pixel pair successes, and three fewer ten-pixel pair successes**. Candidate versus retained PR35: 12 more misses, 25 more unmatched outputs, four more five-pixel pair successes and two fewer ten-pixel pair successes.

The full front denominator remains **666**, including partial fronts and fronts without any locatable corner. The separate pair denominator remains **584 fronts with two locatable reference corners**, including eligible missed fronts as failures. It is not a complete-scene accuracy rate.

| Reference corner availability | Fronts | PR35 misses | Control misses | Candidate misses |
| --- | ---: | ---: | ---: | ---: |
| No locatable corner | 23 | 10 | 10 | 13 |
| One locatable corner | 59 | 7 | 9 | 14 |
| Two locatable corners | 584 | 1 | 1 | 3 |

The candidate returns valid endpoint coordinates meeting the endpoint-confidence criterion for **1,206/1,227 locatable reference endpoints**, versus 1,216/1,227 for control and 1,218/1,227 for PR35. At five pixels, individual endpoint successes are 1,145, 1,150 and 1,149 respectively. Thus the modest pair-score gain does not mean every localization measure improved.

All three models return zero outputs on the three empty images. None achieves an error-free nonempty scene under the fixed matching rule.

### Historical reference: 663 fronts

The separately frozen manifest is `bbc1dca71f7a3e0fe2f54a4391962e562c3353a62533eb233b697819edcc64b1`: 1,227 locatable endpoints, 99 missing slots, and 584 two-corner fronts.

| Model | Matched fronts | Scored misses | Unmatched outputs | Miss-free nonempty scenes |
| --- | ---: | ---: | ---: | ---: |
| Retained PR35 | 648/663 | 15 | 80 | 7/17 |
| Fresh r5 control | 646/663 | 17 | 83 | 5/17 |
| r5 candidate | 637/663 | 26 | 104 | 3/17 |

Five- and ten-pixel pair scores are unchanged from the primary table. The candidate loses nine matches and adds 21 unmatched outputs versus control. These two references share inherited labels; agreement between them is not independent physical verification. Neither the r5 662-front review nor the later r6 review was substituted into either score.

## What the scored losses mean

Candidate versus control gains primary-reference matches at **008:1 and 009:53** and loses them at **008:33, 008:34, 009:56, 010:15, 010:31, 010:32, 011:61, 056:15, 057:11, 089:43, 090:41 and 090:42**. The notation abbreviates `new_000NNN:row`; rows refer to the frozen primary reference.

Every lost reference front has best candidate box overlap below 0.50. Thus these losses are not caused solely by a qualifying candidate box being consumed by another one-to-one assignment. They still include different situations:

| Reference front | Control best IoU | Candidate best IoU | Confidence of candidate best-overlap output | Supported conclusion |
| --- | ---: | ---: | ---: | --- |
| 010:15 | 0.59447 | 0.45604 | 0.88554 | A strong overlapping proposal remains, but its extent fails the fixed match. |
| 057:11 | 0.59176 | 0.48040 | 0.87791 | A strong overlapping proposal remains; do not describe this as no prediction. |
| 008:33 | 0.60125 | 0.09743 | 0.13989 | The retained outputs have much weaker coverage of this reference extent. |
| 009:56 | 0.54728 | 0.00000 | Not applicable | No saved candidate output overlaps this reference box at the fixed operating point. |

This audit does not adjudicate whether each changed model extent or inherited reference box is physically correct. That requires the original-image review. Keeping the reference fixed prevents changing the answer key to favor the newer model.

At five pixels, 14 pair identities become successful and eight cease to be successful, giving the net gain of six. At ten pixels, the lost pair identities are **008:10, 010:15 and 057:11** with no gains. The last two also fail through box association.

The candidate has 741 total predictions versus 729 for control. Twelve additional outputs plus ten fewer matches explain the 22 additional unmatched outputs. The duplicate subset decreases from 29 to 27; other unmatched outputs increase from 54 to 78. No claim is made that all 105 candidate unmatched outputs are background errors.

## Reproduce from saved predictions

The saved-prediction replay needs Python 3.11+, Git and Git LFS. It does not load weights, require PyTorch, or run training. After checking out the frozen source commit and obtaining its validation PNGs, run the following with `pr72_source` set to that checkout and `pr72_evidence` set to this directory:

```bash
python -B "$pr72_evidence/measure_pr72.py" \
  --root "$pr72_source" \
  --pr35 "$pr72_evidence/eval-pr35/predictions.jsonl" \
  --pr35-manifest "$pr72_evidence/checkpoint-receipts/pr35/checkpoint-manifest.json" \
  --reference "$pr72_evidence/eval-reference/predictions.jsonl" \
  --reference-manifest "$pr72_evidence/checkpoint-receipts/reference/checkpoint-manifest.json" \
  --candidate "$pr72_evidence/eval-candidate/predictions.jsonl" \
  --candidate-manifest "$pr72_evidence/checkpoint-receipts/candidate/checkpoint-manifest.json" \
  --output /path/to/new-fixed-comparison.json
cmp "$pr72_evidence/fixed-comparison.json" /path/to/new-fixed-comparison.json
```

The delivered copies of the script, receipts and predictions were replayed into a fresh output and reproduced the report **byte for byte**. The evaluator's individual `eval-*` summaries refer to original 664-front labels; use `fixed-comparison.json` for the 666/663 result.

For fresh inference, [run_pr72_inference.py](run_pr72_inference.py) accepts a frozen source checkout, Python interpreter, a new output directory and the three checkpoint manifests. Obtain full weights from the exact original archives first:

| Archive | Immutable source | SHA-256 |
| --- | --- | --- |
| PR35; use its candidate arm | [PR35 ZIP](https://raw.githubusercontent.com/Siuuuers/dl/8e8bb99cabcb44ed0ec0acd14db911aa44013fd8/results/gpu/baomai-reviewed-annotations-seed46-l4-001/baomai-reviewed-annotations-seed46-results.zip) | `df53683ccf6f266a605b2844f42a98e808f75686cfab09f14d680c9d061e8a91` |
| PR72; use both arms | [PR72 ZIP](https://raw.githubusercontent.com/Siuuuers/dl/1a97010569e20bfb46bc060e3b00483b489ee775/results/gpu/baomai-front-review-r5-l4-001/baomai-front-review-r5-seed46-v4-results.zip) | `cb89cee476ec45ac37eb974ee7746e8fb190395af72735e3449e3d4c73e6027a` |

The saved inference commands and exit codes are in [inference-commands.json](inference-commands.json). CPU inference is used only to compare all three checkpoints under one runtime.

## Decision and limits

The result supports retaining PR35 and continuing the evidence-based annotation review. It does not prove that every r5 edit is wrong, that the unchanged gold boxes are all physically correct, or that PR35 is ready for deployment. It also does not guarantee that another correction pass will improve the model.

This is one seed and one fixed recipe on repeatedly used development scenes. New camera sessions and application-level error tolerances are still needed before claiming dependable robot positioning. The next notebook should be understood as a controlled test of revised supervision, with its labels, holds and references pinned before execution.
