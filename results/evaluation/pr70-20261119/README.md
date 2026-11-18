# PR70 r4 result audit

**The fixed comparison does not support promoting the r4 candidate. Retain PR35 as the selected model.** The candidate improves the 5-pixel both-corner score, but has poorer matched-front coverage and more unmatched outputs than the fresh control and retained PR35. Some counted losses are box-extent disagreements; the evidence does not show that r4 is universally worse.

**A scored miss means no one-to-one match at the predeclared IoU of 0.50. It does not necessarily mean the model produced no proposal for the visible material.** This distinction matters for the two new losses with two locatable corners, `new_000010:15` and `new_000057:11`, and several partial fronts. Their model or reference geometry needs physical adjudication before either is declared correct. The fixed 666/663 references remain unchanged throughout this audit.

## Verified execution and inputs

[PR70](https://github.com/Siuuuers/dl/pull/70), head `df279a3aec025249db15fd5394ae051492c3f508`, contains a results-only return. The 39,778,742-byte ZIP has SHA-256 `a60215759e9f800db687a11346c5edb08d363cca8dad61232a9553ec610f641b`. ZIP integrity, all 12 indexed member hashes and sizes, both strict checkpoint bundles, and the matched source/runtime/settings checks passed. [verified-return.json](verified-return.json) records the independent audit; the original receipts and logs are preserved in [checkpoint-receipts](checkpoint-receipts).

Both YOLO26s pose arms completed all 30 epochs on an NVIDIA L4, with 1,050 batches, 4,110 image presentations and 78 recorded optimizer updates. Each starts independently from the same epoch173 checkpoint, with seed 46, batch 4, image size 1280, FP32 and the fixed AdamW recipe. Reference supervision contains 3,771 training fronts and 6,841 locatable corners; candidate supervision contains 3,789 and 6,853. The candidate changes 94 training photographs relative to the control. The unresolved `new_000014` photograph retains its whole PR35 label file in both arms. Training validation remains the original 20 photographs and 664 fronts.

The audit uses an independent source snapshot from [`c8d8f30a5770c3cee25d0bd431df4f565ca5471f`](https://github.com/Siuuuers/dl/tree/c8d8f30a5770c3cee25d0bd431df4f565ca5471f), including the original r4 verifier. The notebook at `dba4d685fa7a62dbda1ec32e5f40a44b42bd6054` pins that source. All 157 original PNG hashes and the original epoch173 bytes were verified. Later r5 source or label edits cannot change this recorded comparison.

### What the training validation logs show

These are **epoch-30 metrics on the original 664-front validation labels**, not the fixed operating-point counts below. No best epoch was selected.

| Metric | Fresh control | r4 candidate |
|---|---:|---:|
| Box precision | 0.96808 | 0.96612 |
| Box recall | 0.91365 | 0.90177 |
| Box mAP50 | 0.93881 | 0.93277 |
| Box mAP50–95 | 0.57249 | 0.55610 |
| Pose mAP50 | 0.96042 | 0.96071 |
| Pose mAP50–95 | 0.96021 | 0.96062 |

Sources: [control CSV](checkpoint-receipts/reference/fit/results.csv) and [candidate CSV](checkpoint-receipts/candidate/fit/results.csv). The small pose mAP increase does not establish better complete-image front detection.

## Fresh fixed comparison

All three checkpoints were freshly inferred on the same 20 original development PNGs in one CPU runtime: Python 3.12.14, PyTorch 2.14.1+cpu, torchvision 0.29.1+cpu, Ultralytics 8.4.182, NumPy 2.4.6, Pillow 11.3.0 and OpenCV 5.0.0.93. All use four CPU threads, FP32, ordinary NMS, image size 1280, box confidence 0.13, NMS IoU 0.70, endpoint confidence 0.50 and matching IoU 0.50. Distances are measured in original-image pixels.

The [r4 protocol](https://github.com/Siuuuers/dl/blob/c8d8f30a5770c3cee25d0bd431df4f565ca5471f/experiments/front-review-2026-11-18-r4.json) has SHA-256 `409e853e49d755a19becd0778272eda24b8dfabbbb2c77554b5e24dfb7a48928`. The existing scoring logic is unchanged; [measurement-adaptation.patch](measurement-adaptation.patch) shows only the updated experiment and checkpoint identities. [fixed-comparison.json](fixed-comparison.json) contains both references, every model score, matching traces and gained/lost identities.

### Primary reference: 666 fronts

This reference has 20 images, 17 nonempty images, 3 empty images, 1,227 locatable endpoints and 105 missing endpoints. Its manifest SHA-256 is `0de64be3797dce2e43d4cb98d09620eab969bb95725ed9e75682dcfcd53c7f96`.

| Model | Matched | Scored misses | Unmatched outputs | Duplicate subset | Both corners ≤5 px | Both ≤10 px | Miss-free nonempty scenes |
|---|---:|---:|---:|---:|---:|---:|---:|
| Retained PR35 | 648 | 18 | 80 | 32 | 533/584 | 582/584 | 6/17 |
| Fresh r4 control | 646 | 20 | 83 | 29 | 531/584 | 583/584 | 4/17 |
| r4 candidate | 636 | 30 | 107 | 28 | 541/584 | 580/584 | 3/17 |

Candidate versus fresh control: **10 more scored misses, 24 more unmatched outputs, 10 more 5-pixel pair successes and 3 fewer 10-pixel pair successes.** Candidate versus PR35: 12 more scored misses, 27 more unmatched outputs, 8 more 5-pixel pair successes and 2 fewer 10-pixel pair successes.

The pair denominator is the **584 fronts with two locatable reference corners**, including misses in that denominator. It is not all 666 fronts and is not a complete-scene accuracy figure. All three models return zero outputs on the three empty images. None produces an error-free nonempty image under this fixed association rule.

| Reference corner availability | Fronts | PR35 misses | Control misses | Candidate misses |
|---|---:|---:|---:|---:|
| Zero locatable corners | 23 | 10 | 10 | 13 |
| One locatable corner | 59 | 7 | 9 | 14 |
| Two locatable corners | 584 | 1 | 1 | 3 |

The candidate returns coordinates meeting the endpoint criterion for 1,206/1,227 locatable reference endpoints, versus 1,216/1,227 for control and 1,218/1,227 for PR35. Confidence on a prediction of an unlocatable reference corner cannot establish its physical correctness.

### Required historical reference: 663 fronts

This separate reference has 1,227 locatable endpoints and 99 missing endpoints: 20 zero-corner, 59 one-corner and 584 two-corner fronts. Its manifest SHA-256 is `bbc1dca71f7a3e0fe2f54a4391962e562c3353a62533eb233b697819edcc64b1`.

| Model | Matched | Scored misses | Unmatched outputs | Miss-free nonempty scenes |
|---|---:|---:|---:|---:|
| PR35 | 648 | 15 | 80 | 7/17 |
| Fresh control | 646 | 17 | 83 | 5/17 |
| r4 candidate | 637 | 26 | 106 | 3/17 |

The 5/10-pixel pair counts are unchanged from the primary table. Candidate versus control is 9 more scored misses and 23 more unmatched outputs. These references share inherited labels; agreement in their rankings is not independent physical verification of every boundary. The separate 662-front reviewed reference and r5 labels were not substituted into either comparison.

## What changed in the images

Relative to control, primary-reference matches are gained at **008:1 and 009:53**, and lost at **008:33, 008:34, 009:56, 010:15, 010:31, 010:32, 011:61, 056:15, 057:11, 089:43, 090:41 and 090:42**. IDs abbreviate `new_000NNN:row`; row numbers belong to the immutable primary reference.

Both gains adjust boxes for material already proposed by control. Every scored loss has best candidate overlap below 0.50, so these losses are not cases where an otherwise qualifying candidate box was merely consumed by another one-to-one assignment. They still have different physical interpretations:

| Case | Control best IoU | Candidate best IoU | Candidate box confidence | Evidence and interpretation |
|---|---:|---:|---:|---|
| 010:15 | 0.59447 | 0.45445 | 0.89121 | [Original/gold/output crop](figures/new_000010-rows-15.png). A strong proposal remains; changed box extent crosses the matching cutoff. Model-versus-gold correctness needs physical adjudication. |
| 057:11 | 0.59176 | 0.48051 | 0.87675 | [Crop](figures/new_000057-rows-11.png). The visible front is still proposed; box extent/association causes the scored loss. Do not describe it as an absent prediction. |
| 008:33 | 0.60125 | 0.09990 | 0.17832 for the best-overlap neighboring proposal | [Crop](figures/new_000008-rows-33-34.png). The partial image-edge front is visible; candidate's nearby retained proposal follows material below it. This is a visually supported missed front at the fixed setting. |

Several other losses contract toward exposed material beside the thumb or gripper: [010:31/32](figures/new_000010-rows-31-32.png), [089:43](figures/new_000089-rows-43.png), and 056:15/090:41/42. A valid axis-aligned box may span an occluder when separated visible pieces belong to the same front. Therefore a broad gold box is not automatically wrong, and a smaller candidate box is not automatically correct. The gripper-aperture cases [009:56](figures/new_000009-rows-56.png) and [011:61](figures/new_000011-rows-61.png) also require surface/identity care. Candidate has no overlapping retained output for 009:56; the historical reference omits that row.

At 10 pixels, the three pair successes lost versus control are 010:15, 057:11 and 008:10. The first two fail through box matching. 008:10 remains matched, with right-corner error increasing from 3.43 to 15.00 pixels. At 5 pixels, 17 pair identities improve and 7 worsen, yielding the net gain of 10.

### Unmatched outputs

The +24 count versus control comes from 14 additional total predictions and 10 fewer matches. The duplicate subset decreases 29→28; other unmatched outputs increase 54→79. Confidence is not the main separator: unmatched outputs in [0.13,0.25) rise 29→31, while those at or above 0.25 rise 54→76. These are descriptive bins of the saved predictions, not a new threshold sweep.

Per-image extra-output changes are: **090 +9, 089 +8, 010 +6, 087 +5, 056 +3, 008/057/088 +2 each; 009 −7, 011 −2, and 054/055/058/059 −1 each**. They sum to +24. [fixed-output-diagnostics.json](fixed-output-diagnostics.json) contains every unmatched output, its confidence, nearest reference overlap, duplicate flag and complete per-image counts.

Some are demonstrable background detections: [090's keyboard crop](figures/new_000090-keyboard.png) has **five candidate proposals versus two control proposals**, and [089's top-rack crop](figures/new_000089-top-rack.png) has **two versus one**. Other unmatched outputs include partial-front extent disagreements and overlapping fragments; all 107 have not been individually certified as background errors. [figure-provenance.json](figure-provenance.json) binds the original image hashes and exact crop coordinates.

## Reproduce from saved predictions

This replay needs Python 3.11+, Git and Git LFS. It does not load model weights or require PyTorch, Ultralytics or a GPU. The preserved checkpoint receipts bind each saved prediction file to the already verified returned checkpoint. Their original relative paths are retained; weight bytes remain in the original result archives.

From the repository root containing this evidence directory:

```bash
set -e
pr70_evidence="$PWD/results/evaluation/pr70-20261119"
pr70_replay="$(mktemp -d "${TMPDIR:-/tmp}/dl-pr70-replay.XXXXXX")"
git init "$pr70_replay/source-c8d8f30"
git -C "$pr70_replay/source-c8d8f30" remote add origin https://github.com/Siuuuers/dl.git
GIT_LFS_SKIP_SMUDGE=1 git -C "$pr70_replay/source-c8d8f30" fetch --depth 1 origin c8d8f30a5770c3cee25d0bd431df4f565ca5471f
GIT_LFS_SKIP_SMUDGE=1 git -C "$pr70_replay/source-c8d8f30" checkout --detach FETCH_HEAD
git -C "$pr70_replay/source-c8d8f30" lfs pull --include='data/images/val/*.png' --exclude=''

python -B "$pr70_evidence/measure_pr70.py" \
  --root "$pr70_replay/source-c8d8f30" \
  --pr35 "$pr70_evidence/eval-pr35/predictions.jsonl" \
  --pr35-manifest "$pr70_evidence/checkpoint-receipts/pr35/checkpoint-manifest.json" \
  --reference "$pr70_evidence/eval-reference/predictions.jsonl" \
  --reference-manifest "$pr70_evidence/checkpoint-receipts/reference/checkpoint-manifest.json" \
  --candidate "$pr70_evidence/eval-candidate/predictions.jsonl" \
  --candidate-manifest "$pr70_evidence/checkpoint-receipts/candidate/checkpoint-manifest.json" \
  --output "$pr70_replay/fixed-comparison.json"
cmp "$pr70_evidence/fixed-comparison.json" "$pr70_replay/fixed-comparison.json"
```

The delivery's own script and receipt copies were replayed once into a fresh output and reproduced `fixed-comparison.json` byte for byte. The `eval-*` directories also retain the evaluator's initial summaries against the **original 664-front labels**; use `fixed-comparison.json` for the 666/663 results above.

### Optional fresh inference

The existing [run_pr70_inference.py](run_pr70_inference.py) accepts `--root`, `--python`, a new `--output` directory, and the three `--*-manifest` paths. It enforces the frozen source/protocol/model identities and runs the exact three CPU commands. Fresh inference requires the full original archives, including weights:

| Artifact | Immutable source | SHA-256 |
|---|---|---|
| PR35; use its **candidate** arm | [PR35 ZIP](https://raw.githubusercontent.com/Siuuuers/dl/8e8bb99cabcb44ed0ec0acd14db911aa44013fd8/results/gpu/baomai-reviewed-annotations-seed46-l4-001/baomai-reviewed-annotations-seed46-results.zip) | `df53683ccf6f266a605b2844f42a98e808f75686cfab09f14d680c9d061e8a91` |
| PR70; use both arms | [PR70 ZIP](https://raw.githubusercontent.com/Siuuuers/dl/df279a3aec025249db15fd5394ae051492c3f508/results/gpu/baomai-front-review-r4-l4-001/baomai-front-review-r4-seed46-v3-results.zip) | `a60215759e9f800db687a11346c5edb08d363cca8dad61232a9553ec610f641b` |

Install the recorded main packages in an isolated environment:

```bash
python -m venv "$pr70_replay/venv"
"$pr70_replay/venv/bin/python" -m pip install torch==2.14.1+cpu torchvision==0.29.1+cpu --index-url https://download.pytorch.org/whl/cpu
"$pr70_replay/venv/bin/python" -m pip install ultralytics==8.4.182 numpy==2.4.6 Pillow==11.3.0 opencv-python==5.0.0.93
```

After hash-verifying and extracting the complete archives into `pr35` and `returned` beneath the replay directory:

```bash
python "$pr70_evidence/run_pr70_inference.py" \
  --root "$pr70_replay/source-c8d8f30" \
  --python "$pr70_replay/venv/bin/python" \
  --output "$pr70_replay/fresh-inference" \
  --pr35-manifest "$pr70_replay/pr35/candidate/checkpoint-manifest.json" \
  --reference-manifest "$pr70_replay/returned/reference/checkpoint-manifest.json" \
  --candidate-manifest "$pr70_replay/returned/candidate/checkpoint-manifest.json"
```

Then run `measure_pr70.py` using the new prediction paths and original extracted manifests. Fresh inference metadata/timings may differ by machine; do not expect prediction-file byte identity across environments. [inference-pip-freeze.txt](inference-pip-freeze.txt) records the actual full package inventory used here.

## Interpretation and limits

The useful result is a measured tradeoff: more 5-pixel pair successes on the two-corner subset, fewer matched partial fronts, and additional real background proposals. Preserve the existing model selection while the named annotation ownership/extent issues receive independent original-image review. Any later reference amendment should have a separate manifest and replay result; it should not replace this preregistered comparison.

This is one seed and one fixed recipe on a repeatedly used development split. The scenes have already informed annotation and model choices. Scores do not establish unseen-session reliability, the physical correctness of unknown corners, or deployment readiness. CPU inference compares the three models under one runtime; it is not an L4 or production latency benchmark. No thresholds, seeds, epochs or NMS settings were searched, and no model was promoted.
