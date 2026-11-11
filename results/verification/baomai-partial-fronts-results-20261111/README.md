# Completed partial-front comparison

The [result report](../../../docs/partial-front-results-2026-11-11.md) retains the current PR35 seed46 checkpoint. The new labels recover no additional object at any of the three declared confidence thresholds. The [compact summary](summary.json) separates the corrected primary reference, diagnostics and original-reference results.

## What is preserved

- `evaluate_comparison.py`: the exact executed driver, which composes the unchanged repository scorers and preserves distinct model identities.
- `evidence.zip` and `evidence.zip.sha256`: real predictions, scores, source/runtime identities, completed-training metadata, independent audit and original-pixel case review. Weights and full original images remain in their existing sources.
- `evidence-inventory.json`: capsule identity and every retained payload hash/size.
- `summary.json`: concise verified results, both references, fixed thresholds, object/corner changes, control equivalence and inherited uncertainty.

The live evaluation ran from clean published source **`19459913c8be2f08978d8eee93387a6087407444`**, with driver SHA-256 **`8fdbd6cd37ee5ef40f6e5a6e8d844c37c1bc5e5be1e8895c39cbe0b2a9667814`**. It performed 180 original-image predictions and three excluded warmups. The subsequent driver replay performed zero model loads and reproduced all six scoring files byte for byte. A separate audit independently recomputed all 360 per-image scores and 360 quality records, aggregates and object/corner transitions.

## External inputs

| Input | Pinned source | SHA-256 |
| --- | --- | --- |
| Returned old/extended comparison | [PR40 archive](https://github.com/Siuuuers/dl/blob/fd1cb723a0be8859a50382b8903f0b2556103df3/results/gpu/baomai-partial-fronts-a100-001/baomai-partial-fronts-seed46-results.zip) | `717781a642c1ac5597eb76d17e5b9c07e9306f64baed6ba82759007ebcd8379c` |
| Currently used candidate | [PR35 archive](https://github.com/Siuuuers/dl/blob/8e8bb99cabcb44ed0ec0acd14db911aa44013fd8/results/gpu/baomai-reviewed-annotations-seed46-l4-001/baomai-reviewed-annotations-seed46-results.zip) | `df53683ccf6f266a605b2844f42a98e808f75686cfab09f14d680c9d061e8a91` |

Use the pinned evaluation source and its unchanged metadata, original labels, corrected reference and original validation PNGs. Git LFS must hydrate `data/images/val/*.png`. Verify both archive hashes before extraction into distinct new directories. Keep the complete `old_reviewed`, `extended_reviewed` and historical `candidate` bundles; checkpoint eligibility still verifies their external weight bytes during replay, without deserializing them.

## Replay without inference

After verifying and extracting the evidence capsule, run from the pinned repository:

```bash
python3 -B results/verification/baomai-partial-fronts-results-20261111/evaluate_comparison.py \
  --root /path/to/pinned-dl \
  --returned-root /path/to/pr40-extracted \
  --current-manifest /path/to/pr35-extracted/candidate/checkpoint-manifest.json \
  --replay-directory /path/to/evidence-extracted/run \
  --output /path/to/new-replay
```

Compare the six `scoring-*.json` hashes with `verification/replay-verification.json` inside the capsule. The comparison is identical after omitting only `mode`, `fresh_image_prediction_calls`, `warmup_calls_excluded` and `total_wall_seconds`; command, timestamp and source identity are separately recorded for each invocation.

For the independent audit, use the capsule's `verification/independent_replay_audit.py` with the same `--root`, `--returned-root` and `--current-manifest`, plus `--run /path/to/evidence-extracted/run` and `--output /path/to/new-audit.json`. The audit imports the original scorers directly and never imports the new evaluation driver.

The capsule's `verification.json` records the external URLs, exact source and verification sequence. `review/inference-requirements.txt` records the isolated environment used for fresh CPU inference; replay does not invoke the inference path. Original and corrected-reference scores remain separate. No extra training, threshold search, annotation amendment or automatic model promotion is part of replay.
