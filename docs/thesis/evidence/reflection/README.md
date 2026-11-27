# Reflection experiment evidence

28 November 2026, Asia/Hong_Kong.

The [results note](../../reflection-results.md) explains the experiment, its numerical tradeoffs, and the box-matching example. This folder preserves a compact, independently replayed evidence record. It does not contain a new model run.

## Files

| File | Purpose |
| --- | --- |
| `audit.json` | Verified counts, paired changes, numerical stability, runtime, and the boundary example |
| `case-changes.json.gz` | Complete per-target results for both raw and boundary-suppressed stages under all four references |
| `input-provenance.json` | Immutable source, model, archive, and replay-script identities |

The returned archive is [the immutable PR152 result](https://raw.githubusercontent.com/Siuuuers/dl/dfe5392109433fcee51f41716710001bb123a3d5/results/a100/baomai-flip-inference-r56-a100-001/baomai-flip-inference-r56-results.zip). Its SHA-256 is `88689ce9f3d03178683ea98ce98d50f5ddc4e07ba1fb24262e7820e9488bd114` and its size is 2,931,102 bytes. It contains 16 ordinary ZIP members with a complete inventory.

## Reproduce the compact audit

From the repository root, provide the downloaded result archive and a new output directory:

```bash
python -B scripts/audit_reflection_return.py \
  --archive /path/to/baomai-flip-inference-r56-results.zip \
  --output /path/to/reflection-audit
```

This uses Python's standard library. The script verifies the exact returned archive and each member's size and hash. It loads the unchanged annotation bytes and scorer from the repository, checking their recorded hashes before use. It independently applies the continuous inverse reflection and corner-slot swap, repeats the existing boundary suppression, replays the original scorer, and computes complete-pair success directly. The full native and repeated arrays are verified, including all three repeats per arm. Images, weights, a GPU, and model packages are unnecessary for this compact replay.

The original pinned driver was also run in its `verify` mode using the verified upstream input archive, model bytes, and twenty original PNGs. Its complete source/data preflight, baseline-equivalence calculation, cases, complementarity, timing, and status checks passed. That broader check uses source commit `dc7eec630bb0b613fc1fe784e4d9818a0893658e`; the reproduction protocol is in [the original reflection experiment guide](../../flip-experiment-r56.md).

## Interpret counts correctly

“Raw” means the model's returned predictions after native processing and inverse reflection, before the additional boundary-suppression stage. “Postclip” means after that existing additional suppression. Neither contains pre-native-NMS proposals.

The audit contains 16 aggregate result tables: two arms, four annotation references, and two stages. The same combinations yield 320 per-image tables. Complete-pair success counts use all reference fronts with two known corners. The separate “all” complementarity group in the returned archive includes threshold changes on one-corner targets, so use its two-corner group when checking both-corner claims.

All four reference versions describe the same development photographs. Source hashes establish which data and calculations were used; they do not establish independent-scene accuracy.
