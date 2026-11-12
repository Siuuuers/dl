# PR47 consistency comparison: evidence and reproduction

**Decision: retain the current PR35 checkpoint and the explicit 0.13 reviewed-proposal command.** The returned L4 training pair is valid. The fourteen-box candidate does not establish an additional visible-front recovery and does not preserve all prior 5 px pair successes. This completes the fixed comparison; no further GPU run is requested.

Read the [complete report and original-image comparisons](https://github.com/Siuuuers/dl/blob/codex/consistency-results-20261112/docs/consistency-results-2026-11-12.md). The working branch is `codex/consistency-results-20261112`. [PR47](https://github.com/Siuuuers/dl/pull/47) contains returned results, rather than the complete working workflow.

## Result at the declared primary settings

Image size 1280, box confidence 0.13, ordinary NMS IoU 0.70, matching IoU 0.50, endpoint cutoff 0.50. The primary reference has 663 objects and 584 objects with both physical lower-front corners locatable. The only diagnostic box confidence is 0.15. Both references remain fixed and are reported separately.

| Primary measure | Current PR35 | Old-reviewed L4 control | Fourteen-box L4 candidate |
| --- | ---: | ---: | ---: |
| Matches / 663 | 643 | 641 | 642 |
| Misses | 20 | 22 | 21 |
| Scored extras | 85 | 88 | 88 |
| Both corners within 5 px / 584 | 529 | 527 | 530 |
| Both corners within 10 px / 584 | 577 | 578 | 578 |
| Miss-free nonempty images / 17 | 3 | 3 | 3 |
| Predictions on three empty images | 0 | 0 | 0 |

The control-to-candidate match gain, 056:14, comes from an existing proposal crossing reference IoU 0.50 after a maximum 0.267456 px edge shift. PR35 already matches that front. The PR35-relative lost match, 090:43, also retains a near-identical proposal. Corner success identities have both gains and losses. The 10 px gain versus PR35 is already present in the L4 control and reflects assignment among existing alternatives in 007:15.

All 14 unique candidate-only proposals across 23 comparison/threshold contexts were reviewed on original pixels: nine are alternatives around existing fronts, four cover background or rack structure, and one remains unresolved. None establishes a new physical-front recovery. These association counts are review aids, not physical object counts.

## 1. Verify and replay the included evidence

Run from this evidence directory in a checkout containing `evidence.zip` and its adjacent checksum file. Use Python 3.11 or newer. The following verifies every included payload and independently recomputes the saved-prediction results. It needs no model binaries, full original PNGs, PyTorch or Ultralytics.

```bash
sha256sum -c evidence.zip.sha256
PR47_WORK="$(mktemp -d)"
python -m zipfile -e evidence.zip "$PR47_WORK/capsule"
python -B "$PR47_WORK/capsule/tools/verify_evidence.py" \
  "$PR47_WORK/capsule" \
  --replay \
  --output "$PR47_WORK/capsule-replay.json"
```

Expected result: `status=verified`, with all 12 aggregate cases, 240 image cases, 7,962 object assignment/endpoint states and 12 gained/lost-identity comparison cases agreeing with the recorded scorer to absolute tolerance `1e-9`. This independently implemented arithmetic imports no repository scoring or model helper. The verifier also validates included original/candidate/reference labels and the completed-training metadata chain. Its output explicitly records `model_weights_reverified=false`, `original_pngs_reverified=false`, `training_executed=false` and `inference_executed=false`.

The replay writes its final record outside the capsule. Its detailed intermediate independent results use a temporary directory and are discarded; the compact cross-check remains in the final record. To retain those detailed results, run the two included tools explicitly:

```bash
python -B "$PR47_WORK/capsule/tools/independent_metrics.py" \
  --root "$PR47_WORK/capsule/source" \
  --predictions "$PR47_WORK/capsule/run-001" \
  --output "$PR47_WORK/independent-retained.json"
python -B "$PR47_WORK/capsule/tools/compare_official.py" \
  --root "$PR47_WORK/capsule/source" \
  --predictions "$PR47_WORK/capsule/run-001" \
  --independent "$PR47_WORK/independent-retained.json" \
  --output "$PR47_WORK/cross-check-retained.json"
```

Always choose new output paths. Do not add replay output or Python bytecode inside the inventoried capsule.

## 2. Acquire the immutable external inputs for stronger verification

This step is optional for the included numerical replay. It is required to rehash the actual final weights and original validation images, or to run fresh inference. The capsule deliberately excludes final weights, full original PNGs and the original epoch-173 binary.

| External input | Immutable commit | Repository path | SHA256 | Bytes |
| --- | --- | --- | --- | ---: |
| Returned L4 pair | `24e43a407da01b1e26834b6dee60a51e1ec54577` | `results/gpu/baomai-consistency-l4-001/baomai-consistency-seed46-results.zip` | `bdae4de69b3560d4fc8821dcd6f364c77eef2ed4c624c3720eae696eaa5ee5cb` | 39,776,445 |
| Current PR35 bundle | `8e8bb99cabcb44ed0ec0acd14db911aa44013fd8` | `results/gpu/baomai-reviewed-annotations-seed46-l4-001/baomai-reviewed-annotations-seed46-results.zip` | `df53683ccf6f266a605b2844f42a98e808f75686cfab09f14d680c9d061e8a91` | 39,776,456 |
| Source and validation-image identity | `14561b7f01cc89c4472e11ffb37b3943ea3e34f9` | `data/manifest.json` | `72ae8365d5abf0b45b75d400a2e6d22119c0e31b54ee33035feb8edc1f967926` | See manifest |

Use the same `PR47_WORK` created above. Git and Git LFS must be installed. This checks out the exact training/evaluation implementation and hydrates only the 20 original validation images. Do not run this from an existing worktree containing local changes.

```bash
GIT_LFS_SKIP_SMUDGE=1 git clone --no-checkout \
  https://github.com/Siuuuers/dl.git "$PR47_WORK/source"
git -C "$PR47_WORK/source" fetch origin \
  14561b7f01cc89c4472e11ffb37b3943ea3e34f9 \
  24e43a407da01b1e26834b6dee60a51e1ec54577 \
  8e8bb99cabcb44ed0ec0acd14db911aa44013fd8
GIT_LFS_SKIP_SMUDGE=1 git -C "$PR47_WORK/source" checkout --detach \
  14561b7f01cc89c4472e11ffb37b3943ea3e34f9
git -C "$PR47_WORK/source" lfs install --local
git -C "$PR47_WORK/source" lfs pull \
  --include='data/images/val/*.png' --exclude=''

git -C "$PR47_WORK/source" show \
  '24e43a407da01b1e26834b6dee60a51e1ec54577:results/gpu/baomai-consistency-l4-001/baomai-consistency-seed46-results.zip' \
  > "$PR47_WORK/baomai-consistency-seed46-results.zip"
git -C "$PR47_WORK/source" show \
  '8e8bb99cabcb44ed0ec0acd14db911aa44013fd8:results/gpu/baomai-reviewed-annotations-seed46-l4-001/baomai-reviewed-annotations-seed46-results.zip' \
  > "$PR47_WORK/pr35-results.zip"

cd "$PR47_WORK"
sha256sum -c <<'SHA256'
bdae4de69b3560d4fc8821dcd6f364c77eef2ed4c624c3720eae696eaa5ee5cb  baomai-consistency-seed46-results.zip
df53683ccf6f266a605b2844f42a98e808f75686cfab09f14d680c9d061e8a91  pr35-results.zip
SHA256
python -m zipfile -e baomai-consistency-seed46-results.zip "$PR47_WORK/returned"
python -m zipfile -e pr35-results.zip "$PR47_WORK/current"

python -B "$PR47_WORK/source/scripts/package_consistency_results.py" \
  --root "$PR47_WORK/source" \
  --old-reviewed "$PR47_WORK/returned/old_reviewed" \
  --consistency-reviewed "$PR47_WORK/returned/consistency_reviewed" \
  --output "$PR47_WORK/baomai-consistency-seed46-results.zip" \
  --verify-existing
```

The strict packager verifies the existing returned ZIP without rewriting it. The fixed protocol hash is `da692f2261fae441dfa7d68b3f86ed850acbb861c75e86e8c54f40b7184e540b`; the corrected reference hash is `2e60b3e81195448cc565a1000b4305e3215baba2bf0ee0c976e6551c2e64937c`. The external driver binds each final checkpoint:

| Role | Final checkpoint SHA256 |
| --- | --- |
| `current` / PR35 | `420c00f73eeba24e587342400878b7d96371d3e745772113809aa705ec8163b1` |
| `old_reviewed` / L4 control | `11978a3c843ff8415aee416387d2af5acf4d913db774a26e086e88da12710780` |
| `consistency_reviewed` / L4 candidate | `61d0beea750bcbbc4bb02e17b11dec387fa288382ef2a1b5423e486a401b1cdb` |

The completed full runs each record 30 epochs, 1,050 batches, 4,110 image presentations and 78 actual optimizer updates. Their receipt/runtime/source/settings checks agree. The ZIP contains the full-run artifacts, not the separate resource-trial bundle, so it does not independently prove trial execution. Validation PNG bytes are freshly checked in this reproduction path; training-image identities remain checks against recorded staging inventories unless those originals are separately hydrated and rehashed.

## 3. Replay with external-input verification, without inference

After steps 1 and 2, this uses the exact executed driver and unchanged repository scorers. It verifies actual returned/current checkpoint bytes and the 20 original validation PNGs before replaying saved predictions. It does not load a model or train one.

```bash
python -B "$PR47_WORK/capsule/tools/evaluate_comparison.py" \
  --root "$PR47_WORK/source" \
  --returned-root "$PR47_WORK/returned" \
  --current-manifest "$PR47_WORK/current/candidate/checkpoint-manifest.json" \
  --replay-directory "$PR47_WORK/capsule/run-001" \
  --output "$PR47_WORK/full-replay-001"
```

The six `scoring-*.json` files should be byte-identical to their counterparts in `capsule/run-001`. Comparison metrics and identities agree, while command paths, creation times, mode and timing differ. Keep the source checkout clean and outputs outside it.

Executed driver SHA256: `aee77223ed90b36b4700dcbde1c88ade59a3a262a078c6ea6df3b80316b7177c`. The adjacent plaintext `evaluate_comparison.py` and the capsule copy are identical.

## 4. Optional fresh CPU inference

This is a reproduction option, not a request for another experiment. The completed evaluation used **Python 3.12.14, PyTorch 2.10.0+cpu, torchvision 0.25.0+cpu, Ultralytics 8.4.182, NumPy 2.2.6, Pillow 11.3.0 and OpenCV 4.11.0.86**, with the complete 40-package lock in `runtime/requirements-inference.lock.txt`. The lock matches the archived PR43 inference environment exactly.

With a Python 3.12.14 interpreter available as `python3.12`, use a fresh virtual environment:

```bash
python3.12 -m venv "$PR47_WORK/inference-venv"
"$PR47_WORK/inference-venv/bin/python" -m pip install \
  --index-url https://download.pytorch.org/whl/cpu \
  'torch==2.10.0+cpu' 'torchvision==0.25.0+cpu'
"$PR47_WORK/inference-venv/bin/python" -m pip install \
  -r "$PR47_WORK/capsule/runtime/requirements-inference.lock.txt"
"$PR47_WORK/inference-venv/bin/python" -B \
  "$PR47_WORK/capsule/tools/evaluate_comparison.py" \
  --root "$PR47_WORK/source" \
  --returned-root "$PR47_WORK/returned" \
  --current-manifest "$PR47_WORK/current/candidate/checkpoint-manifest.json" \
  --output "$PR47_WORK/fresh-inference-001"
```

The driver makes 120 image predictions and three excluded blank warmups, then scores 240 model/image/reference/threshold combinations. It fixes FP32 CPU inference, four PyTorch threads, one inter-op thread, rectangular preprocessing and ordinary NMS, and blocks network access during evaluation. These timings are descriptive offline observations, not a live-camera benchmark. Exact floating-point array equality across CPU executions is not guaranteed: fresh PR35 kept all prior score/identity outcomes but differed from its archived PR43 arrays by at most 0.000244141 px in box coordinates and 0.000122070 px in endpoint coordinates.

## Capsule contents and limits

| Path inside capsule | Purpose |
| --- | --- |
| `inventory.json` | Exact payload hashes/sizes and pinned external-input identities; excludes itself |
| `summary.json` | Compact derived metrics, decision and proposal-review tally |
| `run-001/` | Exact fresh predictions, both-reference scores, comparison identities, protocol and run status |
| `training/` | All three selected checkpoint manifests, receipts, staging manifests, logs and training CSVs; no weight binaries |
| `source/` | Unchanged helpers, protocol, all 157 original label files, selected candidates, corrected reference and required metadata |
| `audit/provenance/` | Independent returned-archive and training-chain checks |
| `audit/metrics/` | Independent arithmetic, control-state comparison and historical numerical-stability checks |
| `audit/visual-independent/` | 056:14, 090:43 and 007:15 original-pixel decisions, precise numerics and report figures |
| `audit/proposal-review/` | All 14 unique changed proposals, raw crops, overlays and exact comparison contexts |
| `tools/` | Exact executed driver, portable verifier, independent replay tools and deterministic capsule builder |
| `runtime/` | Complete inference lock, runtime verification and lock provenance |

The cropped review figures use actual original-image pixels. Their source hashes, crop coordinates, drawing code and review records are preserved. They are not new observations or synthetic imagery. One-based reference row identifiers are distinct from zero-based prediction indices. Prediction indices are local to a model/image/threshold and must not be treated as stable physical identities.

The PR35 model was trained on an A100; the returned matched pair was trained on an L4. Their loaded states differ. Within-L4 differences address the fixed annotation intervention, while PR35 comparison addresses the operational replacement decision. Hardware alone is not established as the cause of control differences.

The 20 reused development views remain correlated and are not an independent generalization test. Nineteen unresolved reference dispositions and the separate 011:61 caveat stay recorded. The working target remains the observed front-face box and two actual lower-front corners; hidden corners have no certified coordinate target, and this study does not establish full hidden extent or physical dimensions.

The evidence preserves the accepted fourteen-box candidate without adopting it as the operational model. Original data, references, protocol, training recipe, architecture and current prediction defaults remain unchanged.
