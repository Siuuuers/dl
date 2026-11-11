# PR43 visible-extent result evidence

**Retain the existing PR35 checkpoint at proposal confidence 0.13.** This matched annotation experiment does not meet its promotion rule. No further training is needed to reproduce these numbers.

The primary reference is the frozen corrected set of **663 objects, 1,228 visible endpoints and 584 objects with both corners locatable**. All eligible objects remain in the corner denominators, including detection misses.

| Confidence | Model | Matches | Misses | Extras | Both corners within 5 px | Within 10 px |
|---|---|---:|---:|---:|---:|---:|
| **0.13 primary** | Old-reviewed / current PR35 | 643 | 20 | 85 | 529/584 | 577/584 |
| **0.13 primary** | Extended-reviewed | 642 | 21 | 87 | 526/584 | 578/584 |
| 0.15 diagnostic | Old-reviewed / current PR35 | 642 | 21 | 76 | 529/584 | 577/584 |
| 0.15 diagnostic | Extended-reviewed | 641 | 22 | 76 | 526/584 | 578/584 |

The candidate gains no matched reference object at either threshold. The old control and current PR35 have identical recorded loaded model-state tensors (879/879) and all 40 ordered image/threshold prediction arrays; fresh current predictions also reproduce the earlier hash-pinned 0.13/0.15 evidence. The extended checkpoint changes 705/879 recorded model-state tensors. The training provenance audit separately compares serialized tensor storage: 883/883 equal for old/current and 705/883 different for old/extended. Those denominators describe different representations.

Both training arms independently started from epoch173 and completed 30 epochs, 1,050 batches, 4,110 image presentations and **78 optimizer updates**, under the unchanged seed46 recipe. The accepted label difference is 11 box replacements and one missing-front addition across images 061, 079, 082 and 094; every prior point token and all eight inherited old-reviewed label files are preserved. Images 062 and 063 remain unchanged. The original **664-object in-training validation** and the corrected post-training reference are reported separately.

## Quick verification and numerical replay

Use Python 3.11 or newer. This command needs only the standard library and the extracted capsule: **no model weights, original PNGs, GPU, PyTorch, Ultralytics or network access**.

From the directory containing `evidence.zip` and its checksum:

```bash
sha256sum -c evidence.zip.sha256
python3 -m zipfile -e evidence.zip capsule
python3 -B capsule/tools/verify_evidence.py capsule --replay --output capsule-replay.json
```

Use a new extraction directory and a new output JSON name. The verifier rejects missing, changed, extra or symlinked capsule files before importing the included scorers. It checks the protocol, 157 original label files, both cumulative candidates, reference labels, run inventory and checkpoint/receipt metadata; then recomputes all **240 per-image scores, 240 quality records, 12 full aggregates and 72 subset aggregates**, corner groups and gained/lost identities. It also verifies the saved old/current and historical-current prediction equality.

This mode replays recorded predictions. It does not re-observe the original image pixels, rehash omitted model binaries or freshly derive their recorded state hashes. The original full driver and independent auditor are preserved unchanged for the stronger external-input verification below.

## Contents and immutable identities

| Path in the capsule | Purpose |
|---|---|
| `run-001/` | Every original run JSON/JSONL, including ordered predictions, both references, full score/quality outputs, identities, uncertainty notes and runtime settings |
| `run-001.stdout.log` | Original inference console output |
| `tools/evaluate_comparison.py` | Exact executed driver, SHA256 `01da5a0f067d3734cdaa990674d9092c768febc4a089dbb35f8ad3e19ef74b8f` |
| `tools/independent_replay_audit.py` | Unchanged independent numerical auditor |
| `tools/audit_returned_training.py` | Unchanged training/provenance audit, retaining its original execution paths |
| `verification/` | Original audit results and packager verification output |
| `training/` | Both returned arms and current PR35: archive indices, checkpoint manifests, receipts, staging inventories, training logs and CSVs; weights excluded |
| `source/` | Hash-bound scorers/helpers, policy, protocol, model card, all original labels, old/new candidate labels and corrected reference |
| `prior-current/` | Exact earlier current prediction member and its original archive inventory |
| `review/` | Bounded visual-review queue, decisions, source-pixel crops/figures and associated provenance |
| `summary.json` | Unmodified independently derived report summary; identical bytes are supplied beside the ZIP, with its generator at `tools/build_report_summary.py` |
| `external-inputs.json` | Source/result commits and omitted archive/model/image identities |
| `inventory.json` | SHA256 and byte size of every payload file; inventory itself is excluded to avoid self-reference |

The source is [`d43a58b9456a782ace14e20f0bf7ed2da3572ca1`](https://github.com/Siuuuers/dl/tree/d43a58b9456a782ace14e20f0bf7ed2da3572ca1); the final notebook is at [`dedee7042e541da7415276e0dbbe95e0f554fcb4`](https://github.com/Siuuuers/dl/tree/dedee7042e541da7415276e0dbbe95e0f554fcb4). The unchanged protocol SHA256 is `42aa275a5d2ed1ccabb786c07b537a8f0d515b27b7804fec2b71e942d6c9022b`.

| External artifact | Commit | SHA256 |
|---|---|---|
| PR43 returned ZIP, 39,776,639 bytes | `a964b9bc5437c47fe76a11cfd02e15c467e4da0d` | `e415c0ccbd07806851e9221ced6df6cd43d2ae54857ae7e3a6d7938ec56dcfca` |
| PR35 comparator ZIP, 39,776,456 bytes | `8e8bb99cabcb44ed0ec0acd14db911aa44013fd8` | `df53683ccf6f266a605b2844f42a98e808f75686cfab09f14d680c9d061e8a91` |
| Earlier current prediction evidence ZIP | `d43a58b9456a782ace14e20f0bf7ed2da3572ca1` | `baf48d2e52ef3571ef500de74c0f2a1b5a25e990ba290e4692f440d457d0bcb3` |

## Acquire external inputs for full replay or fresh inference

These optional steps preserve the heavyweight inputs outside the capsule. Set the first two paths to your extracted capsule and a **new** working directory. Git and Git LFS are required. The standard credential-free remote is shown; use your normal authenticated Git configuration if needed.

```bash
CAPSULE_DIR="$(pwd)/capsule"
WORK_DIR="$(pwd)/pr43-reproduction"
mkdir "$WORK_DIR"
git clone --filter=blob:none --no-checkout https://github.com/Siuuuers/dl.git "$WORK_DIR/source"
git -C "$WORK_DIR/source" fetch origin d43a58b9456a782ace14e20f0bf7ed2da3572ca1 dedee7042e541da7415276e0dbbe95e0f554fcb4 a964b9bc5437c47fe76a11cfd02e15c467e4da0d 8e8bb99cabcb44ed0ec0acd14db911aa44013fd8
GIT_LFS_SKIP_SMUDGE=1 git -C "$WORK_DIR/source" checkout --detach d43a58b9456a782ace14e20f0bf7ed2da3572ca1
git -C "$WORK_DIR/source" lfs install --local
git -C "$WORK_DIR/source" lfs pull --include="data/images/val/*.png" --exclude=""
git -C "$WORK_DIR/source" show a964b9bc5437c47fe76a11cfd02e15c467e4da0d:results/gpu/baomai-visible-extent-a100-001/baomai-visible-extent-seed46-results.zip > "$WORK_DIR/baomai-visible-extent-seed46-results.zip"
git -C "$WORK_DIR/source" show a964b9bc5437c47fe76a11cfd02e15c467e4da0d:results/gpu/baomai-visible-extent-a100-001/baomai-visible-extent-seed46-results.zip.sha256 > "$WORK_DIR/baomai-visible-extent-seed46-results.zip.sha256"
git -C "$WORK_DIR/source" show 8e8bb99cabcb44ed0ec0acd14db911aa44013fd8:results/gpu/baomai-reviewed-annotations-seed46-l4-001/baomai-reviewed-annotations-seed46-results.zip > "$WORK_DIR/current-seed46-results.zip"
git -C "$WORK_DIR/source" show d43a58b9456a782ace14e20f0bf7ed2da3572ca1:results/verification/baomai-proposal-filter-20261111/confidence-evidence.zip > "$WORK_DIR/prior-current-evidence.zip"
```

Hash-check all three archives before extracting the two model bundles:

```bash
export CAPSULE_DIR WORK_DIR
python3 - <<'PY'
import hashlib, json, os
from pathlib import Path
work = Path(os.environ['WORK_DIR'])
specs = json.loads((Path(os.environ['CAPSULE_DIR'])/'external-inputs.json').read_text())['archives']
names = {'returned':'baomai-visible-extent-seed46-results.zip',
         'current':'current-seed46-results.zip', 'prior_current_predictions':'prior-current-evidence.zip'}
for role, name in names.items():
    path, expected = work/name, specs[role]
    with path.open('rb') as stream:
        assert hashlib.file_digest(stream, 'sha256').hexdigest() == expected['sha256'], name
    assert 'size_bytes' not in expected or path.stat().st_size == expected['size_bytes'], name
print('All three external archive identities verified.')
PY
python3 -m zipfile -e "$WORK_DIR/baomai-visible-extent-seed46-results.zip" "$WORK_DIR/returned"
python3 -m zipfile -e "$WORK_DIR/current-seed46-results.zip" "$WORK_DIR/current"
git -C "$WORK_DIR/source" status --porcelain
```

The source status should be empty. Do not place audit outputs inside this checkout.

### Full driver replay: no model loading or inference

This stronger replay hashes the three external final checkpoints and 20 original validation PNGs. It uses the original labels, manifests and exact driver bytes. It needs no epoch173 initialization bytes, training PNGs, PyTorch or Ultralytics.

```bash
python3 -B "$CAPSULE_DIR/tools/evaluate_comparison.py" \
  --root "$WORK_DIR/source" --returned-root "$WORK_DIR/returned" \
  --current-manifest "$WORK_DIR/current/candidate/checkpoint-manifest.json" \
  --replay-directory "$CAPSULE_DIR/run-001" --output "$WORK_DIR/full-replay-001"
python3 -B "$CAPSULE_DIR/tools/independent_replay_audit.py" \
  --root "$WORK_DIR/source" --run "$CAPSULE_DIR/run-001" \
  --returned-root "$WORK_DIR/returned" \
  --current-manifest "$WORK_DIR/current/candidate/checkpoint-manifest.json" \
  --driver "$CAPSULE_DIR/tools/evaluate_comparison.py" \
  --prior-current-zip "$WORK_DIR/prior-current-evidence.zip" \
  --output "$WORK_DIR/independent-replay.json"
```

Compare all six full-replay `scoring-*.json` SHA256s with the corresponding descriptors in `run-001/inventory.json`. New execution timestamps and path strings need not match the historical run. The fixed scorers and scores must match.

### Optional fresh CPU inference

The recorded environment is Python 3.12.14 on Linux x86_64, Torch 2.10.0+cpu, torchvision 0.25.0+cpu, Ultralytics 8.4.182, NumPy 2.2.6, Pillow 11.3.0 and OpenCV 4.11.0.86. The included 40-package lock matches the used environment. Use the PyTorch CPU wheel index for the CPU-specific builds before installing the unchanged lock:

```bash
python3.12 -m venv "$WORK_DIR/inference-env"
"$WORK_DIR/inference-env/bin/python" -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.10.0+cpu torchvision==0.25.0+cpu
"$WORK_DIR/inference-env/bin/python" -m pip install -r "$CAPSULE_DIR/tools/requirements-inference.lock.txt"
"$WORK_DIR/inference-env/bin/python" -B "$CAPSULE_DIR/tools/evaluate_comparison.py" \
  --root "$WORK_DIR/source" --returned-root "$WORK_DIR/returned" \
  --current-manifest "$WORK_DIR/current/candidate/checkpoint-manifest.json" \
  --output "$WORK_DIR/fresh-inference-001"
```

The driver fixes confidence 0.13/0.15, image size 1280, NMS IoU 0.70, matching IoU 0.50 and endpoint confidence 0.50. It records 120 image/threshold/model predictions plus three excluded warmups, blocks network access during evaluation and uses a bounded runtime. A matching recorded environment is needed for a strict prediction-equivalence comparison; another platform is a new measurement, not an assumed identical result.

### Full returned-training verification

The strict packager checks the returned pair without loading models or training. Run this against the original archive, outside the source and bundle directories:

```bash
python3 -B "$WORK_DIR/source/scripts/package_visible_extent_results.py" \
  --root "$WORK_DIR/source" --old-reviewed "$WORK_DIR/returned/old_reviewed" \
  --extended-reviewed "$WORK_DIR/returned/extended_reviewed" \
  --output "$WORK_DIR/baomai-visible-extent-seed46-results.zip" --verify-existing \
  > "$WORK_DIR/package-verification.stdout" 2> "$WORK_DIR/package-verification.stderr"
```

The included historical training-audit script additionally verifies all 157 image bytes, original epoch173 bytes, source Git identity and serialized checkpoint storage comparisons. Its recorded absolute paths describe the original audit workspace; use the portable strict packager command above for returned-bundle verification. The weights/PNG-independent capsule replay separately rechecks the small annotation/protocol inputs.

## Repacking and limits

To rebuild a ZIP from a verified extraction, without the external inputs:

```bash
python3 -B capsule/tools/build_evidence.py --repack capsule --destination repacked
```

The builder fixes member order, timestamps and permissions, rejects existing outputs and checks complete ZIP byte readback and CRCs. Repacking is byte-identical in the audited Python/zlib environment; payload hashes remain the portable integrity contract across compressor versions.

The archive preserves evidence, not an independent observation of remote execution. Drive completion markers and a restoration transcript were not in PR43. The pinned notebook supports verified completed-arm recovery without retraining, and the returned receipts are consistent with completed matched runs. A reused, correlated 20-image development set, one seed and retained reference uncertainty do not establish unseen-scene reliability. Reference-unlocatable endpoint output is an availability disagreement; a hidden-coordinate error is not measured.
