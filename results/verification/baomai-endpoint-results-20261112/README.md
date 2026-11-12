# PR52 endpoint comparison evidence

Date: 2026-11-12, Asia/Hong_Kong

**Decision: retain PR35.** Read the [complete result](../../../docs/endpoint-results-2026-11-12.md). This directory records one completed fixed comparison, not a new training request.

## Contents

- `evaluate_comparison.py` is the exact driver executed for the fresh three-model evaluation. SHA256: `54a66f22775fc8dbc1469a651991977d1f145efee99427bca6835e693abde64c`.
- `summary.json` contains compact results for both frozen references, model identities, the common-slot comparisons and the decision.
- `evidence.zip` contains the saved predictions, complete score and comparison records, frozen scoring-label bytes and source metadata, training provenance, independent arithmetic audit, runtime lock, and original-image review artifacts.
- `evidence.zip.sha256` identifies the capsule bytes. `inventory.json` indexes every capsule member and records the completed readback and numerical-replay checks.

The capsule excludes model weights and complete original PNGs. Its original-pixel review crops are derived from the separately verified originals; they are not replacements for the complete images.

## Verify and repeat the saved-prediction arithmetic

From this directory, first verify the archive, then extract it into a new directory:

```bash
sha256sum -c evidence.zip.sha256
unzip evidence.zip -d /path/to/new-evidence-directory
```

The following independent audit requires only the Python standard library. It parses the included label bytes, repeats matching and corner calculations directly from saved predictions, and compares the resulting aggregates, object states and pairwise changes with the recorded results:

```bash
python3 -B /path/to/new-evidence-directory/audit/independent-audit.py \
  --root /path/to/new-evidence-directory/source \
  --evaluation /path/to/new-evidence-directory/evaluation \
  --output /path/to/new-audit-result.json \
  --skip-image-bytes
```

This numerical replay explicitly records `original_image_bytes_verified: false` and count `0`. It checks the saved image identities against the frozen manifest but does not re-observe the excluded image bytes. It also records that the optional historical replay comparison was skipped.

The archived full audit in `audit/independent-audit.json` did verify all twenty actual original PNGs. To repeat that stronger mode, point `--root` to a checkout with those original PNGs restored, and omit `--skip-image-bytes`. The audit also checks the two scoring-label sets and their manifest hashes.

To repeat the historical control/PR35 equality check, add `--prior-archive /path/to/prior-consistency-evidence.zip`. The script requires prior-archive SHA256 `25173e53d2c011c8039b0b1fc14804549708e10be2291fe8b032a1075fcdf953` before reading it. Its immutable source is [the PR48 consistency capsule](https://github.com/Siuuuers/dl/blob/c9f510a881c212d0aee936f85626af69ef5dbe3b/results/verification/baomai-consistency-results-20261112/evidence.zip).

## Repeat fresh inference

Fresh inference uses external inputs, rather than synthesizing weights or images from the capsule:

1. Use a clean checkout of `Siuuuers/dl` at `c9f510a881c212d0aee936f85626af69ef5dbe3b`, and restore its twenty original `data/images/val/*.png` files through Git LFS. Keep the original labels, candidate overlays, reference and source files unchanged.
2. Download and verify the [PR52 archive](https://github.com/Siuuuers/dl/blob/8714dbb254895e3aafae92bc43d6fbad4622ddc4/results/gpu/baomai-endpoint-visibility-l4-001/baomai-endpoint-visibility-seed46-results.zip), SHA256 `e3dc8755728733ad9527ef48d0d45682d80184b57871dbad515483d343c8f3d6`. Extract its `old_reviewed` and `endpoint_reviewed` bundles.
3. Download and verify the [PR35 archive](https://github.com/Siuuuers/dl/blob/8e8bb99cabcb44ed0ec0acd14db911aa44013fd8/results/gpu/baomai-reviewed-annotations-seed46-l4-001/baomai-reviewed-annotations-seed46-results.zip), SHA256 `df53683ccf6f266a605b2844f42a98e808f75686cfab09f14d680c9d061e8a91`. Select its `candidate/checkpoint-manifest.json`.
4. In a separate Python 3.12 environment, install the capsule's `runtime/requirements-inference.lock.txt`, using `--extra-index-url https://download.pytorch.org/whl/cpu` for the pinned CPU PyTorch wheels. The recorded run used Python 3.12.14 and verified all forty package versions.
5. Run the exact driver below, writing to a new directory outside the source checkout.

```bash
python -B evaluate_comparison.py \
  --root /path/to/clean-c9f510-checkout \
  --returned-root /path/to/extracted-PR52 \
  --current-manifest /path/to/extracted-PR35/candidate/checkpoint-manifest.json \
  --output /path/to/new-fresh-evaluation
```

The driver verifies the protocol, source helpers, labels and checkpoints, fixes confidence at 0.13, disables network activity during evaluation, and rejects incomplete or incompatible inputs. It makes sixty real-image predictions and three excluded blank warmups; no training runs. GPU training and CPU inference runtimes are recorded separately.

## Original-image review

`review/changed-cases/` preserves all fourteen changed reference cases, including raw-first notes, crop coordinates, image/reference/prediction hashes, matched and overlapping alternative predictions, continuous errors, adjudications, and reproducible figures. `review/proposals/` preserves all eight distinct endpoint-only proposals from the two review associations, including their raw observations and informed decisions.

The proposal association is a review aid, separate from the official confidence-ordered reference matcher. Four of the eight proposals are alternative extents of represented fronts, three cover foreground/background structures, and one remains ambiguous. No additional physical cassette-front recovery was established by this bounded review.

The renderer scripts accept explicit source/evaluation/output paths. They need the corresponding complete original images to reproduce crops. Numerical replay does not certify visual judgments, hidden-corner coordinates, independent capture-session performance, or model superiority.
