# PR58 composed-label evaluation evidence

Date: 2026-11-13 UTC. **Decision: retain PR35.** Read the [complete report](../../../docs/composed-results-2026-11-13.md). This records a completed comparison; it does not request another training run.

## Start here

- `summary.json`: compact results for all three frozen references, changed identities, selected settings and decision.
- `recovery-verification.json`: independent continuation checks of the returned PR58 archive, recovered original records and this exact public derivative. The frozen capsule and historical audit files are unchanged.
- `evaluation.zip`: a minimized public derivative containing saved numerical predictions, frozen labels/manifests, the independent auditor, exact executed driver, essential provenance and original-image review records.
- `evaluation.zip.sha256`: the archive checksum, with the exact basename `evaluation.zip`.
- `inventory.json`: every archive member's size and SHA256, plus final extraction/replay verification.
- `figures/`: two original-image comparison figures used in the report.

The archive is **2,121,577 bytes**, SHA256 `1d87aa19fdb62ec87420b07bec44919a04bfe95d713838c596007bcc062baf6a`, with **162 members including 161 inventoried payloads**. It contains no images or model weights. All 80 PNGs from the original capsule, full host-environment inventories/configuration and unrelated package snapshots are omitted. Private workspace prefixes in six generated metadata files are replaced by a documented symbolic path.

All 60 per-image prediction-record lines, 60 frozen reference label files, score files, protocol, auditor and driver retain their original bytes. Prediction run headers and whole-file hashes change where paths are normalized. Inside the archive, `public-derivation.md` and `public-derivation.json` give exact exclusions, original/public hash mappings and verification scope. The unchanged original capsule is identified by SHA256 `ceaed6f73e7aea16746e605ce920eafa58950596d7ebd765653163a703ac0c37` and size 21,215,096 bytes; it is not included in this publication or its branch history.

The source images already belong to the public dataset. The six full training originals used by the historical pilot and the 20 recorded validation-image identities match the [immutable upstream manifest](https://github.com/an94-doll/embedding_vision/blob/351eb3cf960ccc34829e2bc72e1fa4645acc6295/data/manifest.json). Review records preserve their image hashes and crop geometry. This public archive can replay the saved arithmetic; visual review additionally requires obtaining the source images. The two selected comparison figures are published separately under `figures/`, and the two new annotation figures are under [border-front review](../../review/border-fronts-20261113/README.md).

## Verify the archive and replay the arithmetic offline

From this directory:

```bash
sha256sum -c evaluation.zip.sha256
unzip evaluation.zip -d /path/to/new-pr58-replay
```

From the newly extracted directory:

```bash
python3 -S -B verify_payload.py
python3 -S -B audit/independent-audit.py \
  --root source \
  --evaluation evaluation \
  --driver driver/evaluate_comparison.py \
  --skip-image-bytes \
  --output /path/to/new-replay-audit.json
```

Only the Python standard library is needed. No GPU, Torch, model loading or network access is involved. `verify_payload.py` checks every file listed in `payload-inventory.json`; that inventory excludes itself to avoid a recursive hash. The external ZIP checksum covers the complete archive, including the inventory.

The numerical auditor independently parses the included labels and recomputes confidence-ordered matching, duplicate/other-extra classifications, returned endpoints, Euclidean corner errors, quantiles, object groups, image counts and pairwise identity changes. It does not import the project's scorer or inference driver.

The expected result is `status: verified`, with 180 image/model/reference cases, 5,970 reference-object states, nine pairwise comparisons and 221,196 scalar/structure checks. Numerical tolerances are relative `1e-12` and absolute `1e-10`.

Portable replay explicitly reports `original_image_bytes_verified: false`, count `0`, and `checkpoint_bytes_verified: false`. It checks recorded identifiers and saved-prediction arithmetic; it neither re-observes absent validation images nor proves anew that a checkpoint generated those arrays. The distinct fresh-inference record verified the actual checkpoint and image bytes before making predictions.

`audit/public-replay.json` records the completed arithmetic audit of this derivative. `historical/audit/full-audit.json` preserves the stronger earlier audit of the original record that also checked all 20 original validation PNG hashes. Historical audit/index records identify their original scope; the active inventories identify the public derivative. To repeat the image checks with those files available, replace `--skip-image-bytes` with:

```bash
--image-root /path/to/original-validation-pngs
```

This still replays stored predictions; it does not rerun inference.

## Pinning and provenance

| Input | Immutable identity |
| --- | --- |
| PR58 archive source commit | `ffa3865fb59e56d603b53de5ceb797f9f459badd` |
| PR58 archive SHA256 | `9c59ed51d733c5d32a176f2dbd2d803c9d8d348c3c74a86e2f3da38bdd5664f8` |
| PR58 archive bytes | 39,780,775 |
| Training source revision | `082ea773426b6422f859c68b8b6ec8496ef07df0` |
| Clean source used for fresh evaluation | `036479af6152dab0a4d20edb156ffb0a2fbff243` |
| Frozen protocol SHA256 | `2e73d4d81585e8ed1759bdb02aea98d8c9db3a856f8887ef9f41069eb799fa3f` |
| Primary reference manifest SHA256 | `75e8689f65f7f32ca7f4657f27d214aa1ba371347812de780c6e5e6ac21af61e` |
| Sensitivity reference manifest SHA256 | `28ba06fe7183ebbfa6f295639194a67c4b473c17a8d67c28a9733eb359bdba92` |
| PR35 checkpoint SHA256 | `420c00f73eeba24e587342400878b7d96371d3e745772113809aa705ec8163b1` |
| Matched-control checkpoint SHA256 | `a3d9fdc2cf1fa067340ad4f38c9a345e7d480c298852e56b992adaeb9c8d1387` |
| Candidate checkpoint SHA256 | `7c5f3114dfaeff58a873647d31e1be26e16a1941d3d20890c6a613ab853e41fb` |
| Executed driver SHA256 | `9757801574de9c77714e65cedcd41e2f01d22740eba584996fc7cb5f8c68ed06` |
| Independent auditor SHA256 | `64ce88cf38361ddbc568077913bd2cefd6a1bde907b32221bc9337428892c761` |

The unmodified composed-result verifier accepted both 30/30-epoch returns and their matched recipe, staging, source and optimizer-step evidence. Its record is `provenance/return-verification.json`. Original returned receipts, staging manifests, CSVs and logs are preserved under `provenance/pr58/`; the selected PR35 provenance is under `provenance/pr35/`.

`provenance/prior-prediction-reproduction.json` records exact equality of all 20 PR35 arrays and all 20 matched-control arrays with their corresponding prior PR53 outputs. Differing checkpoint archive hashes are not described as byte identity or a universal cross-runtime reproducibility guarantee.

## Repeat fresh inference separately, if needed

This is a reproduction recipe for the completed evaluation, not a new GPU handoff.

1. Use a clean checkout at `036479af6152dab0a4d20edb156ffb0a2fbff243`, with Git LFS active and all 20 original `data/images/val/*.png` files materialized. Keep the source, original labels, candidate overlays and references unchanged.
2. Download the [immutable PR58 archive](https://github.com/Siuuuers/dl/blob/ffa3865fb59e56d603b53de5ceb797f9f459badd/results/gpu/baomai-composed-fronts-l4-001/baomai-composed-fronts-seed46-results.zip), verify its SHA256 above, and extract its `reference` and `candidate` bundles.
3. Download the [immutable PR35 archive](https://github.com/Siuuuers/dl/blob/8e8bb99cabcb44ed0ec0acd14db911aa44013fd8/results/gpu/baomai-reviewed-annotations-seed46-l4-001/baomai-reviewed-annotations-seed46-results.zip). Its SHA256 is `df53683ccf6f266a605b2844f42a98e808f75686cfab09f14d680c9d061e8a91`. Extract it and use `candidate/checkpoint-manifest.json`.
4. Consult `runtime/requirements-inference.lock.txt` inside the capsule. The actual run used Python 3.12.14, Torch `2.10.0+cpu`, torchvision `0.25.0+cpu`, Ultralytics `8.4.182`, FP32 and CPU with four Torch threads. The observed 40-distribution inference version list is retained; the full host inventory is omitted. A fresh installation from that lock was not separately tested. It is distinct from PR58's L4 training runtime.
5. Run the exact included driver, writing to a new directory outside the source checkout:

```bash
python -B /path/to/extracted-capsule/driver/evaluate_comparison.py \
  --root /path/to/clean-036479-checkout \
  --returned-root /path/to/extracted-PR58 \
  --current-manifest /path/to/extracted-PR35/candidate/checkpoint-manifest.json \
  --output /path/to/new-fresh-evaluation
```

The driver pins its source helpers, protocol, references and checkpoints; checks a clean source tree; disables network activity during evaluation; and performs 60 real-image predictions plus three excluded blank warmups. It fixes box confidence at `0.13`, NMS IoU at `0.70`, endpoint confidence at `0.50`, input size at `1280`, and box-matching IoU at `0.50`. No training is run. Single-call CPU timing is descriptive, not a deployment throughput benchmark.

## Visual review and the annotation pilot

- `review/changed-cases/`: records for eight changed object identities across the three references, source hashes, crop geometry and root review decisions. `new_000010:28` retains separate primary and sensitivity records.
- `review/proposals/`: records for all 23 outputs selected by the fixed proposal association, original-first observations, geometry and decisions. The selection is separate from official scoring; unpaired does not mean a new physical object.
- `review/training-pilot/`: the six-image selection saved before viewing, original-image/label hash checks and historical findings. At that pilot's completion, two omitted front identities needed coordinate adjudication and one top-bound proposal was supported; no dataset was changed or trained. The later [border-front adjudication](../../review/border-fronts-20261113/README.md) completes those three changes separately.

PNG paths inside preserved historical records refer to images in the original capsule and are intentionally absent from this derivative. Their omission is enumerated in `public-derivation.json`; it does not imply that visual review was rerun or that historical pending decisions were rewritten.

These reviews are original-first assistant judgments, not fully blinded independent human certification. Nearest-neighbor enlargements preserve the source pixels without adding detail. Scoring replay cannot certify visual identity, unobservable corner coordinates, generalization or model superiority. The report keeps the fixed official scores and their physical interpretation distinct.

## Historical review revisions and the corner diagnostic

The unchanged annotation records name historical local review revision `85ce9515c172d9563051c4472a1479a150c00ee4`; the [corner requirement diagnostic](../../review/corner-requirement-20261113.json) names `6fa530035030d6703e80839700506e05d9c921b0`. Those local commits are not published. Their relevant original labels, source manifest, PR35 parent, annotation policy, renderers, validators and scorer inputs are byte-identical to public base `036479af6152dab0a4d20edb156ffb0a2fbff243`. The public branch includes the reviewed outputs and preserves their original recorded hashes.

The corner diagnostic is the original saved-output analysis, so its `inputs` identify the original prediction file. Its `evidence_capsule` path now provides the documented public derivative. The mapping is explicit:

| Input | SHA256 |
| --- | --- |
| Original `predictions-current.jsonl` | `2435dad151751745a0019cbbbe9e34113236db1da59b49434e8d0708a1eba3d9` |
| Public derivative `predictions-current.jsonl` | `e656f23186671aff7777cc178ebbfca9f2c69f90a21052156bb1a596373dfcb5` |
| Concatenated 20 per-image record lines, identical in both | `9e998e690095730536ae23df1bba2964f112039ae21e5defbb8e15b6b3162b8a` |
| `scoring-current-primary.json`, unchanged | `e3c3a9665754382b38be38e126574dfeae17e7dad888e50009406a03178129d4` |

Only the prediction run header's workspace prefix changes. The 715 drawable-pair counts, all coordinates/confidences and all reference assignments are unchanged. No new inference was performed, and no prediction or scoring result changed during publication preparation.
