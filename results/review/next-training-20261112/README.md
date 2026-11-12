# Next training proposal: combine the reviewed front corrections

**Status: reviewable compatibility proposal only. This is not an accepted, staged or GPU-ready training dataset.** No inference or training was executed for this preparation, and no new visual certification of training images is claimed. PR35 remains the selected model.

The [operation specification](combined-operations-spec.json) and three [prototype label files](compatibility-prototype/labels/train/) make the proposed next intervention concrete. They reconcile previously recorded decisions by original source row ID, preserve the historical files, and pass label parsing and geometry checks. They do not yet implement the required training-data contract.

## One controlled question

With the accepted endpoint supervision held fixed, do the previously untested PR49 front-box corrections and duplicate removal improve detection without losing supported successes?

| Role | Effective supervision | Train objects | Visible endpoints | Missing slots |
| --- | --- | ---: | ---: | ---: |
| Selected PR35, operational comparator | Original labels plus the exact eight-file `annotations-2026-11-10` overlay | 3,771 | 6,841 | 701 |
| Proposed control | Current **ten-slot** endpoint candidate, including those eight inherited files | 3,771 | 6,842 | 700 |
| Proposed intervention | That same control plus **17 box corrections and one duplicate deletion** from PR49 | 3,770 | 6,842 | 698 |

All three views retain 137 training images and 13 empty training images. The separately supplied original validation split remains 20 images and 664 objects. The three prototype files supplement the existing eight-file parent; they are not a complete dataset on their own.

The [current endpoint candidate](../../../data/candidates/endpoint-visibility-2026-11-12/README.md) contains **ten changed endpoint slots**, including `new_000023` original row 33's right corner at `(612,549)`. PR52 already tested this ten-slot intervention and [retained PR35](../../../docs/endpoint-results-2026-11-12.md). The older nine-slot recovery file must not overwrite that later accepted point.

The PR49 [box/deletion decisions](../baomai-review-recovery-20261112/correction-record.json) affect `023`, `031` and `045`. PR47's [fourteen box corrections](../../../data/candidates/consistency-2026-11-11/manifest.json) affect `018`, `034` and `049`; these image sets are disjoint. The seventeen PR49 boxes and deletion were also excluded from PR52. The proposed comparison therefore tests a new, bounded delta rather than repeating either completed experiment. A favorable result is not guaranteed.

## Exact composition and row identity

The specification contains **28 operations**: 17 box replacements, ten endpoint-slot replacements and one deletion. Every operation has its before/after tokens, source identity and evidence link. The endpoint and box operations share two images but affect disjoint source rows. The eight PR35 parent files remain byte-identical.

| Image | Original source row | Combined row | Disposition |
| --- | ---: | ---: | --- |
| `023` | 22 | Deleted | Confirmed duplicate; both removed slots are missing |
| `023` | 33 | 32 | Preserve current endpoint pair, including right `(612,549)` |
| `023` | 34 | 33 | Preserve corrected left point and missing right slot |
| `023` | 35 | 34 | Preserve corrected left point and missing right slot |
| `023` | 36 | 35 | Retain this physical front and its reviewed box correction |
| `031` | 22–24 | 22–24 | Preserve missing left slots and reviewed visible right points |

The `031` prototype matches the older PR49 combined file exactly, and `045` matches its box proposal exactly. The `023` prototype differs from the older PR49 combined endpoint file only in combined row 32's right triple, carrying forward the later ten-slot correction. The specification records all other row mappings and the effective 137-file training inventory.

## Freeze the evaluation reference separately

The proposed evaluation reference is [`baomai-visible-front-adjudication-2026-11-12`](../../../data/references/visible-front-adjudication-2026-11-12/README.md), with [manifest](../../../data/references/visible-front-adjudication-2026-11-12/manifest.json) SHA256:

`75e8689f65f7f32ca7f4657f27d214aa1ba371347812de780c6e5e6ac21af61e`

It has **663 objects, 1,227 locatable endpoints and 584 objects with both corners locatable**. Pin this identity before any future training comparison. The evaluator must derive denominators from the manifest and include missed eligible objects in success denominators. These revised labels are evaluation inputs only; they must not enter either training arm or silently replace the original validation labels used by the historical training recipe.

The proposed numerical recipe follows the completed PR52 matched setup: independent epoch173 initialization, seed 46, 30 epochs, batch 4, input 1280, rectangular batches, FP32, AdamW and the recorded augmentation/loss settings. Both new arms must use the same numerical settings and runtime/device. Evaluate them and retained PR35 in one inference environment at the fixed 0.13 box threshold, NMS 0.70, matching IoU 0.50 and endpoint cutoff 0.50. No seed, epoch, threshold or subset sweep is proposed. A new recipe/protocol identity is still required to declare the new data contract.

Report matched-control effects separately from replacement of PR35. Review changed object and endpoint identities on originals: an existing proposal crossing an IoU boundary is not established recovery of a previously unproposed front. Preserve unresolved cases and close the experiment after its declared analyses.

## Restore existing source assets

The [availability record](image-availability.json) found no materialized original training PNG in the inspected workspaces or relevant archives. Exact training-image filenames were LFS pointers. Existing crops and earlier review records support this compatibility reconstruction but do not substitute for originals in a fresh whole-image review.

Before staging, restore **all 137 original training PNGs and 20 original validation PNGs** from the existing baomai LFS data or a verified existing GPU-input package. Verify every payload against [`data/manifest.json`](../../../data/manifest.json), including its 1280×720 dimensions; a pointer file is not an image. Restore and verify the original epoch173 checkpoint against its recorded identity before running the proposed recipe. This requires existing assets, not new camera captures.

For a correctly pinned source checkout with Git LFS access, the relevant payload paths can be requested with:

```bash
git lfs pull --include="data/images/**/*.png,weights/cassette_lower_epoch173.pt"
```

The subsequent manifest checks are required regardless of the download route. No source-payload restoration or completed staging is claimed here.

## Required explicit composition contract

Two independent integration issues prevent treating these files as ready to train.

**The standard annotation contract forbids the required endpoint edits.** The unchanged [`verify_pose_annotation_candidate.py`](../../../scripts/verify_pose_annotation_candidate.py) requires original `v=2` token triples to remain exact. A real-input probe, using all 157 source label files and a separately named manifest pinned to the current policy, failed with:

```text
original visible endpoint tokens must be preserved: new_000023:33
```

The [rejection receipt](standard-contract-rejection.json) preserves that result. New manifest names cannot alter this contract. Recasting retained objects as deletions/additions would misrepresent their lineage. Implement an explicit composed kind and verifier, plus the matching stager/recipe integration; retain the existing guards.

**Historical policy identity must remain historical.** The [PR35 parent manifest](../../../data/candidates/annotations-2026-11-10/manifest.json) pins policy SHA256 `3cda361ac4c5c9ba36b037e7e8968e0629eb3bca8d489c187beb8dd66e208a91`. The current policy adds the separate context-proposal section and has SHA256 `74dd943976a4abc22ecbf73551c69f715666060d5961b2bed81e6a9c73a3afc9`. The current strict parent verifier rejects that identity mismatch.

Validate the historical parent and endpoint candidate in an isolated verification snapshot containing their own immutable, hash-verified policy and source inputs. The new composed contract should separately pin the current policy, verify compatible observed-point semantics, and bind the exact historical control supervision. Do not replace the live policy during verification, weaken hash checks or rewrite old manifests to appear current.

### Acceptance checks before a runnable staging handoff

- Verify immutable source labels, manifests, review evidence and both historical/current policy identities. Keep historical validators unchanged.
- Prove the control's effective 137 training-label files are byte-identical to the current ten-slot endpoint candidate, including all eight PR35 parent overrides.
- Prove the intervention differs from that control only by the seventeen recorded boxes and deletion of `023` source row 22. Preserve the ten endpoint edits by source identity, source row 36 and all other row bytes.
- Reject altered evidence, missing/extra operations, wrong source-row mappings, undeclared endpoint edits, changed validation labels, malformed missing points and out-of-bounds geometry.
- Restore the real source PNGs, build fresh separate control/intervention staging directories, and independently verify their counts and file inventories. Leave the evaluation reference outside both training inputs.
- Add explicit recipe/protocol support for the composed kind, pin identical numerical settings and source/runtime identities for both arms, and verify completed-run receipts before accepting returned checkpoints.

Until those checks pass, the useful deliverable is this inspectable proposal and its exact compatibility artifacts. It is not a GPU-run request or evidence of model improvement.
