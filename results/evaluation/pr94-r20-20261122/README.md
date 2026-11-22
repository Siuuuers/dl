# PR94 / r20 returned-model audit

**Recommendation: retain PR35. No model was promoted.** The r20 candidate has more matching misses and fewer both-corner successes on the unchanged 666-front primary reference. Its ranking reverses on the separately pinned pre-return r20 review. Exact identity tracing shows that reference-box definitions explain many of those changes; the 35 fixed-primary misses must not be described as 35 physically absent detector outputs.

No r21 labels were loaded into this evaluation. Existing historical results are preserved.

## Fixed measurements

All models used the same CPU inference runtime and operating point: input1280, box confidence0.13, endpoint confidence0.50, NMS IoU0.70, matching IoU0.50, FP32, max300 outputs, four threads. The pair-success denominator is584 fronts with two locatable corners, including missed objects as failures. An unmatched output is a matching outcome; it is not automatically an independently confirmed background false positive. Duplicates are a subset of unmatched outputs.

| Reference | Model | Matched | Misses | Unmatched outputs | Duplicates | Both corners ≤5px | Both corners ≤10px |
|---|---|---:|---:|---:|---:|---:|---:|
| Frozen666 primary | Retained PR35 |648|18|80|32|533/584|582/584|
| Frozen666 primary | Fresh r20 control |646|20|83|29|531/584|583/584|
| Frozen666 primary | r20 candidate |631|35|103|29|520/584|574/584|
| Historical663 | Retained PR35 |648|15|80|32|533/584|582/584|
| Historical663 | Fresh r20 control |646|17|83|29|531/584|583/584|
| Historical663 | r20 candidate |631|32|103|29|520/584|574/584|
| Pre-return r20/662 exploratory | Retained PR35 |625|37|103|21|538/584|573/584|
| Pre-return r20/662 exploratory | Fresh r20 control |624|38|105|16|534/584|573/584|
| Pre-return r20/662 exploratory | r20 candidate |648|14|86|30|549/584|583/584|

All three models return zero outputs on all three empty scenes. On666 the candidate misses13 zero-corner,14 one-corner and8 two-corner fronts. On r20/662 it misses7 zero-corner and7 one-corner fronts, with all584 two-corner fronts matched. Candidate miss-free nonempty scenes are4/17 on666 and10/17 on r20/662. These reused development images do not establish reliability in unseen sessions.

Compared with the fresh control on666, the candidate gains2 and loses17 matched identities. Compared with retained PR35, it gains2 and loses19. The complete identities, box overlaps and predictions are in [fixed-comparison.json](fixed-comparison.json). The primary and historical references retain their8 and13 unresolved rows respectively; those rows remain in the scores. The r20 exploratory ledger has no unresolved development rows, which is a recorded review status rather than certification of every physical contour.

## Why the ranking changes

[reference-sensitivity.json](reference-sensitivity.json) follows each instance through the hash-bound recovery and review row maps, replaying the recorded edits. It does not equate row numbers across versions or infer identities from nearest boxes. Of666 primary identities,660 survive,6 are deleted, and2 distinct identities are added, producing662 r20 objects.

The candidate's35 primary misses divide into:

- **16 reference-box crossings:** the exact same returned prediction has IoU below0.5 against the fixed box and at least0.5 against the r20 box for the same retained annotation identity.
- **6 deleted primary identities:** these no longer appear in the r20 reference.
- **13 retained identities still unmatched** in r20.

One previously matched retained identity becomes unmatched after reference tightening. Both added r20 identities match. Thus the candidate moves from631 matches to648 without changing any model output. The primary decision remains unchanged; the decomposition explains the metric's sensitivity, not the independent correctness of either reference.

Original-pixel representative inspection supports that distinction:

| Same candidate output | Fixed→r20 IoU | Pixel observation |
|---|---:|---|
|057 primary row12→r20 row11 |0.473908→0.767834|A visible output lies on the same white front. The fixed box extends into dark material beneath it. Reference corners are unchanged; errors are0.184px and0.930px.|
|090 row41 |0.468312→0.574950|The fixed box spans a broad rack strip. The r20 box is narrower, but the candidate still extends into that strip and returns a right point over rack material. Only the left reference corner is locatable.|
|010 row32 |0.183625→0.541570|A hand occludes most of the front. The fixed box spans the hand, while the newer box covers a much narrower visible sliver. Neither reference corner is locatable; corner accuracy cannot be certified.|
|011 primary row58→r20 row57 |0.514757→0.285193|The converse case: the candidate extends beyond a narrow visible front into a gray rack strip. The tighter reference exposes an oversized-box error; its locatable left point is still2.556px from the r20 point.|

The four raw/overlay triptychs and observation records are preserved under [representative-cases](representative-cases) and [representative-visual-notes.json](representative-visual-notes.json). A box match is not proof of a physically correct complete contour or a valid hidden corner.

## Returned artifact and training provenance

The audit read [PR94](https://github.com/Siuuuers/dl/pull/94) at head `b547a541d6473e6432ee3a5628d0581200d337ec`. Its seven changed files only return the archive and integrity documentation. The direct Git archive is39782532 bytes, SHA256 `d9647d3a84904d4a5de5b1b6f826ff50366f462c6cb61f7f67448d58e71c38ef`. CRC, safe unique paths, exact13-member inventory and all12 indexed payload hashes/sizes pass. Chunk reconstruction was unnecessary because the exact full archive was available directly.

The source is `7207bcf4728140e03ef3fe3e6bce6918692d0a9e`; protocol SHA256 is `caa524e65dc4fc44cee520e150f7a3b60f5dddabfd31a7f4a2142a137ff0482f`. The final notebook at `bf13f8c293c316b62ede247324ef1e4e45c979a7` has SHA256 `7bea79f1f115a57764d7eaa0f8f13be876debcdf853d4f947f54fce5f8700c3f` and pins that source/protocol exactly.

Both runs independently initialize from the original epoch173 weights, complete30 epochs/1050 batches/4110 image presentations/78 optimizer steps, and exit0. Both use `reviewed-incremental-fronts-seed46-v1`, seed46, batch4, input1280, FP32, AdamW at0.0001, final-epoch selection and the same NVIDIA L4 runtime. Wall times are460.459170437s for control and449.358720554s for candidate. The fresh pair is internally matched. PR35 trained on an A100, so cross-run differences cannot be attributed solely to labels.

All157 original images, initialization weights, staged source bytes, source-script hashes and both arm receipts were independently verified. There are92 differing training-label files. Both arms preserve original664-front trainer validation, and photo015 plus31 old-series photos retain identical PR35 supervision. Control training has3771 fronts/6841 locatable endpoints; candidate has3784/6838. The source's held/excluded distinctions remain intact.

| Checkpoint | SHA256 |
|---|---|
| PR35 |`420c00f73eeba24e587342400878b7d96371d3e745772113809aa705ec8163b1`|
| Fresh control |`6253c516991e33887a56edfe8aee76981fb445acfb1698941e6b44d5a3427496`|
| r20 candidate |`8cfdc473004c835eef50a58befef7316faa9d22d77a396d6ac3a75ef0c5c50ac`|

No training was repeated. CPU inference uses Python3.12.14, Torch2.14.1+cpu, torchvision0.29.1+cpu, Ultralytics8.4.182, NumPy2.4.6, Pillow11.3.0 and OpenCV5.0.0.93. Retained PR35 prediction arrays exactly reproduce the earlier PR92 evaluation on all20 photos. Fresh control outputs differ from PR35. The isolated inference checkout records158 materialized LFS image/weight paths as modified because it lacks an LFS clean filter; every LFS oid, size and manifest hash was rechecked. No scorer or label changed.

## Replay and notebook check

[audit_return.py](audit_return.py) validates the return against a checkout of the exact source, hydrated original inputs, the original ZIP and extracted bundles. It also requires the pinned notebook commit to be available as a Git object. [run_pr94_inference.py](run_pr94_inference.py) repeats the three fixed CPU inferences when complete checkpoint bundles and the pinned runtime are supplied. [measure_pr94.py](measure_pr94.py) replays all three references from the preserved predictions and text receipts without loading weights. Its arguments mirror the PR92 runner; use `--root`, three prediction paths, three `--*-manifest` paths and a fresh `--output`. [reference_sensitivity.py](reference_sensitivity.py) reproduces the identity/IoU decomposition from `--root`, `--evidence` and a fresh `--output`.

The original measurement machinery is retained; [measurement-adaptation.patch](measurement-adaptation.patch) documents the r20 pin/import changes and additional pair/notebook checks. Receipt logs are byte-preserved as `train-log.txt`; the scorer restores their original `train.log` name in a temporary directory for strict validation.

The suspected cell6 constraints-file newline issue was checked against the exact notebook blob. Decoded Python AST and isolated expression evaluation show **two real newlines**, not literal backslash-n text. [notebook-constraints-check.json](notebook-constraints-check.json) records the check. No repair to this expression is indicated.
