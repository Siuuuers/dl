# Expanded supplementary-data pilot r47 T4 results

Provenance for the returned archive:

- Experiment: additional-front-pilot-r47, expanded-data equal-epoch pilot (`schema` `baomai-r47-expanded-data-equal-epoch-v1`). Both arms start independently from `weights/cassette_lower_epoch173.pt` (staging `initializer` SHA-256 `e0aab1b377491700318e0b14cd4c558ec3c1684087a13b7dde2298ebdbf2847e`; receipt model path `/content/baomai-r47/work/initializer/cassette_lower_epoch173.pt`) with recipe `reviewed-incremental-fronts-seed46-v1` (recipe SHA-256 `f18e0a813a1057e1b3ab3a44ea3ce9b2a3e7673cc2bf04ca09b394bfdd8c51b8`). Reference: 138 photos / 3827 fronts / 1050 batches / 78 optimizer updates / 4140 presentations. Candidate: 142 photos / 3999 fronts / 1080 batches / 80 updates / 4260 presentations, adding complete R47 photos `frame_0146`, `frame_0160`, `frame_0178`, `frame_0371`, `frame_0383`. Shared `frame_0383` labels are used in both arms (both train label files, and `review/complete-candidates/labels/frame_0383.txt`, are 4042 bytes, SHA-256 `1848d4dad3185444726de644f5303ee548fc74e08d4361d11c8bd8220fece7b0`, matching staging `preserved_baseline_extra_label_sha256`). Held photographs (`frame_0001`, `frame_0038`, `frame_0074`, `frame_0128`, `frame_0225`, `frame_0280`, `frame_0301`) are excluded as whole images. The original 20 development photos and the three evaluation references remain unchanged (both roles have the same 20 val photos; staging `all_original_20_development_photos_excluded_from_training` is true). Staging `isolated_data_from_compute` is false: this does not isolate new data from compute. Staging also records `equal_epochs` true, `equal_optimizer_updates` false, and `automatic_promotion` false. This results pack does not merge, close, or edit earlier PRs, including draft PR 136.
- Source revision: `96b05f0d2149ab71c9211c81f35be67f79344b81`, printed by Cell 2 as `SOURCE_REVISION` and stored as receipt/`staging.json`/`evaluation/summary.json` `source`. `RELEASE_SHA256` `1d4f3ac4cb3efc97c6e05ef392b83654d4eed2039f3f167340a425ff837641a7` (matches `staging.json` `release_sha256`).
- Notebook: `notebooks/baomai_additional_front_pilot_r47.ipynb`, pinned publish commit `d0136ae8b2283d38da953e8f109e542029682678` ("R47: publish source-pinned notebook and fresh CPU execution proof"), 11603 bytes, SHA-256 `35913696174310fb9e7e16ee26672e5aa56b1b16b81cc63460d77e69b6ef065a`, LF-only, unchanged. Drive copy run unchanged: https://colab.research.google.com/drive/1yxaR75pt8lbGMalQPBOPbOVctT1GarRM?authuser=1 (authuser=1, GloriaWingyin@gmail.com). `USE_DRIVE` True. Drive backup `/content/drive/MyDrive/baomai-r47/96b05f0d2149ab71c9211c81f35be67f79344b81/` matched the ZIP hash.
- Runtime: one Colab Tesla T4 High-RAM session. A100 High-RAM was requested; Colab fell back to an authorized T4, which was kept. Hostname `fa83304623e7` and the NVIDIA-SMI / UUID / session times below were observed in the run account / Colab terminal; they are not stored in the receipts (receipts have no `device_name`, `device_actual`, `hostname`, hours, wall, or `created_at` keys). NVIDIA-SMI / Driver 580.82.07, CUDA 13.0; GPU 0 Tesla T4 15360 MiB, UUID `GPU-60201e96-90f7-b293-3225-1271c1410655`; System RAM 51.0 GB, GPU RAM 15.0 GB. Receipt and `evaluation/summary.json` runtime block: gpu Tesla T4, torch 2.14.1+cu130, python 3.13.16, cuda 13.0, cudnn 92800, ultralytics 8.4.182. Settings in receipts: amp false, batch 4, imgsz 1280, optimizer AdamW, seed 46, device 0. The runtime was released; Manage Sessions was empty; Reconnect was not clicked.
- Resource-trial: status `trial-complete`, role candidate, discarded, Drive-backed. Trial receipt: `trial` true, batches 4, images 16, steps 4; `artifacts` is null (no trial `last.pt`). Reference complete at 12:54:20 PM HKT / Asia/Hong_Kong on 27 Nov 2026 (04:54:20 UTC): 1050 batches, 30 epochs, 78 steps, 4140 images, status `complete`, `trial` false. Candidate complete at 1:08:01 PM HKT / Asia/Hong_Kong on 27 Nov 2026 (05:08:01 UTC): 1080 batches, 30 epochs, 80 steps, 4260 images, status `complete`, `trial` false. Training logs record reference `30 epochs completed in 0.211 hours` and candidate `30 epochs completed in 0.215 hours`. Receipt `staging_sha256` and `staging.json` SHA-256 `6440e7e75976ac7b8f9bd94651c8c9b4a9443a972f4bc2d68f8573871954ae29`.
- Archive: `baomai-additional-front-pilot-r47-results.zip`, 42395650 bytes, SHA-256 `fc8c40357c83831d17f88b0bb5a5ecd2e0cb1a8128be697e37e80620e4981023`. The zip contains 500 members: `inventory.json` plus 499 inventory entries. Members include both 30-epoch `last.pt` files (21704686 bytes each), receipts/progress, logs, `evaluation/summary.json`, `evaluation/predictions.json`, and dataset labels. `unzip -t` and `ZipFile.testzip()` reported no errors. This matches the Cell 8 Return line.

Independently measured archive members (SHA-256 and size). These values were hashed from the extracted zip members and match the hashes recorded in the receipts and `inventory.json`:

- `candidate/fit/weights/last.pt`, 21704686 bytes, SHA-256 `222bf3c91a651183047a10eb1a50e16ae8fe999f15372afe005bfca2d0b157f9`
- `reference/fit/weights/last.pt`, 21704686 bytes, SHA-256 `605f067d6fc43ed263f956190132371921228ac93efb0c169f22ff14941c0308`
- `candidate/receipt.json`, 4815 bytes, SHA-256 `d8f9719add319fcfab5084cef39d77721c759ff4bbc1998e167c7715c3a1e8be`
- `reference/receipt.json`, 4795 bytes, SHA-256 `b311a5bd4ae2d94454e67df99bb80e9e33cc04429f42846c17f8e34a858e27e8`
- `resource-trial/receipt.json`, 3713 bytes, SHA-256 `6024c28bf00dd592a3630a30a8e45de987d8f73505ffe2146e4675735dbeb895`
- `candidate/progress.json`, 4815 bytes, SHA-256 `d8f9719add319fcfab5084cef39d77721c759ff4bbc1998e167c7715c3a1e8be`
- `reference/progress.json`, 4795 bytes, SHA-256 `b311a5bd4ae2d94454e67df99bb80e9e33cc04429f42846c17f8e34a858e27e8`
- `resource-trial/progress.json`, 3713 bytes, SHA-256 `6024c28bf00dd592a3630a30a8e45de987d8f73505ffe2146e4675735dbeb895`
- `candidate/fit/results.csv`, 6048 bytes, SHA-256 `18b9af092455f00c54db0102b104dd2bc44fa8c1021596c238b3406c2a718ea7`
- `reference/fit/results.csv`, 6052 bytes, SHA-256 `60af198740e50923b02f599df6acf2d1918b86a350bc2d3789623c8c825e4eb6`
- `candidate.log`, 232882 bytes, SHA-256 `b8cae5b1eb85861139d3d2fa5cd209d390d2ecc141f40823970f8773061853a4`
- `reference.log`, 229004 bytes, SHA-256 `df19d61b5213021a0e8b49c5b6d1947808b0f9b93569e96e011bbb0462178f03`
- `resource-trial.log`, 9421 bytes, SHA-256 `ab5b9299ed06966fd0001802555577414054f7fdedbd8716703670cdf6ff31a7`
- `evaluation/summary.json`, 2497971 bytes, SHA-256 `4496597be1a97615825f8c176a5b0618e248b7d80035538e5a498f5eb36276d5`
- `evaluation/predictions.json`, 1680186 bytes, SHA-256 `82ddb64a7f58492dfa49ff0953cc7f5778ad5da36fdc4f3f66b495393dcadca7`
- `inventory.json`, 77976 bytes, SHA-256 `a032996264fe858888100ee0e4ebd02767ca6f341284e6d87729de1f268ce4e7`
- `staging.json`, 182055 bytes, SHA-256 `6440e7e75976ac7b8f9bd94651c8c9b4a9443a972f4bc2d68f8573871954ae29`
- `datasets/candidate/labels/train/frame_0146.txt`, 4102 bytes, SHA-256 `533da8e4c77be0ecf16be8957c2e75ede93c29c9dc5e931a6337e8014177e907`
- `datasets/candidate/labels/train/frame_0160.txt`, 4042 bytes, SHA-256 `0fc63ecdceb96ea26ec3ae17eccb4365bb1442325de0c7076e997778e19366e3`
- `datasets/candidate/labels/train/frame_0178.txt`, 4072 bytes, SHA-256 `34545a83102bddfea1de5e64a4a65dabfdb2e7d2a6a21e3e3af0876c9850d5f2`
- `datasets/candidate/labels/train/frame_0371.txt`, 4042 bytes, SHA-256 `83c5ab727ee95c937cbc7f91afe38b832d0b0eb6a8dc25664afeb396f477e2f5`
- `datasets/candidate/labels/train/frame_0383.txt`, 4042 bytes, SHA-256 `1848d4dad3185444726de644f5303ee548fc74e08d4361d11c8bd8220fece7b0`
- `datasets/reference/labels/train/frame_0383.txt`, 4042 bytes, SHA-256 `1848d4dad3185444726de644f5303ee548fc74e08d4361d11c8bd8220fece7b0`

Evaluation (from Cell 8, also recorded in `evaluation/summary.json`; the Cell 8 postclip corner-eval lines below were checked against that summary, using `corners_by_locatable_count.2.eligible` as the denominator, and are recorded here, not interpreted as a promotion). Inference settings from `evaluation/summary.json`: references frozen666, historical663, r38; imgsz 1280, conf 0.13, iou 0.7, matching_iou 0.5, max_det 300, nms true, batch 1.

```
r38 reference misses 13 unmatched 90 duplicate subset 18 other 72 both within5px 569 / 583 within10px 581
r38 candidate misses 15 unmatched 84 duplicate subset 14 other 70 both within5px 566 / 583 within10px 582
frozen666 reference misses 33 unmatched 98 duplicate subset 22 other 76 both within5px 518 / 584 within10px 575
frozen666 candidate misses 34 unmatched 91 duplicate subset 17 other 74 both within5px 511 / 584 within10px 575
historical663 reference misses 30 unmatched 98 duplicate subset 22 other 76 both within5px 518 / 584 within10px 575
historical663 candidate misses 30 unmatched 90 duplicate subset 17 other 73 both within5px 511 / 584 within10px 575
Return SHA256: fc8c40357c83831d17f88b0bb5a5ecd2e0cb1a8128be697e37e80620e4981023
```

Raw counts from `evaluation/summary.json` (clearly labeled; not the Cell 8 printed postclip lines; also not a promotion):

```
r38 raw reference misses 13 unmatched 98 duplicate subset 26 other 72 both within5px 569 / 583 within10px 581
r38 raw candidate misses 15 unmatched 93 duplicate subset 22 other 71 both within5px 566 / 583 within10px 582
frozen666 raw reference misses 33 unmatched 106 duplicate subset 29 other 77 both within5px 518 / 584 within10px 575
frozen666 raw candidate misses 34 unmatched 100 duplicate subset 24 other 76 both within5px 511 / 584 within10px 575
historical663 raw reference misses 30 unmatched 106 duplicate subset 29 other 77 both within5px 518 / 584 within10px 575
historical663 raw candidate misses 30 unmatched 99 duplicate subset 24 other 75 both within5px 511 / 584 within10px 575
```

This pack uses the same method as the r45 / r37 6-file packs: split the raw ZIP into 20971264-byte parts, wrap each raw part with ZIP_STORED (one member whose name is the raw part filename), and commit the wraps plus `baomai-additional-front-pilot-r47-results.zip.sha256`, `parts.sha256`, `chunks.sha256`, and `README.md`. The full ZIP is not committed and is not stored with Git LFS. Because this archive is 42395650 bytes, which is larger than two 20971264-byte parts (41942528 bytes), a literal 20971264-byte split yields three raw parts (`part00` and `part01` at 20971264 bytes, `part02` at 453122 bytes) and therefore three stored wraps. The wrap overhead is 202 bytes per part (it depends on the r47 part filename length and is not the r45 218-byte or r37 182-byte overhead). All three wraps start with `PK\x03\x04`.

Reconstruction: unzip each stored chunk zip, then concatenate the raw parts in order (`part00`, then `part01`, then `part02`):

```
unzip baomai-additional-front-pilot-r47-results.zip.part00.zip
unzip baomai-additional-front-pilot-r47-results.zip.part01.zip
unzip baomai-additional-front-pilot-r47-results.zip.part02.zip
cat baomai-additional-front-pilot-r47-results.zip.part00 \
    baomai-additional-front-pilot-r47-results.zip.part01 \
    baomai-additional-front-pilot-r47-results.zip.part02 \
    > baomai-additional-front-pilot-r47-results.zip
sha256sum -c baomai-additional-front-pilot-r47-results.zip.sha256
sha256sum -c parts.sha256
sha256sum -c chunks.sha256
```

The archive was reassembled that way before this pack was committed. `sha256sum -c` passed for the archive, the raw parts, and the chunk wraps. The reconstructed concatenation hashed to `fc8c40357c83831d17f88b0bb5a5ecd2e0cb1a8128be697e37e80620e4981023`. The committed chunk blobs are the raw bytes, not Git LFS pointers. The uploads/repo directory omits the full ZIP; reconstruct it from the three stored parts before hashing.

This records the returned archive. It makes no accuracy or performance claim. It is not an automatic model promotion. PR35 remains selected. Frozen 666 is the primary evaluation reference and 663 is historical/diagnostic only. The three reference sets are scored separately.
