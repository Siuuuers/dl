# Front-review-r35 reviewed-annotation L4 results

Provenance for the returned archive:

- Experiment: `front-review-r35`
- Source revision: `bfe57e5ceb95ffa1d94ee56880010052ed84c86d` (published immutable source; the notebook fetches this commit directly; there is no separate BASE_REVISION or local assembly). This results pack does not merge, close, or edit earlier PRs.
- Notebook: `baomai_front_review_r35.ipynb`, 32783 bytes, SHA-256 `b0279083e7bda44ef652acb7fb7a6c67950af051a5480f5657197afeef7ed2b3`; pinned GitHub/Colab blob 635e6a75d573f87571eae99b3631b95762ec4567; Colab https://colab.research.google.com/drive/1LKZjBu6336m-Z9Zf8JyEpSna9NyRBSKY?authuser=1
- Protocol: `experiments/front-review-2026-11-25-r35.json` SHA-256 `b74cf472f42380f7f8c373d9ad9fda1e88560d16ca2fb64d079741d403c5a80d`
- Runtime: Colab NVIDIA L4 High-RAM, authuser=1, hostname e86c2f64d4f6, 52.96 GB RAM. Both roles share the same recorded L4 runtime fingerprint.
- Roles: reference (accepted R34 training labels) and candidate (accepted R35 annotation corrections)
- Both roles are complete at 30/30 epochs, 1050 batches, 78 optimizer steps, 4110 presentations, exit code 0, final-epoch checkpoints, independently started from original epoch 173
- Archive: `baomai-front-review-r35-results.zip`, 39795858 bytes, SHA-256 `2a90c3c038caa42ebbc331376671dcf4b3022d690707bd6f7906505703df1c30`. The zip contains 15 members: 14 files declared in `gpu-results-manifest.json` plus that manifest.
- Drive folder: `/content/drive/MyDrive/baomai-front-review-r35-20261125`; https://drive.google.com/drive/u/1/folders/1kqSDVNGMbjBKDoQJp-wZ1P48Mjw7tRcI. The Drive path keeps the protocol created_date suffix 20261125, even though this run finished on 26 Nov 2026 HKT.
- `gpu-results-manifest.json` created_at_utc `2026-11-25T17:54:32.195600+00:00` (1:54:32 AM HKT / Asia/Hong_Kong on 26 Nov 2026). The manifest and zip were written about 1:54 AM HKT on 26 Nov 2026.

Reference finished about 1:45 AM HKT on 26 Nov 2026. Recorded `reference/training-receipt.json` created_at_utc `2026-11-25T17:45:10.524912+00:00` (1:45:10 AM HKT / Asia/Hong_Kong); worker_wall_seconds `454.329360015`; execution wall_seconds `457.62067787499996`; execution sampled_peak_process_tree_rss_mib `2978.97265625`.

Candidate finished about 1:53 AM HKT on 26 Nov 2026. Recorded `candidate/training-receipt.json` created_at_utc `2026-11-25T17:53:26.938002+00:00` (1:53:26 AM HKT / Asia/Hong_Kong); worker_wall_seconds `451.29949236100003`; execution wall_seconds `454.365911609`; execution sampled_peak_process_tree_rss_mib `3036.453125`.

The Colab NVIDIA L4 session on hostname e86c2f64d4f6 was terminated at 2:02:03 AM HKT / Asia/Hong_Kong on 26 Nov 2026. Colab Manage Sessions then showed no active sessions. No other sessions were present or touched.

Independently measured archive members (SHA-256 and size). These values were hashed from the extracted zip members and match the hashes recorded in the receipts, checkpoint manifests, and `gpu-results-manifest.json`:

- `reference/fit/weights/last.pt`, 21704750 bytes, SHA-256 `5acba132cc4e003c728ea8786be7dae807b991f5205170ca32c70e3a006f32f5`
- `candidate/fit/weights/last.pt`, 21704750 bytes, SHA-256 `57b368387281c8ef499004f1ee49f01cefd2f4cbf1cbaa5444bee63b1115faeb`
- `reference/training-receipt.json`, 20669 bytes, SHA-256 `30fc2fb2a37e48fd8df7c221fcbac7e34e5d5492ebd211a0f883017a1d45bdf9`
- `candidate/training-receipt.json`, 20670 bytes, SHA-256 `8e739aa14d387707783e17c003e59673002dbbd252a561b67bc37b0f6b7e0805`
- `reference/checkpoint-manifest.json`, 801 bytes, SHA-256 `e2aea0e28eeea75431ef6038e7fa5e6a7792a5a142fcbb4681e63620313682d2`
- `candidate/checkpoint-manifest.json`, 801 bytes, SHA-256 `6c84c739858e9210f75ebb9538e765b5a7bb4899e7f054fce2ae9fbd095b1cd1`
- `reference/staging-manifest.json`, 226410 bytes, SHA-256 `fced7b190ee8973d37f9ded82f8552c572518f34630c24b7b3f91ad460067854`
- `candidate/staging-manifest.json`, 226410 bytes, SHA-256 `1cfce5adcaf6e6c13eb2ab0c9ce33230bf386b0fccc09579cb11cdcf34701648`
- `reference/fit/results.csv`, 6053 bytes, SHA-256 `01e478618f7f7afe92138eec215c3d7c60012c6213fa026a3d665de2e61ec0f0`
- `candidate/fit/results.csv`, 6037 bytes, SHA-256 `2d258b3ce8124439f3830c5e7aa019fa3e72cb750ed2375940aab8aa52bdd6f0`
- `reference/train.log`, 229103 bytes, SHA-256 `82796934ce44eec88ed2d2c5769d950d6c62dfb2fdcad3d8dc5f132ea2c60db3`
- `candidate/train.log`, 228880 bytes, SHA-256 `fa192891e2792e4bfa53dd8d3d90cca8b33d9897fd62750a9a91f1e11585acf6`
- `gpu-results-manifest.json`, 16793 bytes, SHA-256 `c616f7472e428965235b379e4b0110a8b9ec1433a15046ccbe2291f73fbdc6a9`
- `experiment-protocol.json`, 19606 bytes, SHA-256 `b74cf472f42380f7f8c373d9ad9fda1e88560d16ca2fb64d079741d403c5a80d`
- `training-input-delta.json`, 5974 bytes, SHA-256 `b9b8158a24ef1044c5627611e51a4cfec7fa8b00345a232b31595c02ee819797`

This is the R35 reviewed-annotation comparison. Both roles independently started from `weights/cassette_lower_epoch173.pt` SHA-256 `e0aab1b377491700318e0b14cd4c558ec3c1684087a13b7dde2298ebdbf2847e` and recipe reviewed-incremental-fronts-seed46-v1 SHA-256 `f18e0a813a1057e1b3ab3a44ea3ce9b2a3e7673cc2bf04ca09b394bfdd8c51b8`. The reference uses accepted R34 training labels `data/candidates/front-review-2026-11-25-r34/manifest.json` SHA-256 `e1af61c80d86a15dce445c546d7c4b2b7b7f8c235e6ca03e9e45eea530b9318d`. The candidate uses the R35 overlay `data/candidates/front-review-2026-11-25-r35/manifest.json` SHA-256 `53181503da2ff636dc9c7b83f646efc051b40db10224320ad6c5694606d58b70`, which applies 5 eligible training-row box-only corrections across 3 photographs (new_000006 row 53, new_000013 row 36, and new_000034 rows 34, 35, and 43). Visibility is unchanged: visibility_changes 0 and classes_visibility_preserved true. It is recorded in `training-input-delta.json` SHA-256 `b9b8158a24ef1044c5627611e51a4cfec7fa8b00345a232b31595c02ee819797`. Seed 46, batch 4, imgsz 1280, FP32 (amp false), AdamW lr0=0.0001, 4110 presentations. Both roles see each of 137 train images 1x/epoch. Both arms use the unmodified native Pose26 loss; partial-box factor 2 remains discontinued. Held photograph new_000015 and the 31 old-series photographs keep prior supervision. Development-label amendments, including removal of preserved R34 new_000009 rows 56 and 57, do not enter training or trainer validation. Both roles share the same recorded NVIDIA L4 runtime fingerprint and the same training budget.

The archive was reassembled by concatenating raw `part00` then `part01` after unzipping the stored chunk zips. `sha256sum -c` passed for the archive, the chunk zips, and the raw parts. `unzip -t` and `ZipFile.testzip()` reported no errors. The committed ZIP and chunk blobs are the raw bytes, not Git LFS pointers. The uploads directory omits the full ZIP; reconstruct it from the two stored parts before hashing.

This records the returned archive. It makes no accuracy or performance claim. PR35 remains selected. Frozen 666 is the primary evaluation reference and 663 is historical/diagnostic only. Exploratory 659 and trainer 664 must not replace 666/663. automatic_promotion is false. This r35 pack records a native-loss comparison of accepted R34 training labels against the accepted R35 training-annotation delta. It does not establish new-session reliability, and it is not an automatic model promotion.
