# Front-review-r37 reviewed-annotation L4 results

Provenance for the returned archive:

- Experiment: `front-review-r37`
- Source revision: `86dea96b5294568dcc96f8912f6949d12433380c` (published immutable source; the notebook fetches this commit directly; there is no separate BASE_REVISION or local assembly). This results pack does not merge, close, or edit earlier PRs.
- Notebook: `notebooks/baomai_front_review_r37.ipynb`, 35508 bytes, SHA-256 `b793f36b546e3905aa0d2b8be3ffb018761cf610f67ec1e8547f5ca33f2a3ded`; pinned GitHub/Colab blob 6103bb872431ade34e23334a5940e4ccd1e70896; Colab https://colab.research.google.com/drive/1p0jD83R-MBouoXCBWQXPde71k2l1Vln4?authuser=1
- Protocol: `experiments/front-review-2026-11-26-r37.json` SHA-256 `46a8edcafc41d39f14d230534ef34a7499aa82191d0352d75ae64766f600c9c4`
- Runtime: Training arms were fit on the 09:28 Colab NVIDIA L4 High-RAM session, authuser=1, hostname 4d9aad985cd8, GPU UUID GPU-902771bf-75d4-054e-1080-a96ebeda9c96 (saved to Drive), then packaged on a later L4 hostname 28da6fc9404e, GPU UUID GPU-76f6ccce-745d-b9fe-2e62-73dabe31cbc1, after a completed-pair restore (Cell 7 printed `Both roles are complete; no training will repeat.`). Both machines were NVIDIA L4 23034 MiB, driver 580.82.07, CUDA 13.0, High-RAM ~53 GB. Receipts and `gpu-results-manifest.json` record device_name NVIDIA L4 and torch_cuda 13.0; they do not record hostname, GPU UUID, VRAM, driver, or host RAM. Both training arms share the same recorded receipt runtime block and the same training budget.
- Roles: reference (accepted R36 training labels / r36 candidate overlay) and candidate (accepted R37 annotation corrections)
- Both roles are complete at 30/30 epochs, 1050 batches, 78 optimizer steps, 4110 presentations, exit code 0, final-epoch checkpoints, independently started from original epoch 173
- Archive: `baomai-front-review-r37-results.zip`, 39799383 bytes, SHA-256 `41039efb8643f97fc35beb29c71679fd10d0430cdaad43fafa790849d10dac58`. The zip contains 15 members: 14 files declared in `gpu-results-manifest.json` plus that manifest.
- Drive folder: https://drive.google.com/drive/u/1/folders/1v8FuE3wfQ0zMpvjNB9auR_QT4eNcNKQv (receipts do not record a /content/drive path)
- `gpu-results-manifest.json` created_at_utc `2026-11-26T02:56:05.869014+00:00` (10:56:05 AM HKT / Asia/Hong_Kong on 26 Nov 2026). The manifest and zip were written about 10:56 AM HKT on 26 Nov 2026.

Reference finished about 9:47 AM HKT on 26 Nov 2026. Recorded `reference/training-receipt.json` created_at_utc `2026-11-26T01:47:04.300688+00:00` (9:47:04 AM HKT / Asia/Hong_Kong); worker_wall_seconds `456.1423175429999`; execution wall_seconds `459.26798160099986`; execution sampled_peak_process_tree_rss_mib `3072.27734375`.

Candidate finished about 9:55 AM HKT on 26 Nov 2026. Recorded `candidate/training-receipt.json` created_at_utc `2026-11-26T01:55:30.738425+00:00` (9:55:30 AM HKT / Asia/Hong_Kong); worker_wall_seconds `458.0662612220001`; execution wall_seconds `461.43260507700006`; execution sampled_peak_process_tree_rss_mib `2940.75`.

Independently measured archive members (SHA-256 and size). These values were hashed from the extracted zip members and match the hashes recorded in the receipts, checkpoint manifests, and `gpu-results-manifest.json`:

- `reference/fit/weights/last.pt`, 21704750 bytes, SHA-256 `69b5ee75152523aef294cba79727310ab24b12b81d482a5db0df3716417f9d57`
- `candidate/fit/weights/last.pt`, 21704750 bytes, SHA-256 `51d8a60f95b1b76097513d8b07732de5d5deed5e6e17251a28ddd28e6be1ebc1`
- `reference/training-receipt.json`, 21579 bytes, SHA-256 `1b35a408ebd5e19e881f42245cc34c17b6069085d9a3f4b3546aeb0b90f3b9b9`
- `candidate/training-receipt.json`, 21570 bytes, SHA-256 `af30fa7c7921ab25b7e62a78aea7b63b768056a87ca2538d06d778c774e79500`
- `reference/checkpoint-manifest.json`, 801 bytes, SHA-256 `8365a1ede65f247861d2f76ebe8d9e88acf13a9c8a2de30a4e97bcc26d0b77ce`
- `candidate/checkpoint-manifest.json`, 801 bytes, SHA-256 `d6d5e687786f291a8dfe31e94eefc14aa26e66f637ad7cffcadfc8bc3175cf7a`
- `reference/staging-manifest.json`, 225383 bytes, SHA-256 `4928159eb23a92f442b83a4f787e6eec51b0826a0751cf171d9a48b188cc95f6`
- `candidate/staging-manifest.json`, 225383 bytes, SHA-256 `ef343738dce1f23ee0b177de3c57bdef2b928d916aefe97dae3f3c311589c926`
- `reference/fit/results.csv`, 6055 bytes, SHA-256 `a492b2878f4c31ca85dbbc283ad5ef3d1b82081703761fdf3e89a00ec2aff87f`
- `candidate/fit/results.csv`, 6061 bytes, SHA-256 `703ebd68631ac9f6ecaa94943c8ff004c2516722000b7f547e657b80ddd01b63`
- `reference/train.log`, 228886 bytes, SHA-256 `35e351308aa6d922ebe884b9414e2b7b8844fa252b98efffaa6221ba626e1d9f`
- `candidate/train.log`, 229116 bytes, SHA-256 `b7a04c4d3c119c5b1b705a4a150fb319525d1b4e0977e1bf18ebf65666a441ed`
- `gpu-results-manifest.json`, 17699 bytes, SHA-256 `aa63672bc58e590550fd60ac4d20addda64c646f8c363f390566b9ad3aaeeed9`
- `experiment-protocol.json`, 24075 bytes, SHA-256 `46a8edcafc41d39f14d230534ef34a7499aa82191d0352d75ae64766f600c9c4`
- `training-input-delta.json`, 4624 bytes, SHA-256 `819cb1cc053bf5649fd144c0adcd9c1e9a2ef4903e931c265ac59ae5168f33d4`

This is the R37 reviewed-annotation comparison. Both roles independently started from `weights/cassette_lower_epoch173.pt` SHA-256 `e0aab1b377491700318e0b14cd4c558ec3c1684087a13b7dde2298ebdbf2847e` and recipe reviewed-incremental-fronts-seed46-v1 SHA-256 `f18e0a813a1057e1b3ab3a44ea3ce9b2a3e7673cc2bf04ca09b394bfdd8c51b8`. The reference uses the R36 candidate overlay `data/candidates/front-review-2026-11-26-r36/manifest.json` SHA-256 `3ea3c07c9735cba6b407d702ca20f00877ac39069eb0f7f81995de48597b90cb`. The candidate uses the R37 overlay `data/candidates/front-review-2026-11-26-r37/manifest.json` SHA-256 `ccfe09cae8e890b80896dac5d025f981573e5da3f49bb9f40175f2d0685de2f3`, which applies 2 eligible training-row corrections on 2 photographs (new_000006 row 53 box/xmax and new_000034 row 35 left-corner). Visibility is unchanged: visibility_changes 0 and classes_visibility_preserved true. Train inventories are identical: 3784 objects, 6837 visible / 731 missing endpoints. It is recorded in `training-input-delta.json` SHA-256 `819cb1cc053bf5649fd144c0adcd9c1e9a2ef4903e931c265ac59ae5168f33d4`. One inherited pending val request remains: new_000010 row 24 (preserve_row, not in training). Seed 46, batch 4, imgsz 1280, FP32 (amp false), AdamW lr0=0.0001, 4110 presentations. Both roles see each of 137 train images 1x/epoch. Both arms use the unmodified native Pose26 loss; partial-box factor 2 remains discontinued. Held photograph new_000015 and the 31 old-series photographs keep prior supervision. R37 development-label amendments do not enter training or trainer validation. Both roles share the same recorded receipt runtime block and the same training budget.

The archive was reassembled by concatenating raw `part00` then `part01` after unzipping the stored chunk zips. `sha256sum -c` passed for the archive, the chunk zips, and the raw parts. `unzip -t` and `ZipFile.testzip()` reported no errors. The committed ZIP and chunk blobs are the raw bytes, not Git LFS pointers. The uploads directory omits the full ZIP; reconstruct it from the two stored parts before hashing.

This records the returned archive. It makes no accuracy or performance claim. PR35 remains selected. Frozen 666 is the primary evaluation reference and 663 is historical/diagnostic only. Exploratory 659 and trainer 664 must not replace 666/663. automatic_promotion is false. This r37 pack records a native-loss comparison of accepted R36 training labels against the accepted R37 training-annotation delta. It does not establish new-session reliability, and it is not an automatic model promotion.
