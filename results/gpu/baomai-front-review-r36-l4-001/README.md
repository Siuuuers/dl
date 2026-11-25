# Front-review-r36 reviewed-annotation L4 results

Provenance for the returned archive:

- Experiment: `front-review-r36`
- Source revision: `83aeca67264e1f98872abeca40debc212c0775e9` (published immutable source; the notebook fetches this commit directly; there is no separate BASE_REVISION or local assembly). This results pack does not merge, close, or edit earlier PRs.
- Notebook: `notebooks/baomai_front_review_r36.ipynb`, 35547 bytes, SHA-256 `26226fc729aa2a0b29689db236e17d61ff1ffe7efb33c1deb7311903154c55db`; pinned GitHub/Colab blob a0db36122c4c03a7fa3a9115525113dc2a21636b; Colab https://colab.research.google.com/drive/1zZqxSGxOHVvG1HE33Xh52IXVHBgKByEG?authuser=1
- Protocol: `experiments/front-review-2026-11-26-r36.json` SHA-256 `aa0f50570ac06b8b9da95938955ee53413f25087e53f53779fb3fc5f317c970b`
- Runtime: Colab NVIDIA L4 High-RAM, authuser=1, hostname b88a34463905, NVIDIA L4 23034 MiB, 52.96 GB RAM, driver 580.82.07, CUDA 13.0. Both roles share the same recorded L4 runtime fingerprint.
- Roles: reference (accepted R35 training labels) and candidate (accepted R36 annotation corrections)
- Both roles are complete at 30/30 epochs, 1050 batches, 78 optimizer steps, 4110 presentations, exit code 0, final-epoch checkpoints, independently started from original epoch 173
- Archive: `baomai-front-review-r36-results.zip`, 39797960 bytes, SHA-256 `20714596f21eb9728f7a33d1535592ec18aa03980ff6108ec598e78ae96286ba`. The zip contains 15 members: 14 files declared in `gpu-results-manifest.json` plus that manifest.
- Drive folder: `/content/drive/MyDrive/baomai-front-review-r36-20261126`; https://drive.google.com/drive/u/1/folders/1i2zX9O17bYLBNKnIX1JpQ0EqPUeyaucm
- `gpu-results-manifest.json` created_at_utc `2026-11-25T19:56:12.285459+00:00` (3:56:12 AM HKT / Asia/Hong_Kong on 26 Nov 2026). The manifest and zip were written about 3:56 AM HKT on 26 Nov 2026.

Reference finished about 3:47 AM HKT on 26 Nov 2026. Recorded `reference/training-receipt.json` created_at_utc `2026-11-25T19:46:48.015793+00:00` (3:46:48 AM HKT / Asia/Hong_Kong); worker_wall_seconds `453.68530273400006`; execution wall_seconds `456.722876673`; execution sampled_peak_process_tree_rss_mib `3025.64453125`.

Candidate finished about 3:55 AM HKT on 26 Nov 2026. Recorded `candidate/training-receipt.json` created_at_utc `2026-11-25T19:55:05.208803+00:00` (3:55:05 AM HKT / Asia/Hong_Kong); worker_wall_seconds `450.6437535119999`; execution wall_seconds `453.80805786199994`; execution sampled_peak_process_tree_rss_mib `2961.29296875`.

A discarded resource trial ran 4 batches / 4 updates in about 16.5 s and was saved under Drive `/content/drive/MyDrive/baomai-front-review-r36-20261126/resource-trials/292c3a8fd5a73808270a`. It does not initialize the full runs.

The Colab NVIDIA L4 session on hostname b88a34463905 was disconnected and deleted (this r36 runtime only). Colab Manage Sessions then showed no active sessions. No other sessions were present or touched.

Independently measured archive members (SHA-256 and size). These values were hashed from the extracted zip members and match the hashes recorded in the receipts, checkpoint manifests, and `gpu-results-manifest.json`:

- `reference/fit/weights/last.pt`, 21704750 bytes, SHA-256 `68b5bcc1c3e940b0db7630938391938b6fd51aeb4475739afd478ac96fbf1070`
- `candidate/fit/weights/last.pt`, 21704750 bytes, SHA-256 `f501bb80f7e77f6a35cf72630ca5ff2a3b2169690ee5d9852919d1cbd2661ba4`
- `reference/training-receipt.json`, 21120 bytes, SHA-256 `79ad16fcf63220ca96b55fd4ac35a4479c62e19eecf50987d737efb4d22cd84b`
- `candidate/training-receipt.json`, 21128 bytes, SHA-256 `46923d6e04f752fbb579ca178b94931cb25ff42f71a9a491963755bbe4f1886b`
- `reference/checkpoint-manifest.json`, 801 bytes, SHA-256 `fc1274e99d5f36d4420b72a983e447fe0477b4b918aed7a9862ed59c0c7d017b`
- `candidate/checkpoint-manifest.json`, 801 bytes, SHA-256 `45b41190a64ea9b3d9db530bcb27101c7f6ba9ed357d616228a9c5a2607f82c2`
- `reference/staging-manifest.json`, 225648 bytes, SHA-256 `4560f922cc3371f0b53bf44ad780f635731589bc0a83de422d65768e23f3e38a`
- `candidate/staging-manifest.json`, 225648 bytes, SHA-256 `842e9d702997ddb7f89efda519a1d020a36072e98eabe01151cc7220873de139`
- `reference/fit/results.csv`, 6035 bytes, SHA-256 `9080f6f02d3c9b7c28d5846256ccf5594aa53007822b86344b96251a22647325`
- `candidate/fit/results.csv`, 6061 bytes, SHA-256 `bd21992263781163e49ab8e25693101196cf052426a1a1189a616bea14a0dff6`
- `reference/train.log`, 228872 bytes, SHA-256 `8911f2287c64525e4ff87891f06b667b5d6bf3f79e5672e574fcfbd548bd0458`
- `candidate/train.log`, 228876 bytes, SHA-256 `85169a051bcf8bdf74cf77912ec2e43c5f6bee4de7364dd635a50244f477ba30`
- `gpu-results-manifest.json`, 17246 bytes, SHA-256 `f7e7f6f3d7b94b211d3fcbbfbb4b414b8539d08257bd46e9ca0a982ceab103e9`
- `experiment-protocol.json`, 22438 bytes, SHA-256 `aa0f50570ac06b8b9da95938955ee53413f25087e53f53779fb3fc5f317c970b`
- `training-input-delta.json`, 5020 bytes, SHA-256 `cdf1a3144ee09011a4d993db4e8326592bd6c956b75b1c21897e10faf728061d`

This is the R36 reviewed-annotation comparison. Both roles independently started from `weights/cassette_lower_epoch173.pt` SHA-256 `e0aab1b377491700318e0b14cd4c558ec3c1684087a13b7dde2298ebdbf2847e` and recipe reviewed-incremental-fronts-seed46-v1 SHA-256 `f18e0a813a1057e1b3ab3a44ea3ce9b2a3e7673cc2bf04ca09b394bfdd8c51b8`. The reference uses accepted R35 training labels `data/candidates/front-review-2026-11-25-r35/manifest.json` SHA-256 `53181503da2ff636dc9c7b83f646efc051b40db10224320ad6c5694606d58b70`. The candidate uses the R36 overlay `data/candidates/front-review-2026-11-26-r36/manifest.json` SHA-256 `3ea3c07c9735cba6b407d702ca20f00877ac39069eb0f7f81995de48597b90cb`, which applies 2 eligible training-row box-only corrections on 1 photograph (new_000034 rows 34 and 35). Visibility is unchanged: visibility_changes 0 and classes_visibility_preserved true. Train inventories are identical: 3784 objects, 6837 visible / 731 missing endpoints. It is recorded in `training-input-delta.json` SHA-256 `cdf1a3144ee09011a4d993db4e8326592bd6c956b75b1c21897e10faf728061d`. Seed 46, batch 4, imgsz 1280, FP32 (amp false), AdamW lr0=0.0001, 4110 presentations. Both roles see each of 137 train images 1x/epoch. Both arms use the unmodified native Pose26 loss; partial-box factor 2 remains discontinued. Held photograph new_000015 and the 31 old-series photographs keep prior supervision. R36 development-label amendments do not enter training or trainer validation. Both roles share the same recorded NVIDIA L4 runtime fingerprint and the same training budget.

The archive was reassembled by concatenating raw `part00` then `part01` after unzipping the stored chunk zips. `sha256sum -c` passed for the archive, the chunk zips, and the raw parts. `unzip -t` and `ZipFile.testzip()` reported no errors. The committed ZIP and chunk blobs are the raw bytes, not Git LFS pointers. The uploads directory omits the full ZIP; reconstruct it from the two stored parts before hashing.

This records the returned archive. It makes no accuracy or performance claim. PR35 remains selected. Frozen 666 is the primary evaluation reference and 663 is historical/diagnostic only. Exploratory 659 and trainer 664 must not replace 666/663. automatic_promotion is false. This r36 pack records a native-loss comparison of accepted R35 training labels against the accepted R36 training-annotation delta. It does not establish new-session reliability, and it is not an automatic model promotion.
