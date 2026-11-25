# Front-review-r34 reviewed-annotation L4 results

Provenance for the returned archive:

- Experiment: `front-review-r34`
- Source revision: `84b568f16f539b2189802d13726d528a0f21e237` (published immutable source; the notebook fetches this commit directly; there is no separate BASE_REVISION or local assembly). This results pack does not merge, close, or edit earlier PRs.
- Notebook: `baomai_front_review_r34.ipynb`, 32659 bytes, SHA-256 `1d7a0556bb95eed4ae3611b7428e1f6f19d9a2c2eb7eab5fc1ea1cb6b902b93c`; pinned GitHub/Colab blob 9b69f1f4fd4e6dc2ad1ca1f41d7667fa9504151a; Colab https://colab.research.google.com/drive/17Z3NYh4wAsCdPY3GNK8j5N7Q4Tu7yZZ3?authuser=1
- Protocol: `experiments/front-review-2026-11-25-r34.json` SHA-256 `05a23cbef0fab6a2b5be1e1b7c8aafc4a261ab32f13cde26d13d68bbff3e3277`
- Runtime: Colab NVIDIA L4 High-RAM, authuser=1, hostname 96f754117bef, 52.96 GB RAM. Both roles share the same recorded L4 runtime fingerprint.
- Roles: reference (accepted R33 training labels) and candidate (accepted R34 annotation corrections)
- Both roles are complete at 30/30 epochs, 1050 batches, 78 optimizer steps, 4110 presentations, exit code 0, final-epoch checkpoints, independently started from original epoch 173
- Archive: `baomai-front-review-r34-results.zip`, 39795736 bytes, SHA-256 `1dc96bbcc798773ae848307911858382054a6768cf9105a545e501cb9218462f`. The zip contains 15 members: 14 files declared in `gpu-results-manifest.json` plus that manifest.
- Drive folder: `/content/drive/MyDrive/baomai-front-review-r34-20261125`; https://drive.google.com/drive/u/1/folders/1izQEYTbrIj3rl5uHx8OURWA99_z1CGHV
- `gpu-results-manifest.json` created_at_utc `2026-11-25T14:03:10.606662+00:00` (10:03:10 PM HKT / Asia/Hong_Kong). The manifest and zip were written about 10:03 PM HKT on 25 Nov 2026.

Reference finished about 9:53 PM HKT on 25 Nov 2026. Recorded `reference/training-receipt.json` created_at_utc `2026-11-25T13:53:39.672120+00:00` (9:53:39 PM HKT / Asia/Hong_Kong); worker_wall_seconds `463.63516869200004`; execution wall_seconds `467.10492024300004`; execution sampled_peak_process_tree_rss_mib `2979.1484375`.

Candidate finished about 10:02 PM HKT on 25 Nov 2026. Recorded `candidate/training-receipt.json` created_at_utc `2026-11-25T14:02:06.264078+00:00` (10:02:06 PM HKT / Asia/Hong_Kong); worker_wall_seconds `460.99739766799996`; execution wall_seconds `464.380393541`; execution sampled_peak_process_tree_rss_mib `2957.046875`.

Independently measured archive members (SHA-256 and size). These values were hashed from the extracted zip members and match the hashes recorded in the receipts, checkpoint manifests, and `gpu-results-manifest.json`:

- `reference/fit/weights/last.pt`, 21704750 bytes, SHA-256 `f472cdb8d239a3dedc314d646a841b9876f99f440b497d42f0f72ec2d1123fb9`
- `candidate/fit/weights/last.pt`, 21704750 bytes, SHA-256 `d1580dab164de3db58839dc0dbc54b903c5abba804e24322fdd51072f842d6c0`
- `reference/training-receipt.json`, 20217 bytes, SHA-256 `d3fd05fc56f414831114c6802d82bd6c25d1da87ca0032fd964c3fefe211c800`
- `candidate/training-receipt.json`, 20213 bytes, SHA-256 `0f1e1122679b629f20af80f2944c4b63762537d57788b31e68d7301c33bf0009`
- `reference/checkpoint-manifest.json`, 801 bytes, SHA-256 `e623bc3a98790f3e63caafd9de686573fe25f79032215f6b53c67b4bc2296705`
- `candidate/checkpoint-manifest.json`, 801 bytes, SHA-256 `ee8ee7633709c69fb400ca9b38cfcf6be8f67b0a4f26c17e89b27b0402a83ebd`
- `reference/staging-manifest.json`, 227890 bytes, SHA-256 `35d5b7a14cc7de472b00c0d99ec61d5e170c96d2d6f3e696b1b7cf3cbff00613`
- `candidate/staging-manifest.json`, 227890 bytes, SHA-256 `888e45e7d9555360287789411f1d8dc1cdc626d3c4c19aca764fbd735597b835`
- `reference/fit/results.csv`, 6042 bytes, SHA-256 `27c7ce87af4a4fb51ae40b892f4c9768949a9d4539621b106f2ea4f33796dcec`
- `candidate/fit/results.csv`, 6054 bytes, SHA-256 `2a4db4f5aa88d984a23f143d3dda856a7b988c0c2abbe343033a4cf3afce969c`
- `reference/train.log`, 229302 bytes, SHA-256 `0bcb2364359d6d9be753414cc806d37cf1823e18c69f401acb0dd1538ee19f1d`
- `candidate/train.log`, 229120 bytes, SHA-256 `7ff2c8704dc89bcf34cb62b1c37d9b3a9ee19c9a2b5f03e2ab7e77316db42e2a`
- `gpu-results-manifest.json`, 16340 bytes, SHA-256 `2b1e9c825d5d7d802dfca376eb70927fa3316dd711160539c9db0759606d4a83`
- `experiment-protocol.json`, 18220 bytes, SHA-256 `05a23cbef0fab6a2b5be1e1b7c8aafc4a261ab32f13cde26d13d68bbff3e3277`
- `training-input-delta.json`, 7465 bytes, SHA-256 `16e3ad8cddcafe81ba3302768c4142ae6c6dfd352f98ec02886f9dddc5f8a1e9`

This is the R34 reviewed-annotation comparison. Both roles independently started from `weights/cassette_lower_epoch173.pt` SHA-256 `e0aab1b377491700318e0b14cd4c558ec3c1684087a13b7dde2298ebdbf2847e` and recipe reviewed-incremental-fronts-seed46-v1 SHA-256 `f18e0a813a1057e1b3ab3a44ea3ce9b2a3e7673cc2bf04ca09b394bfdd8c51b8`. The reference uses accepted R33 training labels `data/candidates/front-review-2026-11-25-r33/manifest.json` SHA-256 `a268f6a18237f290886325a51cacd42e25f846ad1dbf50e92b5445a8a184439a`. The candidate uses the R34 overlay `data/candidates/front-review-2026-11-25-r34/manifest.json` SHA-256 `e1af61c80d86a15dce445c546d7c4b2b7b7f8c235e6ca03e9e45eea530b9318d`, which applies 7 eligible training-row corrections across 6 photographs (new_000003 row 61, new_000006 row 53, new_000013 row 36, new_000034 row 43, new_000040 rows 34 and 35, and new_000066 row 14). That delta is not box-only: six box edits, one right-corner edit, and one audited left-corner visibility restoration on new_000066 row 14. It is recorded in `training-input-delta.json` SHA-256 `16e3ad8cddcafe81ba3302768c4142ae6c6dfd352f98ec02886f9dddc5f8a1e9`. Seed 46, batch 4, imgsz 1280, FP32 (amp false), AdamW lr0=0.0001, 4110 presentations. Both roles see each of 137 train images 1x/epoch. Both arms use the unmodified native Pose26 loss; partial-box factor 2 remains discontinued. Held photograph new_000015 and the 31 old-series photographs keep prior supervision. Development-label amendments do not enter training or trainer validation. Both roles share the same recorded NVIDIA L4 runtime fingerprint and the same training budget.

The archive was reassembled by concatenating raw `part00` then `part01` after unzipping the stored chunk zips. `sha256sum -c` passed for the archive, the chunk zips, and the raw parts. `unzip -t` and `ZipFile.testzip()` reported no errors. The committed ZIP and chunk blobs are the raw bytes, not Git LFS pointers. The uploads directory omits the full ZIP; reconstruct it from the two stored parts before hashing.

This records the returned archive. It makes no accuracy or performance claim. PR35 remains selected. Frozen 666 is the primary evaluation reference and 663 is historical/diagnostic only. Exploratory 661 and trainer 664 must not replace 666/663. automatic_promotion is false. This r34 pack records a native-loss comparison of accepted R33 training labels against the accepted R34 training-annotation delta. It does not establish new-session reliability, and it is not an automatic model promotion.
