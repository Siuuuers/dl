# Front-review-r33 native supervision L4 results

Provenance for the returned archive:

- Experiment: `front-review-r33`
- Source revision: `d4b905dc0ed7728815108fed04486bc4215f7a2b` (unpublished local assembly)
- Base / published parent: `337a12d8b11f9925b493cba794f29a4ffcf848ed`. This results pack does not merge, close, or edit earlier PRs.
- Notebook: `baomai_front_review_r33.ipynb`, 3124979 bytes, SHA-256 `f939e8564249b9c43cd5a2704e5f3168288fdcbfbd8b2f145ccc0f138882de0e`; Colab https://colab.research.google.com/drive/1_kSxEDxVZFkrxBbWb9oYGhEze6QChknV?authuser=1
- Protocol: `experiments/front-review-2026-11-25-r33.json` SHA-256 `9782658c99d56740af0302c4edd80a071e169ca404c31b01948c7a2092d078ed`
- Runtime: Colab NVIDIA L4 High-RAM, authuser=1. Both roles share the same recorded L4 runtime fingerprint.
- Roles: reference (previous R31 training labels) and candidate (R33 training-boundary corrections)
- Both roles are complete at 30/30 epochs, 1050 batches, 78 optimizer steps, 4110 presentations, exit code 0, final-epoch checkpoints, independently started from original epoch 173
- Archive: `baomai-front-review-r33-results.zip`, 39794452 bytes, SHA-256 `393709837711e2c3f454ee36190d76b3c11744e7d8137ce44ba5294c1b208c4b`. The zip contains 15 members: 14 files declared in `gpu-results-manifest.json` plus that manifest.
- Drive folder: `/content/drive/MyDrive/baomai-front-review-r33-20261125`
- `gpu-results-manifest.json` created_at_utc `2026-11-25T11:47:03.922041+00:00` (7:47:03 PM HKT / Asia/Hong_Kong). The manifest and zip were written about 7:47 PM HKT on 25 Nov 2026.

Reference finished about 7:37 PM HKT on 25 Nov 2026. Recorded `reference/training-receipt.json` created_at_utc `2026-11-25T11:37:51.291733+00:00` (7:37:51 PM HKT / Asia/Hong_Kong); worker_wall_seconds `450.36262064000016`; execution wall_seconds `453.5981143890001`; execution sampled_peak_process_tree_rss_mib `2983.23828125`.

Candidate finished about 7:46 PM HKT on 25 Nov 2026. Recorded `candidate/training-receipt.json` created_at_utc `2026-11-25T11:46:03.218464+00:00` (7:46:03 PM HKT / Asia/Hong_Kong); worker_wall_seconds `448.6338659889998`; execution wall_seconds `451.84946472499996`; execution sampled_peak_process_tree_rss_mib `2934.87109375`.

Independently measured archive members (SHA-256 and size). These values were hashed from the extracted zip members and match the hashes recorded in the receipts, checkpoint manifests, and `gpu-results-manifest.json`:

- `reference/fit/weights/last.pt`, 21704750 bytes, SHA-256 `b6221ea8cd8d735b77e46863e95c4e8bbe926c09bc01bbed9db0a904ae2b9045`
- `candidate/fit/weights/last.pt`, 21704750 bytes, SHA-256 `d4d92fa6466414ae12c243a5d7db659898de1fdd8ebd9176031913e748b79a15`
- `reference/training-receipt.json`, 19764 bytes, SHA-256 `a014267cb437bb97ef00c8c7a66dffd9d12ee798200b06531ce92aba18cd7a11`
- `candidate/training-receipt.json`, 19766 bytes, SHA-256 `91878b4d98b70755c0ef9403b13d258aff6588533057fc3b5b38235283de1791`
- `reference/checkpoint-manifest.json`, 801 bytes, SHA-256 `f1e9145a113725730f55859c97d9fcbeb43d92308900be25dbb4d0eb4f563256`
- `candidate/checkpoint-manifest.json`, 801 bytes, SHA-256 `f4e0f5bf7ca02fc1f6bb1590ae888454c4d9a47b5f0c30c7cf871098d28a8800`
- `reference/staging-manifest.json`, 228103 bytes, SHA-256 `344ca3d5c88295fac0b0ffaa05ae4786ba4aff4fd76813b1bf3bee8a0868547b`
- `candidate/staging-manifest.json`, 228103 bytes, SHA-256 `b88a2e5d9f00c1330492d7ee3485cb90d001f97d95584d598e231053667ebe03`
- `reference/fit/results.csv`, 6049 bytes, SHA-256 `e4b2de060c5664da89a18818042efcfcfed98c8aefe78b8e9c4c74f857d690db`
- `candidate/fit/results.csv`, 6041 bytes, SHA-256 `812bb4c94013f83dabbb5bd31c4316c54a8da195b3b2ac0ebcf76400ad188171`
- `reference/train.log`, 229079 bytes, SHA-256 `9a637732bf2cefa83409351277200d3ebc98b18136dcb8d5dc48927a89cb65cf`
- `candidate/train.log`, 229089 bytes, SHA-256 `5fdde30a80ae59002513e52687761346693b8122602751a3dc0d02c229630342`
- `gpu-results-manifest.json`, 15885 bytes, SHA-256 `b63d93cab9da8be7d7c9a93a948eb7eb727c0d8b24a65f7800d9ecf446e4cf1d`
- `experiment-protocol.json`, 16789 bytes, SHA-256 `9782658c99d56740af0302c4edd80a071e169ca404c31b01948c7a2092d078ed`
- `training-input-delta.json`, 7844 bytes, SHA-256 `b6c755998f42a2188a66ff013a2ede4d9c184f923af81fbb7995f2ce809b8f64`

This is the R33 native-supervision comparison. Both roles independently started from `weights/cassette_lower_epoch173.pt` SHA-256 `e0aab1b377491700318e0b14cd4c558ec3c1684087a13b7dde2298ebdbf2847e` and recipe reviewed-incremental-fronts-seed46-v1 SHA-256 `f18e0a813a1057e1b3ab3a44ea3ce9b2a3e7673cc2bf04ca09b394bfdd8c51b8`. The reference uses previous R31 training labels `data/candidates/front-review-2026-11-24-r31/manifest.json` SHA-256 `8fb270a31131bf1e20c9bf5aeb05471cdcc29c03860da76e93614eeb7183fdb7`. The candidate uses the R33 overlay `data/candidates/front-review-2026-11-25-r33/manifest.json` SHA-256 `a268f6a18237f290886325a51cacd42e25f846ad1dbf50e92b5445a8a184439a`, which applies 8 box-only training-row corrections across 7 photographs (new_000003, new_000006, new_000013, new_000017, new_000034, new_000040 rows 34 and 35, and new_000066). That delta is recorded in `training-input-delta.json` SHA-256 `b6c755998f42a2188a66ff013a2ede4d9c184f923af81fbb7995f2ce809b8f64`. Seed 46, batch 4, imgsz 1280, FP32 (amp false), AdamW lr0=0.0001, 4110 presentations. Both roles see each of 137 train images 1x/epoch. Both arms use the unmodified native Pose26 loss; partial-box factor 2 is discontinued. Held photograph new_000015 and the 31 old-series photographs keep prior supervision. Development-label amendments do not enter training or trainer validation. Both roles share the same recorded NVIDIA L4 runtime fingerprint and the same training budget.

The archive was reassembled by concatenating raw `part00` then `part01` after unzipping the stored chunk zips. `sha256sum -c` passed for the archive, the chunk zips, and the raw parts. `unzip -t` and `ZipFile.testzip()` reported no errors. The committed ZIP and chunk blobs are the raw bytes, not Git LFS pointers.

This records the returned archive. It makes no accuracy or performance claim. PR35 remains selected. Frozen 666 is the primary evaluation reference and 663 is historical/diagnostic only. Exploratory 661 and trainer 664 must not replace 666/663. automatic_promotion is false. This r33 pack records a native-loss comparison of previous R31 training labels against the accepted R33 training-boundary delta. It does not establish new-session reliability, and it is not an automatic model promotion.
