# Partial-box-replication-r32 seed-47 L4 results

Provenance for the returned archive:

- Experiment: `partial-box-replication-r32`
- Source revision: `3f4c67908f7d59754fc14c520416aef8576eae5a` (unpublished local assembly)
- Base / published parent: `337a12d8b11f9925b493cba794f29a4ffcf848ed`. This results pack does not merge, close, or edit source PR 112 or earlier PRs 111, 110, or others.
- Notebook: `baomai_partial_box_replication_r32.ipynb`, 1126002 bytes, SHA-256 `162c2c7c343761071dba734ab4609afc0a2058fd92d1e6b4673fe890f03e7fb3`; Colab https://colab.research.google.com/drive/1YtZ1t6Ujr9fzUTYjHsyyM5HKazsFvO6m?authuser=1
- Protocol: `experiments/partial-box-replication-2026-11-25-r32.json` SHA-256 `68874ae3d4f6ce9d41131a98e887bcac0bd41aeed96f2fc409294530596880e0`
- Runtime: Colab NVIDIA L4 High-RAM, authuser=1
- Roles: reference (control) and candidate
- Both roles are complete at 30/30 epochs, 1050 batches, 78 optimizer steps, 4110 presentations, exit code 0, final-epoch checkpoints
- Archive: `baomai-partial-box-replication-r32-results.zip`, 40622524 bytes, SHA-256 `7c964db38b8b9898ae8a5920651c2213c92bf7bf043eabb861aa81a28b500f95`
- Drive folder: `/content/drive/MyDrive/baomai-partial-box-replication-r32-20261124`
- `gpu-results-manifest.json` created_at_utc `2026-11-24T19:12:35.761700+00:00` (3:12:35 AM HKT / Asia/Hong_Kong). The manifest and zip were written about 3:12 AM HKT on 25 Nov 2026.

Control finished about 3:03 AM HKT on 25 Nov 2026. Recorded `control/training-receipt.json` created_at_utc `2026-11-24T19:03:12.674431+00:00` (3:03:12 AM HKT / Asia/Hong_Kong); worker_wall_seconds `479.3429944200001`; execution wall_seconds `486.307247038`; execution sampled_peak_process_tree_rss_mib `2986.22265625`.

Candidate finished about 3:11 AM HKT on 25 Nov 2026. Recorded `candidate/training-receipt.json` created_at_utc `2026-11-24T19:11:53.731470+00:00` (3:11:53 AM HKT / Asia/Hong_Kong); worker_wall_seconds `476.4238801409999`; execution wall_seconds `483.4203960269999`; execution sampled_peak_process_tree_rss_mib `2964.75`.

Independently measured archive members (SHA-256 and size). These values were hashed from the extracted zip members and match the hashes recorded in the receipts, checkpoint manifests, and `gpu-results-manifest.json`:

- `control/fit/weights/last.pt`, 21704814 bytes, SHA-256 `ad2299fef36aff5e792dd215b28a1dad5b02693de2158588dd21eebd63fb81a3`
- `candidate/fit/weights/last.pt`, 21704814 bytes, SHA-256 `f761a010f999fb5f25a283bf73f946cb3a523ec08314fd6398009fd63a4777e3`
- `control/training-receipt.json`, 2358898 bytes, SHA-256 `8585a9b251a19763811c2e588c7750357d642aadff9766034d40728cacb274b0`
- `candidate/training-receipt.json`, 2359021 bytes, SHA-256 `57f8e3519907a3d267c9ef548a998aa56edc628328712fe0ab3ab1273f4c9957`
- `control/checkpoint-manifest.json`, 801 bytes, SHA-256 `2298f176be333cbb69a05a8cdc2927e68d55e13a10d6b0969a6f1e3960ba721e`
- `candidate/checkpoint-manifest.json`, 801 bytes, SHA-256 `f598db2df6de7ad2defeb26f68d938dbddfcade5dcf311f4263d48ba035f5543`
- `control/staging-manifest.json`, 220052 bytes, SHA-256 `dbd84ace66479cf335b08be738f55ec7ffc49d4f3748b1fef1475acffdd45fb9`
- `candidate/staging-manifest.json`, 220054 bytes, SHA-256 `734ffbbf0fb647c282939c3c9caf545a78df08e906f1c4037f5dd7c082e7fab1`
- `control/fit/results.csv`, 6038 bytes, SHA-256 `c9122d02356f29b865435c7e2a1da51f816cae8613910e6328507f753933c2c8`
- `candidate/fit/results.csv`, 6043 bytes, SHA-256 `ac10a5860e8ba93780c63953bb46e147facdcd33b1788e3ce7ea30192406204e`
- `control/train.log`, 229545 bytes, SHA-256 `a9f2c8a668e6d6a6473b5348aa202cb8aec01f38d32eb36343d41c8358774af0`
- `candidate/train.log`, 229404 bytes, SHA-256 `f9fb37314ae7db037b53dd1f025de369f64044d54742ebf777ade8bdf6b2919c`
- `gpu-results-manifest.json`, 5205837 bytes, SHA-256 `2d56333df9b9e12645bb33a523151ce8ceb1f03a1e86b5231dc81af5860469de`
- `experiment-protocol.json`, 16683 bytes, SHA-256 `68874ae3d4f6ce9d41131a98e887bcac0bd41aeed96f2fc409294530596880e0`

This is the R32 seed-47 replication of the PR110 / R31 comparison. Both roles independently started from `weights/cassette_lower_epoch173.pt` SHA-256 `e0aab1b377491700318e0b14cd4c558ec3c1684087a13b7dde2298ebdbf2847e`, recipe reviewed-incremental-fronts-seed47-v1 SHA-256 `f65470a7e7f1ff066480cf6fd320fb28883fef2367d92846c2a1a3ca60cecd67`, and the same existing R31 training labels `data/candidates/front-review-2026-11-24-r31/manifest.json` SHA-256 `8fb270a31131bf1e20c9bf5aeb05471cdcc29c03860da76e93614eeb7183fdb7`. Seed 47, batch 4, imgsz 1280, FP32 (amp false), AdamW lr0=0.0001, 4110 presentations. Both roles see each of 137 train images 1x/epoch. Control uses native box-regression loss factor 1. Candidate multiplies native CIoU and normalized-L1 box-regression contributions by 2 only for foreground anchors already assigned to targets with fewer than two locatable (nonzero-visibility) corners in the augmented batch. Classification, keypoint, RLE and native branch scheduling remain at their original objectives. The R31 loss implementation is reused unchanged. R32's 23 development-row replacements are val-only and did not enter training (training_labels_changed 0). Both roles share the same recorded NVIDIA L4 runtime fingerprint and the same staged image and label hashes.

The archive was reassembled by concatenating raw `part00` then `part01` after unzipping the stored chunk zips. `sha256sum -c` passed for the archive, the chunk zips, and the raw parts. `unzip -t` and `ZipFile.testzip()` reported no errors. The committed ZIP and chunk blobs are the raw bytes, not Git LFS pointers.

This records the returned archive. It makes no accuracy or performance claim. PR35 remains selected. Frozen 666 is the primary evaluation reference and 663 is historical/diagnostic only. Exploratory 661 and trainer 664 must not replace 666/663. automatic_promotion is false. Report seed 46 and seed 47 independently and together; do not choose whichever seed looks better. This r32 pack records a second-seed replication of the same partial-box weighting treatment on the same R31 training labels. It does not establish new-session reliability, and it is not an automatic model promotion.
