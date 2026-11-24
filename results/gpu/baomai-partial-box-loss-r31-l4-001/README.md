# Partial-box-loss-r31 seed-46 L4 results

Provenance for the returned archive:

- Experiment: `partial-box-loss-r31`
- Source revision: `0a8ff31cadaf0ee0691de73fd103bba5de82b038` (local assembly, unpublished)
- Base revision: `337a12d8b11f9925b493cba794f29a4ffcf848ed` is the head of open unmerged source PR 108. This results pack does not merge or close PR 108.
- Notebook: `notebooks/baomai_partial_box_loss_r31.ipynb`, 728072 bytes, SHA-256 `3487253855abe19b4199e420dd7e9717caa7ada4198f8dfd256595d8809ca047`; Colab https://colab.research.google.com/drive/1qXggyve0mlu6-1yR0FiAUr40czXFgqZI?authuser=1
- Protocol: `experiments/partial-box-loss-2026-11-24-r31.json` SHA-256 `a24172f618c819c5d2a32d66b9b5c13926ac025e4090c4d582b2622194cd9b64`
- Runtime: Colab NVIDIA L4 High-RAM
- Roles: reference (control) and candidate
- Both roles are complete at 30/30 epochs, 1050 batches, 78 optimizer steps, 4110 presentations, exit code 0, final-epoch checkpoints
- Archive: `baomai-partial-box-loss-r31-results.zip`, 40622971 bytes, SHA-256 `169733506e66f06438d1ead350f52d6e167ad96a4b0a2b1fe066ac02dbc330cf`
- Drive folder: `/content/drive/MyDrive/baomai-partial-box-loss-r31-20261124`
- `gpu-results-manifest.json` created_at_utc `2026-11-24T14:22:43.949649+00:00` (10:22:43 PM HKT / Asia/Hong_Kong). The manifest and zip were written about 10:22 PM HKT on 24 Nov 2026.

Control finished about 10:14 PM HKT on 24 Nov 2026. Recorded `control/training-receipt.json` created_at_utc `2026-11-24T14:13:41.662454+00:00` (10:13:41 PM HKT / Asia/Hong_Kong); worker_wall_seconds `470.02774204800016`; execution wall_seconds `473.6436162059997`; execution sampled_peak_process_tree_rss_mib `2975.3046875`.

Candidate finished about 10:22 PM HKT on 24 Nov 2026. Recorded `candidate/training-receipt.json` created_at_utc `2026-11-24T14:22:05.648646+00:00` (10:22:05 PM HKT / Asia/Hong_Kong); worker_wall_seconds `466.354450583`; execution wall_seconds `470.0246835349999`; execution sampled_peak_process_tree_rss_mib `2945.3515625`.

Independently measured archive members (SHA-256 and size). These values were hashed from the extracted zip members and match the hashes recorded in the receipts, checkpoint manifests, and `gpu-results-manifest.json`:

- `control/fit/weights/last.pt`, 21704814 bytes, SHA-256 `cd5f2cf9648e72e4b6064eaafee19fb84c7ad846294be1c4bbcc3a863151f69b`
- `candidate/fit/weights/last.pt`, 21704814 bytes, SHA-256 `bcfe34f38fe1f6ca5dc566fda4f0b773787f53a6043b7bb35d4284f2ced66850`
- `control/training-receipt.json`, 2358691 bytes, SHA-256 `aba882ac4ef4b1b39aa9d301d25816a0c38ee8ec31b8dbf0ecd8f9ac51b41e8b`
- `candidate/training-receipt.json`, 2358706 bytes, SHA-256 `6039d5fa3c069933d571efe24a8abb434b46a3093e0ae9c06e2efeb2b544615d`
- `control/checkpoint-manifest.json`, 801 bytes, SHA-256 `51835b8610fcfdfaee839b7e141d748e82d98daaf406c5724e3c5a020cb05b7d`
- `candidate/checkpoint-manifest.json`, 801 bytes, SHA-256 `c703ef2671e1980b37304f1d8de0e40f5b3e3e30f30c1931487e1973d46f1315`
- `control/staging-manifest.json`, 220045 bytes, SHA-256 `940ee80a2430144a2b311dcab9fb9b701741a74ada193e4c22972cd1abe2a91e`
- `candidate/staging-manifest.json`, 220047 bytes, SHA-256 `5168bcb5b7c96fb408b779a09987cd01cbf16b0753ffce394155b5be0e5ee237`
- `control/fit/results.csv`, 6052 bytes, SHA-256 `39911aa17664ccc700dabbde1c1444acdc0febcb162e5a60b6dea05ae09ece75`
- `candidate/fit/results.csv`, 6048 bytes, SHA-256 `920b7a930c15ae3ab2d13eba69a21b777e73fd9c2898b155ac31a16a33e653c6`
- `control/train.log`, 228970 bytes, SHA-256 `a218e6ef7ffa21ccf081e78893aa83cc696d693b5df8ec3ffb11dd6649da720e`
- `candidate/train.log`, 229003 bytes, SHA-256 `d1b0c2b3e64ae58ce2a6473c324b25511ea8b51e419b719daff19f28ed2ab456`
- `gpu-results-manifest.json`, 5205914 bytes, SHA-256 `0743c40bd979a4507e37c4a3ef7adde4a8e8e2206269e9e3bebe8f41f0cda6cc`
- `experiment-protocol.json`, 15700 bytes, SHA-256 `a24172f618c819c5d2a32d66b9b5c13926ac025e4090c4d582b2622194cd9b64`

Both roles independently started from `weights/cassette_lower_epoch173.pt` SHA-256 `e0aab1b377491700318e0b14cd4c558ec3c1684087a13b7dde2298ebdbf2847e`, recipe reviewed-incremental-fronts-seed46-v1 SHA-256 `f18e0a813a1057e1b3ab3a44ea3ce9b2a3e7673cc2bf04ca09b394bfdd8c51b8`, shared R31 labels `data/candidates/front-review-2026-11-24-r31/manifest.json` SHA-256 `8fb270a31131bf1e20c9bf5aeb05471cdcc29c03860da76e93614eeb7183fdb7`, seed 46, batch 4, imgsz 1280, FP32 (amp false), AdamW lr0=0.0001, 4110 presentations. Both roles see each of 137 train images 1x/epoch. Control uses loss factor 1. Candidate multiplies native CIoU and normalized-L1 box-regression contributions by 2 only for foreground anchors already assigned to targets with fewer than two locatable (nonzero-visibility) corners in the augmented batch. Classification, keypoint, RLE and native branch scheduling remain at their original objectives. This is a box-regression loss-weighting intervention on shared R31 labels; both roles share the same recorded NVIDIA L4 runtime fingerprint.

The archive was reassembled by concatenating raw `part00` then `part01` after unzipping the stored chunk zips. `sha256sum -c` passed for the archive, the chunk zips, and the raw parts. `unzip -t` and `ZipFile.testzip()` reported no errors. The committed ZIP and chunk blobs are the raw bytes, not Git LFS pointers.

This records the returned archive. It makes no accuracy or performance claim. PR35 remains selected. Frozen 666 is the primary evaluation reference and 663 is historical/diagnostic only. Exploratory 661 and trainer 664 must not replace 666/663. This r31 comparison records a partial-target box-regression loss-weighting change on shared R31 labels. It does not establish new-session reliability, and it is not an automatic model promotion.
