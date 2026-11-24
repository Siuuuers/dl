# Partial-target-loss-r30 seed-46 L4 results

Provenance for the returned archive:

- Source revision: `89c034a74d74b02abc9dd6ffa775897889235347`
- Notebook pin commit: `e48e356e5fd0ca90027d0d5b5aca610d1ffc61f2`; notebook `notebooks/baomai_partial_target_loss_r30.ipynb` SHA-256 `152dac24a9f4d9b46d62bd7fb8766352c1c9b2a4cc380b06d158b6788d15cfc9`
- Protocol: `experiments/partial-target-loss-2026-11-24-r30.json` SHA-256 `2324cd24fbd50acfbce2545d4ae374a83fe3ede978dac3c9414725f363a8a9b0`
- Runtime: Colab NVIDIA L4
- Roles: reference (control) and candidate
- Both roles are complete at 30/30 epochs, 1050 batches, 78 optimizer steps, exit code 0, final-epoch checkpoints
- Archive: `baomai-partial-target-loss-r30-results.zip`, 40615746 bytes, SHA-256 `450fe2b66adc779fe0f0c77302b5efcb8786de41e0a498182e1e96d96f144c47`

Both roles independently started from `weights/cassette_lower_epoch173.pt` SHA-256 `e0aab1b377491700318e0b14cd4c558ec3c1684087a13b7dde2298ebdbf2847e`, recipe reviewed-incremental-fronts-seed46-v1 SHA-256 `f18e0a813a1057e1b3ab3a44ea3ce9b2a3e7673cc2bf04ca09b394bfdd8c51b8`, shared R30 labels `data/candidates/front-review-2026-11-24-r30/manifest.json` SHA-256 `b74562c7d8f7a66deeb08fdafb3ed16dd6ac00aac7122b43719fa5fa09dd1a80`, seed 46, batch 4, imgsz 1280, FP32 (amp false), AdamW lr0=0.0001, 4110 presentations. Both roles see each of 137 train images 1x/epoch. Control uses loss factor 1. Candidate multiplies detection-loss contributions by 2 only for foreground anchors already assigned to targets with fewer than two locatable corners in the augmented batch. This is a loss-weighting intervention on shared R30 labels; both roles share the same recorded NVIDIA L4 runtime fingerprint.

The archive was reassembled by concatenating raw `part00` then `part01` after unzipping the stored chunk zips. `sha256sum -c` passed for the archive, the chunk zips, and the raw parts. `unzip -t` and `ZipFile.testzip()` reported no errors. The committed ZIP and chunk blobs are the raw bytes, not Git LFS pointers.

This records the returned archive. It makes no accuracy or performance claim. PR35 remains selected. Frozen 666 is the primary evaluation reference and 663 is historical only. Exploratory 661 and trainer 664 must not replace 666/663. This r30 comparison tests a partial-target loss-weighting change on shared R30 labels. It does not establish new-session reliability, and it is not an automatic model promotion.
