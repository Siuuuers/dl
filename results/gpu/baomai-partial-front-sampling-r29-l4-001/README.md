# Partial-front-sampling-r29 seed-46 L4 results

Provenance for the returned archive:

- Source revision: `eea8e52a983f162512c4aa34ced73761b089609b`
- Notebook pin commit: `e48e356e5fd0ca90027d0d5b5aca610d1ffc61f2`; notebook `notebooks/baomai_partial_front_sampling_r29.ipynb` SHA-256 `5e3d8371d2f2b364ac0eb9bf41ed43748b49408b7afb8741cda5c1d677f070df`
- Protocol: `experiments/partial-front-sampling-2026-11-24-r29.json` SHA-256 `2ab005b9d64bdd388a71a1026df4885152762be1f992cd90607d5256a046468f`
- Runtime: Colab NVIDIA L4
- Roles: reference (control) and candidate
- Both roles are complete at 30/30 epochs, 1050 batches, 78 optimizer steps, exit code 0, final-epoch checkpoints
- Archive: `baomai-partial-front-sampling-r29-results.zip`, 39855160 bytes, SHA-256 `62d1932ab4681ab239709e80ceb4c84d4339e87ac74f7da5554203c53394edbd`

Both roles independently started from `weights/cassette_lower_epoch173.pt` SHA-256 `e0aab1b377491700318e0b14cd4c558ec3c1684087a13b7dde2298ebdbf2847e`, recipe reviewed-incremental-fronts-seed46-v1 SHA-256 `f18e0a813a1057e1b3ab3a44ea3ce9b2a3e7673cc2bf04ca09b394bfdd8c51b8`, accepted R28 labels `data/candidates/front-review-2026-11-23-r28/manifest.json` SHA-256 `b23e3273aca28e521b5c1fef8025b0888e736aa3f47229360f377d06a21abb9b`, seed 46, batch 4, imgsz 1280, FP32 (amp false), AdamW lr0=0.0001, 4110 presentations. Control sees each of 137 train images 1x/epoch. Candidate upsamples 9 partial-rich photos to 2x with 9 rotating omissions from 83 donors (45 protected at 1x). This is a sampling-only intervention on the accepted R28 labels; both roles share the same recorded NVIDIA L4 runtime fingerprint.

The archive was reassembled by concatenating raw `part00` then `part01` after unzipping the stored chunk zips. `sha256sum -c` passed for the archive, the chunk zips, and the raw parts. `unzip -t` and `ZipFile.testzip()` reported no errors. The committed ZIP and chunk blobs are the raw bytes, not Git LFS pointers.

This records the returned archive. It makes no accuracy or performance claim. PR35 remains selected. Frozen 666 is the primary evaluation reference and 663 is historical only. Exploratory 661 and trainer 664 must not replace 666/663. This r29 comparison tests a sampling-only change on accepted R28 labels. It does not establish new-session reliability, and it is not an automatic model promotion.
