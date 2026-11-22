# Front-review-r20 seed-46 L4 results

Provenance for the returned archive:

- Source revision: `7207bcf4728140e03ef3fe3e6bce6918692d0a9e`
- Notebook pin commit: `bf13f8c293c316b62ede247324ef1e4e45c979a7`; notebook `notebooks/baomai_front_review_r20.ipynb` SHA-256 `7bea79f1f115a57764d7eaa0f8f13be876debcdf853d4f947f54fce5f8700c3f`
- Protocol: `experiments/front-review-2026-11-22-r20.json` SHA-256 `caa524e65dc4fc44cee520e150f7a3b60f5dddabfd31a7f4a2142a137ff0482f`
- Runtime: Colab NVIDIA L4
- Roles: reference and candidate
- Both roles are complete at 30/30 epochs, exit code 0, final-epoch checkpoints
- Archive: `baomai-front-review-r20-results.zip`, 39782532 bytes, SHA-256 `d9647d3a84904d4a5de5b1b6f826ff50366f462c6cb61f7f67448d58e71c38ef`

The archive was reassembled by concatenating raw `part00` then `part01` after unzipping the stored chunk zips. `sha256sum -c` passed for the archive, the chunk zips, and the raw parts. `unzip -t` and `ZipFile.testzip()` reported no errors. The committed ZIP and chunk blobs are the raw bytes, not Git LFS pointers.

This records the returned archive. It makes no accuracy or performance claim. PR35 remains selected. Frozen 666 is the primary evaluation reference and 663 is historical only. Exploratory 662 and trainer 664 must not replace 666/663.
