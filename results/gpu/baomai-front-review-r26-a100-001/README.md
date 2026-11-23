# Front-review-r26 seed-46 A100 results

Provenance for the returned archive:

- Source revision: `5c0a378dccfa16392eb3951531312efa3378889b`
- Notebook pin commit: `cb035d381568ded57917d274385e0fd560c6d6f6`; notebook `notebooks/baomai_front_review_r26.ipynb` SHA-256 `37fe6bb66ce1c4d833b228466b3bce24769c42fa1b383d1384f465ca49561d4d`
- Protocol: `experiments/front-review-2026-11-23-r26.json` SHA-256 `232f81cdf6d367fc9ece902e01b4ab5a03da5ae0898f5f75abae25ef5001372f`
- Runtime: Colab NVIDIA A100-SXM4-40GB
- Roles: reference and candidate
- Both roles are complete at 30/30 epochs, 1050 batches, 78 optimizer steps, exit code 0, final-epoch checkpoints
- Archive: `baomai-front-review-r26-results.zip`, 39784715 bytes, SHA-256 `105e86cc793378f1a47dd7cebc49ef84170d4f00ca9bec8c1e5f4e467df4b69d`

The archive was reassembled by concatenating raw `part00` then `part01` after unzipping the stored chunk zips. `sha256sum -c` passed for the archive, the chunk zips, and the raw parts. `unzip -t` and `ZipFile.testzip()` reported no errors. The committed ZIP and chunk blobs are the raw bytes, not Git LFS pointers.

This records the returned archive. It makes no accuracy or performance claim. PR35 remains selected. Frozen 666 is the primary evaluation reference and 663 is historical only. Exploratory 661 and trainer 664 must not replace 666/663. This r26 comparison tests accumulated eligible supervision, not an isolated 3-row test.
