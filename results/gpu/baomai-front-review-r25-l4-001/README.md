# Front-review-r25 seed-46 L4 results

Provenance for the returned archive:

- Source revision: `86a7ba078b348f2b1cbacf73b4c271e9216fdb63`
- Notebook pin commit: `ace99d4aa71e5c20d7c5d2445f18fdab58d63496`; notebook `notebooks/baomai_front_review_r25.ipynb` SHA-256 `11d0cac7a492f7234e4778490c1ef70d445d5aa309f2bca11af4d02329553877`
- Protocol: `experiments/front-review-2026-11-23-r25.json` SHA-256 `80c3f2849890a664db9fea857342c4d78d38ba736b7a108ce13905d74aedb287`
- Runtime: Colab NVIDIA L4
- Roles: reference and candidate
- Both roles are complete at 30/30 epochs, exit code 0, final-epoch checkpoints
- Archive: `baomai-front-review-r25-results.zip`, 39783961 bytes, SHA-256 `8cb47ae13ad14e325b540c19ccd6fccb3198b85406f3238a581bd2db5b9e3762`

The archive was reassembled by concatenating raw `part00` then `part01` after unzipping the stored chunk zips. `sha256sum -c` passed for the archive, the chunk zips, and the raw parts. `unzip -t` and `ZipFile.testzip()` reported no errors. The committed ZIP and chunk blobs are the raw bytes, not Git LFS pointers.

This records the returned archive. It makes no accuracy or performance claim. PR35 remains selected. Frozen 666 is the primary evaluation reference and 663 is historical only. Exploratory 662 and trainer 664 must not replace 666/663. Photos 008 and 010 are development validation only; trainer validation remains unchanged. This r25 comparison tests accumulated eligible supervision, not the isolated effect of the eight left-point repairs or the four training-point edits in photograph 012. Keep r25 exploratory scores separate from earlier r24 exploratory scores.
