# Front-review-r15 seed-46 L4 results

Provenance for the returned archive:

- Source revision: `fe7ae1095d24ebb6afea2c57194c88b369e86921`
- Notebook pin commit: `833a1ae37f9840b105b73f2e7f36952cb3946f7a`; notebook `notebooks/baomai_front_review_r15.ipynb` SHA-256 `689eab3287ac5df73078857632d1f6157ec27583c4ed0d854feac9b321860807`
- Protocol: `experiments/front-review-2026-11-21-r15.json` SHA-256 `3b3a7f2373fb3297ffcab4d193a4e9bb07fe8970508d228c602dc78a0fb3056b`
- Runtime: Colab NVIDIA L4
- Roles: reference and candidate
- Both roles are complete at 30/30 epochs, exit code 0, final-epoch checkpoints
- Archive: `baomai-front-review-r15-results.zip`, 39781855 bytes, SHA-256 `6a323797cc05eac71a61b5a8d328dfad1c242b4506cbac51eb7e91f84971d22a`

The archive was reassembled by concatenating raw `part00` then `part01` after unzipping the stored chunk zips. `sha256sum -c` passed for the archive, the chunk zips, and the raw parts. `unzip -t` and `ZipFile.testzip()` reported no errors. The committed ZIP and chunk blobs are the raw bytes, not Git LFS pointers.

This records the returned archive. It makes no accuracy or performance claim. PR35 remains selected. Frozen 666 is the primary evaluation reference and 663 is historical only. This exploratory r15 review does not replace trainer 664 val. A comparison of PR35 versus eligible r15 does not isolate the r14-to-r15 delta.
