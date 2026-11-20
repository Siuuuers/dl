# Front-review-r13 seed-46 L4 results

Provenance for the returned archive:

- Source revision: `5c2817f1c1eaa1bbce7060facb82732c3d249e10`
- Notebook pin commit: `4795910cdeb37d23d4b99031eb480c94b883b478`; notebook `notebooks/baomai_front_review_r13.ipynb` SHA-256 `de0146574fff097da7edf45bbd8fdb591569add7dff513fdb63df5b970fbea96`
- Protocol: `experiments/front-review-2026-11-21-r13.json` SHA-256 `b77ab22c5655146b05a870458fcad080f4e6e9dcc890ae248564b86147a9f22d`
- Runtime: Colab NVIDIA L4
- Roles: reference and candidate
- Both roles are complete at 30/30 epochs, exit code 0, final-epoch checkpoints
- Archive: `baomai-front-review-r13-results.zip`, 39781828 bytes, SHA-256 `6143530679a6c9a972119852b9f4922d8e7143bb01cfbd6edc52d0ecd8e70f4c`

The archive was reassembled by concatenating raw `part00` then `part01` after unzipping the stored chunk zips. `sha256sum -c` passed for the archive, the chunk zips, and the raw parts. `unzip -t` and `ZipFile.testzip()` reported no errors. The committed ZIP and chunk blobs are the raw bytes, not Git LFS pointers.

This records the returned archive. It makes no accuracy or performance claim. PR35 remains selected.
