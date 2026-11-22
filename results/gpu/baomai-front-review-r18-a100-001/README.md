# Front-review-r18 seed-46 A100 results

Provenance for the returned archive:

- Source revision: `b6dbcc75efd5994bfdb3de456862608b2cc3aa40`
- Notebook pin commit: `850fd7e36c2e283b798134056b0ecd6f3f534506`; notebook `notebooks/baomai_front_review_r18.ipynb` SHA-256 `eb3a5f852e8d6db143197d4a808cf3e9c68b05725f6d224032f7a0abde9f0125`
- Protocol: `experiments/front-review-2026-11-22-r18.json` SHA-256 `788715d37250483c367513490e20945871cb77628437ed2ca27c72ea96587137`
- Runtime: Colab NVIDIA A100-SXM4-40GB
- Roles: reference and candidate
- Both roles are complete at 30/30 epochs, exit code 0, final-epoch checkpoints
- Archive: `baomai-front-review-r18-results.zip`, 39783562 bytes, SHA-256 `167782e60b0a24d5ba041222a669aff2cb782ef684b3f64c7106cf9bcbb18445`

The archive was reassembled by concatenating raw `part00` then `part01` after unzipping the stored chunk zips. `sha256sum -c` passed for the archive, the chunk zips, and the raw parts. `unzip -t` and `ZipFile.testzip()` reported no errors. The committed ZIP and chunk blobs are the raw bytes, not Git LFS pointers.

This records the returned archive. It makes no accuracy or performance claim. PR35 remains selected. Frozen 666 is the primary evaluation reference and 663 is historical only. Exploratory 662 and trainer 664 must not replace 666/663.
