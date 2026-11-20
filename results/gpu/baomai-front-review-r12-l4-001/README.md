# Front-review-r12 seed-46 L4 results

Provenance for the returned archive:

- Source revision: `4f4423a511b42bc804cbcbeb7a772823cef2eb3b`
- Notebook pin commit: `eb8284dcab71074275d9d22fc0fd974839d66863`; notebook `notebooks/baomai_front_review_r12.ipynb` SHA-256 `a5e11fed8ee962ef569747e29ba320b87d97311b37735dff61b4262c854f94a9`
- Protocol: `experiments/front-review-2026-11-20-r12.json` SHA-256 `6c23b00d5251b9cd98630a796a328dc9cc5574296ccb47bac83cbb1ffa3e7f0a`
- Runtime: Colab NVIDIA L4
- Roles: reference and candidate
- Both roles are complete at 30/30 epochs, exit code 0, final-epoch checkpoints
- Archive: `baomai-front-review-r12-results.zip`, 39781223 bytes, SHA-256 `24b15853253ba20762939ae16e236a1ee99736df58f5ca96a06a57c5f996e572`

The archive was reassembled by concatenating raw `part00` then `part01` after unzipping the stored chunk zips. `sha256sum -c` passed for the archive, the chunk zips, and the raw parts. `unzip -t` and `ZipFile.testzip()` reported no errors. The committed ZIP and chunk blobs are the raw bytes, not Git LFS pointers.

This records the returned archive. It makes no accuracy or performance claim. PR35 remains selected.
