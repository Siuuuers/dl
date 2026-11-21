# Front-review-r17 seed-46 L4 results

Provenance for the returned archive:

- Source revision: `08921b4ccd122798675293221145f2be41171db2`
- Notebook pin commit: `15869c7742e72ee1a11d0e5e2b668206f02c0348`; notebook `notebooks/baomai_front_review_r17.ipynb` SHA-256 `8a1bfc4f548dfb3e67a23f07904a925bac14b9accd4378acf2fca730ef4c0da2`
- Protocol: `experiments/front-review-2026-11-22-r17.json` SHA-256 `320d9e97805ddea968dd04b033037272c0fe41d77f1c149c2b05bc654f4ca12a`
- Runtime: Colab NVIDIA L4
- Roles: reference and candidate
- Both roles are complete at 30/30 epochs, exit code 0, final-epoch checkpoints
- Archive: `baomai-front-review-r17-results.zip`, 39782260 bytes, SHA-256 `339eec148cc5d56d9bd2351f2562f09658685dd9117073338c434af63ad4917f`

The archive was reassembled by concatenating raw `part00` then `part01` after unzipping the stored chunk zips. `sha256sum -c` passed for the archive, the chunk zips, and the raw parts. `unzip -t` and `ZipFile.testzip()` reported no errors. The committed ZIP and chunk blobs are the raw bytes, not Git LFS pointers.

This records the returned archive. It makes no accuracy or performance claim. PR35 remains selected. Frozen 666 is the primary evaluation reference and 663 is historical only. Exploratory 662 uses the r16 parent bytes and must not replace 666/663 or trainer 664 val. A comparison of PR35 versus eligible r17 does not isolate the r16-to-r17 delta. Photo 074 row 4 Y-anchor remains unresolved and recorded; it is a non-blocker and was not changed.
