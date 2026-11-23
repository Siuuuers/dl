# Front-review-r28 seed-46 L4 results

Provenance for the returned archive:

- Source revision: `173e5cecde02d17d0afdcf6632062ca5791f7bbd`
- Notebook pin commit: `c487e7abbb0aca19f59fb143ca2320d3adb01b3e`; notebook `notebooks/baomai_front_review_r28.ipynb` SHA-256 `504f494c94613ccf74b5c4231495b59a21e3fec456ebda7ccebbdb1fcc6aa027`
- Protocol: `experiments/front-review-2026-11-23-r28.json` SHA-256 `69550efb5249dced874575fd7ecd8ddae67cd27b1dd57d35d7475716f1ee9783`
- Runtime: Colab NVIDIA L4
- Roles: reference and candidate
- Both roles are complete at 30/30 epochs, 1050 batches, 78 optimizer steps, exit code 0, final-epoch checkpoints
- Archive: `baomai-front-review-r28-results.zip`, 39792660 bytes, SHA-256 `32cb80de2a6e797196ead979764808f669378f5c1f45e1777fe396274293ae6b`

The archive was reassembled by concatenating raw `part00` then `part01` after unzipping the stored chunk zips. `sha256sum -c` passed for the archive, the chunk zips, and the raw parts. `unzip -t` and `ZipFile.testzip()` reported no errors. The committed ZIP and chunk blobs are the raw bytes, not Git LFS pointers.

This records the returned archive. It makes no accuracy or performance claim. PR35 remains selected. Frozen 666 is the primary evaluation reference and 663 is historical only. Exploratory 661 and trainer 664 must not replace 666/663. This r28 comparison tests 10 coordinate-amended training rows across 9 photographs (fresh R26 supervision versus fresh eligible R28 supervision). It does not establish new-session reliability, and it is not an automatic model promotion.
