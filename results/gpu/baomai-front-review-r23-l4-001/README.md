# Front-review-r23 seed-46 L4 results

Provenance for the returned archive:

- Source revision: `ffe4d4ecb6699180e7b25ee10fc9eb31647fec38`
- Notebook pin commit: `ee2ca5f260af96621aed4bf5803680b9b6403349`; notebook `notebooks/baomai_front_review_r23.ipynb` SHA-256 `c49f188c9d74c4413d6fed92e9ba878a8d4ad769fe553281a665b5941788bb36`
- Protocol: `experiments/front-review-2026-11-22-r23.json` SHA-256 `a4183607c6737a8466aef4edd625ca52450a486dd030456e4d10e3cd98736a54`
- Runtime: Colab NVIDIA L4
- Roles: reference and candidate
- Both roles are complete at 30/30 epochs, exit code 0, final-epoch checkpoints
- Archive: `baomai-front-review-r23-results.zip`, 39783521 bytes, SHA-256 `3f12115b2c15dec87a7e7404115ff933c54fadbea7eef415a098533933059be9`

The archive was reassembled by concatenating raw `part00` then `part01` after unzipping the stored chunk zips. `sha256sum -c` passed for the archive, the chunk zips, and the raw parts. `unzip -t` and `ZipFile.testzip()` reported no errors. The committed ZIP and chunk blobs are the raw bytes, not Git LFS pointers.

This records the returned archive. It makes no accuracy or performance claim. PR35 remains selected. Frozen 666 is the primary evaluation reference and 663 is historical only. Exploratory 662 and trainer 664 must not replace 666/663. This r23 comparison tests accumulated eligible supervision, not the isolated effect of the four reviewed right-point edits.
