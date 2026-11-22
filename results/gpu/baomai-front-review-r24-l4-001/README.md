# Front-review-r24 seed-46 L4 results

Provenance for the returned archive:

- Source revision: `6307dae6a8db545a72446abb448f19a5dcd5d2b2`
- Notebook pin commit: `5feb60b8156c5aff6f9c5ebc3814cbe378ff5628`; notebook `notebooks/baomai_front_review_r24.ipynb` SHA-256 `36135c847fb460f7299628374bea907a01981e48d5336aa5321b1afda8fc8e2a`
- Protocol: `experiments/front-review-2026-11-23-r24.json` SHA-256 `0a6322dbf4ee0cfca60b06478f8feecbd4dce516df92301cc74a2030b73021f5`
- Runtime: Colab NVIDIA L4
- Roles: reference and candidate
- Both roles are complete at 30/30 epochs, exit code 0, final-epoch checkpoints
- Archive: `baomai-front-review-r24-results.zip`, 39783117 bytes, SHA-256 `4b2505711bce656f4b693a5ccff93bd23cbda370a5d5ec6933c86e48c0e4e719`

The archive was reassembled by concatenating raw `part00` then `part01` after unzipping the stored chunk zips. `sha256sum -c` passed for the archive, the chunk zips, and the raw parts. `unzip -t` and `ZipFile.testzip()` reported no errors. The committed ZIP and chunk blobs are the raw bytes, not Git LFS pointers.

This records the returned archive. It makes no accuracy or performance claim. PR35 remains selected. Frozen 666 is the primary evaluation reference and 663 is historical only. Exploratory 662 and trainer 664 must not replace 666/663. Photo 010 is development validation only; trainer validation remains unchanged. This r24 comparison tests accumulated eligible supervision, not the isolated effect of the 26 point repairs.
