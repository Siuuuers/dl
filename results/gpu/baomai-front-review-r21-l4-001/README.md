# Front-review-r21 seed-46 L4 results

Provenance for the returned archive:

- Source revision: `c4fb76afe280552ed13ed3c9b78608deceb7b3c4`
- Notebook pin commit: `41dd2be835f5b0e30b10e15431c6795cb00f7388`; notebook `notebooks/baomai_front_review_r21.ipynb` SHA-256 `1de82d9dbfb53e331048f132d96df01272e2e732d48b1454bbf12c820b005f7a`
- Protocol: `experiments/front-review-2026-11-22-r21.json` SHA-256 `6e79440caa989de5327531ebde2242aa3ba7f25de7497ebac8542ee766400122`
- Runtime: Colab NVIDIA L4
- Roles: reference and candidate
- Both roles are complete at 30/30 epochs, exit code 0, final-epoch checkpoints
- Archive: `baomai-front-review-r21-results.zip`, 39783675 bytes, SHA-256 `b3acac74a82dc5927242aa73b96b4c33978fee06aa1b1678b6382b237c91b802`

The archive was reassembled by concatenating raw `part00` then `part01` after unzipping the stored chunk zips. `sha256sum -c` passed for the archive, the chunk zips, and the raw parts. `unzip -t` and `ZipFile.testzip()` reported no errors. The committed ZIP and chunk blobs are the raw bytes, not Git LFS pointers.

This records the returned archive. It makes no accuracy or performance claim. PR35 remains selected. Frozen 666 is the primary evaluation reference and 663 is historical only. Exploratory 662 and trainer 664 must not replace 666/663.
