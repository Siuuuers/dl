# Front-review-r14 seed-46 L4 results

Provenance for the returned archive:

- Source revision: `4d805995e41e3520b10d0b2f743b1219c8356ca9`
- Notebook pin commit: `a928f6beda7a797eeb21fc3961db7f1028555e4c`; notebook `notebooks/baomai_front_review_r14.ipynb` SHA-256 `70851ef2e71d972cb1d9dad8752f4acd54a0617837aa7d2101b737a4a49e57e3`
- Protocol: `experiments/front-review-2026-11-21-r14.json` SHA-256 `a82e3b13e25ef9d4fdd405848bb4bafac16d658b92b9d1fadcc263ee3055cc12`
- Runtime: Colab NVIDIA L4
- Roles: reference and candidate
- Both roles are complete at 30/30 epochs, exit code 0, final-epoch checkpoints
- Archive: `baomai-front-review-r14-results.zip`, 39781759 bytes, SHA-256 `8df7ef283d2b05a642fd22ae74b52d1758277ee06dba185bbf42f26243936f51`

The archive was reassembled by concatenating raw `part00` then `part01` after unzipping the stored chunk zips. `sha256sum -c` passed for the archive, the chunk zips, and the raw parts. `unzip -t` and `ZipFile.testzip()` reported no errors. The committed ZIP and chunk blobs are the raw bytes, not Git LFS pointers.

This records the returned archive. It makes no accuracy or performance claim. PR35 remains selected. Frozen 666 is the primary evaluation reference and 663 is historical only. This exploratory r14 review does not replace trainer 664 val. The treatment combines annotation amendments and newly approved label eligibility; a result measures the combined effect and cannot isolate either component.
