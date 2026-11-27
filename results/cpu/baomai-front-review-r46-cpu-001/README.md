# Front-review-r46 CPU results

Provenance for the returned archive:

- Experiment: `front-review-r46` (standard-CPU review plus PR133 result replay; not training and not a promotion). `verification.json`: `passed` true, `training` false, `promotion` false, `new_training` false, `human_review_performed` false, `round` 46. Frozen 666 stays primary and 663 historical/diagnostic. PR35 remains selected.
- Runtime: standard CPU Colab (Python 3, High-RAM off). Hostname `8b796bbb702a`. RAM ~1.2 / 12.67 GB. Disk ~0.1 / 225.83 GB. No GPU. `nvidia-smi` command not found. Runtime released; Manage Sessions empty. Reconnect chip was not clicked.
- Notebook: pinned publish commit `d49a55d7c2240bbb18565e643cf01e691d39a7a3` (“Pin and verify the four-cell R46 CPU review notebook”). `SOURCE_REVISION` `27891039b7ab4d7c66d503ccf2312659b85e68ed` (printed by Cell 2). Local notebook sha256 `7ca721b9060970e0bcd4b681871a8949e8588577f2e319bc35db73c0836b1feb`, 8569 bytes, LF-only, unchanged. The Drive copy was run unchanged: https://colab.research.google.com/drive/1geE5X1iuLdU1Hq5mRQGVnd1P6V7KQ8w9?authuser=1 (authuser=1, GloriaWingyin@gmail.com).
- Recorded operation: standard-CPU review plus PR133 result replay. Does not train and is not a promotion. `REPLAY_PR133` was True. The replay is archive verification only, not a promotion. Complete images remain candidate annotations with no assigned training split or human sign-off.
- Archive: `baomai-front-review-r46-results.zip`, 9988074 bytes, SHA-256 `05a4e4cfebc2e4244871de8ddc18ea5cc1a0c2ba934083400c3ec91c7e9c6fe6`, 69 members. `unzip -t` and `ZipFile.testzip()` were clean. The committed ZIP is a single ordinary git `100644` blob, not a Git LFS pointer. The ZIP starts with `PK\x03\x04`. Both PDFs are inside the ZIP and also exist as loose downloads: `r46-annotation-amendments.pdf`, 3064964 bytes, SHA-256 `59a3fe50f75f4d6be827a6307b835bd94caa5ea6fa118bf3e947f92b4040440d`; `r45-pilot-results-and-missed-review.pdf`, 6914928 bytes, SHA-256 `fb95c8f8a43cc24d05826a28f674188eaa1e333480cf8f03b35b9a23f0a16493` (matches in-ZIP `r45-pilot-results-and-missed-review.verification.json`; `cpu_audit_status` passed). This pull request does not add the loose PDFs.
- This pull request is based on current `main` `b4df76935178de5a122119aea48e85d2d661effc`. This results pack does not merge, close, or edit earlier PRs, including draft PR 134 (https://github.com/Siuuuers/dl/pull/134, still draft on `codex/r46-corner-ownership-review-20261127`).

Recorded counts (Cell 4 / `verification.json` `latest_counts`):

- `reviewed_rows` 507, `amended_rows` 29, `retained_rows` 478, `corner_coordinate_slots_changed` 27, `box_boundaries_changed` 17, `added_rows` 0, `deleted_rows` 0, `visibility_changes` 0, `requested_rows` 17, `requested_rows_amended` 17
- All 508 stable identities preserved (`verification.json`)
- Complete-image candidates: frame_0383 only
- Held whole photographs (11): frame_0001, frame_0160, frame_0038, frame_0074, frame_0128, frame_0146, frame_0178, frame_0225, frame_0280, frame_0301, frame_0371
- Complete images remain candidate annotations with no assigned training split or human sign-off

This records the returned archive. It makes no accuracy or performance claim. It is a review result, not a training result, and it is not an automatic model promotion. PR35 remains selected.
