# Front-review-r43 CPU results

Provenance for the returned archive:

- Experiment: `front-review-r43` (standard-CPU review of the cumulative annotations; not training and not a promotion)
- Runtime: standard CPU Colab (Python 3, High-RAM off). Hostname `f47dc9c0a89a`. RAM ~1.20 / 12.67 GB. Disk ~0.08 / 225.83 GB. No GPU. `nvidia-smi` command not found (exit 127). Runtime released; Manage Sessions empty.
- Notebook: pinned commit `a4ead2ee8de56a5e9e125947b78d09b1fce84b92` (“Group R43 review by frame and preserve cumulative corner amendments”). Local notebook sha256 `0c7c1e350d7e00c245b2106b056073d9652732d8b76f18e7ea3e0dc03a1e622c`, 8009 bytes, unchanged. The Drive copy was run unchanged: https://colab.research.google.com/drive/1m6Rhez6w6RslmfSf0xPr6NLnqYfbqzLU?authuser=1
- Recorded operation: review of cumulative annotations (kind: `r43-targeted-corner-review-with-complete-frame-history`). No training or model promotion. Complete images remain candidate annotations with no assigned training split or human sign-off.
- Archive: `baomai-front-review-r43-results.zip`, 4293004 bytes, SHA-256 `ae3d7310fd6b2d3a623bd7b8760dc9d6ffc75e6df1be28bbf0a2555fdb380d6e`, 73 members. `unzip -t` and `ZipFile.testzip()` were clean. The committed ZIP is a single ordinary git `100644` blob, not a Git LFS pointer and not a split ZIP_STORED pack. This is the immutable R43 review capsule (same identity as git blob `ab2fc7fbd37801b8a6b3cf1ce21002aa16f5b408` already referenced by the pinned notebook).
- This pull request is based on current `main` `b4df76935178de5a122119aea48e85d2d661effc`. This results pack does not merge, close, or edit earlier PRs.

Recorded counts:

- 12 original images, 508 identities and 507 current fronts verified
- This round: 9 amended rows; 1 reviewed and retained
- Complete annotation candidates: frame_0383, frame_0146, frame_0371
- Held whole images: frame_0001, frame_0160, frame_0038, frame_0074, frame_0128, frame_0178, frame_0225, frame_0280, frame_0301

This records the returned archive. It makes no accuracy or performance claim. It is a review result, not a training result, and it is not an automatic model promotion. Complete images remain candidate annotations with no assigned training split or human sign-off.
