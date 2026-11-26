# Front-review-r42 CPU results

Provenance for the returned archive:

- Experiment: `front-review-r42` (standard-CPU review of the PR127 annotations; not training and not a promotion)
- Runtime: standard CPU Colab (Python 3, High-RAM off). Hostname `45758521b490`. RAM ~12.67 GB. `nvidia-smi` not found.
- Notebook: pinned commit `bb7d38935e202de126ff2d3eee7df3de8f066a63`. The Drive copy was run unchanged: https://colab.research.google.com/drive/1zOyFtsIylveQfjVBsfl8ypkXx6KGghl2?authuser=1
- Recorded operation: review of PR127 annotations. No training or model promotion.
- Archive: `baomai-front-review-r42-results.zip`, 4181563 bytes, SHA-256 `716f81f9b72fffccfb52120d90b30a50b3d605ebd649736ed1f5ad189b4cb2ef`, 65 members. `unzip -t` and `ZipFile.testzip()` were clean. The committed ZIP is a single ordinary git `100644` blob, not a Git LFS pointer and not a split ZIP_STORED pack. This is the immutable R42 review capsule (same identity as git blob `f9488f0c17062bb87b901620f1ace714995a9543` already referenced by the pinned notebook).
- This pull request is based on current `main` `b4df76935178de5a122119aea48e85d2d661effc`. This results pack does not merge, close, or edit earlier PRs.

Recorded counts:

- 12 images and 507 baseline rows verified
- Complete annotation candidates: frame_0383, frame_0146, frame_0371
- Held whole images: frame_0001, frame_0160, frame_0038, frame_0074, frame_0128, frame_0178, frame_0225, frame_0280, frame_0301

This records the returned archive. It makes no accuracy or performance claim. It is a review result, not a training result, and it is not an automatic model promotion.
