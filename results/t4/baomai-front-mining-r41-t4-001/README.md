# Front-mining-r41 T4 results

Provenance for the returned archive:

- Experiment: `front-mining-r41` (kind `r41-unqualified-upstream-review-mining`; not training and not a promotion)
- Runtime: Tesla T4 Colab. Colab requested L4; A400 was not listed; a T4 fallback was authorized. Hostname `6110921954c6`. NVIDIA-SMI 580.82.07, Driver 580.82.07, CUDA 13.0, 15360 MiB T4. torch 2.14.1+cu130, python 3.13.16. Receipt `device_name` Tesla T4, `device_actual` cuda:0.
- Notebook: the Drive copy was run unchanged: https://colab.research.google.com/drive/1iMYj1NlctugUkgmn1a6VbJiiRfVo3auI?authuser=1
- Recorded operation: mining inference. No training or promotion.
- Archive: `baomai-front-mining-r41-results.zip`, 41161728 bytes, SHA-256 `c73f3cafd916532e727cd6467f580253fa2dd3778f3f19169eb9abc45972f218`, 97 members. `unzip -t` and `ZipFile.testzip()` were clean. The committed ZIP is a single ordinary git `100644` blob, not a Git LFS pointer and not a split ZIP_STORED pack.
- This pull request is based on current `main` `b4df76935178de5a122119aea48e85d2d661effc`. This results pack does not merge, close, or edit earlier PRs.

Recorded counts:

- 320 prediction sets
- 12 full review frames: frame_0001, frame_0160, frame_0383, frame_0038, frame_0074, frame_0128, frame_0146, frame_0178, frame_0225, frame_0280, frame_0301, frame_0371

This records the returned archive. It makes no accuracy or performance claim. It is a mining result, not a training result, and it is not an automatic model promotion.
