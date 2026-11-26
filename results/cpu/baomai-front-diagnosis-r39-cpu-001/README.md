# Front-diagnosis-r39 CPU results

Provenance for the returned archive:

- Experiment: `front-diagnosis-r39` (saved-evidence replay of PR123 against accepted R38 labels; not new training)
- Source revision: `a3b76772f40bdb8beb3005015c86d99a4eb6b89f`. This results pack does not merge, close, or edit earlier PRs.
- Notebook: `notebooks/baomai_front_diagnosis_r39.ipynb`, 20001 bytes, SHA-256 `7171b8d478285fb70f09f55240e1c78968a40de86cc93b7eb73a59c022179a61`; pinned GitHub/Colab blob `b136c324ad8ed86efe8f6ea3542e46653b047029`. The Drive copy was not edited: https://colab.research.google.com/drive/1FmGMoM9hHzwWKQS_27JocMOx3wiz28iZ?authuser=1
- Runtime: standard CPU Colab, hostname `1ba1f7d99819`, RAM ~1.20/12.67 GB, High-RAM off, `nvidia-smi` not found
- Recorded operation: saved-evidence replay. `training` false, `fresh_inference` false, `automatic_promotion` false
- Archive: `baomai-front-diagnosis-r39-results.zip`, 286874 bytes, SHA-256 `f79abc03a0f46ce80c94e2747d19d9bd41d1876b7d6bcdee53f023bc4b296d8a`, 12 members, `date_time` 2026-11-26 00:00:00. This hash matches the published notebook-validation.json archive. `unzip -t` and `ZipFile.testzip()` were clean. The committed ZIP is a single ordinary git `100644` blob, not a Git LFS pointer and not a split ZIP_STORED pack.
- This pull request is based on current `main` `b4df76935178de5a122119aea48e85d2d661effc`.

Recorded counts (outputs / matches / misses / unmatched, with unmatched split as duplicate / other):

- Baseline confidence 0.13: 737 / 643 / 11 / 94 (29 / 65)
- Clipped NMS 0.70: 728 / 643 / 11 / 85 (21 / 64)
- Exploratory confidence 0.05: 810 / 648 / 6 / 162 (48 / 114); this threshold is not adopted

Mechanisms for the 11 remaining baseline misses: 6 `confidence_gate`, 4 `no_iou_eligible_decoded_box`, 1 `nms_suppression`. Extra clipped NMS removes nine unmatched outputs and preserves every existing matched prediction on R38 and frozen666 (`lost_matches` empty; `matched_primary_prediction_associations_exact` true). The clip-NMS probe is not selected.

Independently measured archive members (SHA-256 and size), hashed from the extracted zip members:

- `baseline-predictions.jsonl`, 226474 bytes, SHA-256 `7ec91a12a2882f5b641683963556f1e5e16148f376c118432b4d95bf1da493ef`
- `clip-nms-probe.json`, 56825 bytes, SHA-256 `a297d37f6ddfc5670fb415cbc43bc22c1136e917854a0e878ad4396d67440d52`
- `clip-nms-summary.json`, 12486 bytes, SHA-256 `222f26c98c842cfd13b753029db6cc9d0b69cd948d4807d683544625b8270971`
- `clip-nms-visual-review.json`, 11755 bytes, SHA-256 `58b298edb35dce41cb99e2718a0f0ff3ef98e3439b19f57af38e932b32a42a45`
- `clipped-nms-predictions.jsonl`, 207530 bytes, SHA-256 `d4b404997b891192839c6551fa4513fea49f2c51bbc7cfb9212185ace45ff0f2`
- `counterfactual-predictions.jsonl`, 232197 bytes, SHA-256 `b575a2bc4d2009248db05db8570619913dfc3a495120187521b1e563cc77c717`
- `diagnosis.json`, 218189 bytes, SHA-256 `b40017371a072498886e3f09f27d61fef090dd98ace0ee10fdbaa6eb44829a8e`
- `duplicate-overlap-audit.json`, 24553 bytes, SHA-256 `e524ab2d9c81c6723830195ef9fdce02cd59b0b2f1b8d3867a64bb3efe2665b8`
- `missed-fronts.csv`, 1157 bytes, SHA-256 `09a62dbf2f1fe4c35216c8160f6f286b18c010b84204722f56be80d2a4b16468`
- `next-steps.md`, 11902 bytes, SHA-256 `90a477377b22bfb302c9441d3622178f85c7764773d6cc21a330df806ff88ce2`
- `source.json`, 8451 bytes, SHA-256 `b9c06ab82aabc25b7dbabfb49f96d4e91293f19fd6b8ee931f90fb7e2a41b0f3`
- `summary.json`, 7143 bytes, SHA-256 `ece57cf7f836a7d2fe476e5f552db3f896aaf31502c99b2dec8f1a50091e09ac`

This records the returned archive. It makes no accuracy or performance claim. It is a saved-evidence replay, not a training result, and it is not an automatic model promotion.
