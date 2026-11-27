# Flip-inference-r56 A100 results

Provenance for the returned archive:

- Experiment: `flip-inference-r56` on the unchanged notebook from notebook-publish commit `4e90155d672d5565ca620156e15aab04dc230eec`. This is a development inference comparison, not new training and not a promotion.
- Recorded flags from `summary.json`: `status` completed, `interpretation_allowed` true, `baseline_passed` true, `repeated_outputs_stable` true, `fresh_inference` true, `training` false, `automatic_promotion` false, `treatment_executed` true, `fusion_performed` false, `schema` `baomai-r56-flip-results-v1`.
- `SOURCE_COMMIT` / `source` `dc7eec630bb0b613fc1fe784e4d9818a0893658e`, notebook-publish commit `4e90155d672d5565ca620156e15aab04dc230eec`, driver / `script_sha256` `bf2ff9056f12976ca8c4aa971de2d2a7af1c679369c44dbf73de9f9d2542700d` (the `run_flip_inference_r56.py` member in this ZIP matches that hash), checkpoint `222bf3c91a651183047a10eb1a50e16ae8fe999f15372afe005bfca2d0b157f9`, input archive `fc8c40357c83831d17f88b0bb5a5ecd2e0cb1a8128be697e37e80620e4981023`, 20 images.
- Baseline-equivalence passed (`passed` true) with `exact_json_equal` false and maximum coordinate-component drift 0.0001220703125 px, within the 0.05 px / 1e-5 tolerances (`coordinate_component_px` 0.05, `probability` 1e-5; `endpoint_error_px` tolerance 0.075, observed endpoint-error drift 6.766371575572805e-05 px).
- `summary.json` interpretation: "Development comparison only. No benefit is established until this returned evidence is reviewed; no model, label, filter or threshold is promoted." This is development evidence only. PR35 remains the selected deployment model. Nothing is automatically promoted.
- Notebook: `notebooks/baomai_flip_inference_r56.ipynb` at commit `4e90155d672d5565ca620156e15aab04dc230eec`, run unchanged.
- Drive copy: https://colab.research.google.com/drive/1ybRX58CF33oa7dJzaCIv2gylNJLQ75QJ?authuser=1
- Runtime: A100 + High-RAM was requested and connected. Hostname `ebcc38998981`. NVIDIA A100-SXM4-40GB, High-RAM 83.5 GB RAM / 40 GB GPU. From `runtime.json`: CUDA 13.0, cuDNN 92800, Python 3.13.16, torch 2.14.1+cu130, torchvision 0.29.1+cu130, ultralytics 8.4.182, device `cuda:0`. The runtime was released after the ZIP was saved.
- Settings from `summary.json` contract: experiment `flip`, control geometry `identity` with imgsz 1280, treatment geometry `horizontal_flip` with imgsz 1280, conf 0.13, iou 0.7, matching_iou 0.5, endpoint_conf 0.5, max_det 300, nms true, batch 1, fusion false.
- Archive: `baomai-flip-inference-r56-results.zip`, 2931102 bytes, SHA-256 `88689ce9f3d03178683ea98ce98d50f5ddc4e07ba1fb24262e7820e9488bd114`, 16 ordinary members. `unzip -t` and `ZipFile.testzip()` were clean. Committed as a single ordinary git `100644` blob, not Git LFS, not a split ZIP_STORED pack.
- This pull request is based on current `main` `b4df76935178de5a122119aea48e85d2d661effc`. This results pack does not merge, close, or edit earlier PRs, including draft PR 151 (https://github.com/Siuuuers/dl/pull/151), results PR 150 (https://github.com/Siuuuers/dl/pull/150), results PR 149 (https://github.com/Siuuuers/dl/pull/149), draft PR 148 (https://github.com/Siuuuers/dl/pull/148), draft PR 147 (https://github.com/Siuuuers/dl/pull/147), results PR 146 (https://github.com/Siuuuers/dl/pull/146), results PR 144 (https://github.com/Siuuuers/dl/pull/144), results PR 143 (https://github.com/Siuuuers/dl/pull/143), PR 140 (https://github.com/Siuuuers/dl/pull/140), draft PR 145 (https://github.com/Siuuuers/dl/pull/145), draft PR 142 (https://github.com/Siuuuers/dl/pull/142), draft PR 139 (https://github.com/Siuuuers/dl/pull/139), and PR 137 (https://github.com/Siuuuers/dl/pull/137).

Recorded postclip counts, transcribed from `summary.json` detection as `fn` / unmatched `fp` / `duplicate_fp` and two-corner `within5`/`eligible`. Cell 8 postclip counts match these figures. These are not a selection or promotion:

- frozen666 control 34/91/17 511/584 vs treatment 32/84/23 518/584
- historical663 control 30/90/17 511/584 vs treatment 30/85/22 517/584
- r38 control 15/84/14 566/583 vs treatment 11/75/21 562/583
- r50 control 15/84/13 566/583 vs treatment 11/75/22 562/583

Complementarity, transcribed from `complementarity.json` locatable=`all` (Cell 8 matches): frozen666 flip-only recoveries 4, control-only 2, missed by both 30; historical663 3/3/27; r38 6/2/9; r50 6/2/9. This is a diagnostic bound, not a fused detector (`fusion_performed` false).

Independently measured archive members (SHA-256 and size), hashed from the extracted zip members:

- `baseline-equivalence.json`, 517 bytes, SHA-256 `a4d081beb4e870dff5b798ade4e00793bb5b1975baa2bee953580f54138216f0`
- `cases.json`, 6279804 bytes, SHA-256 `589d3c664df7f02caa1ede27a0434cd7d35c34a3e70045e0871120d62c5d7499`
- `complementarity.json`, 25313 bytes, SHA-256 `1be30758369adf4b88be88c1ad02878f15228de25a15765bbc29329f246c5de3`
- `input-audit.json`, 44712 bytes, SHA-256 `86e11ccff6c085b87c600723abf9015f993098b0120df3e04bd4b90f5296bef1`
- `inventory.json`, 2010 bytes, SHA-256 `8e8958714ac8206b6cfbe89fa6ccc27a7e6126cec7c0527d25e2dfa67f6ba18e`
- `native-predictions.json`, 725107 bytes, SHA-256 `e33e79dc843ab65dddb056a862b90ecbad08b1929130f3019bb5d3cd9ee609b6`
- `predictions.json`, 1668310 bytes, SHA-256 `fc3c2ad7638badd218addf196d23d831c7ea19130f6439fb3dfb28c3b5a4d8d0`
- `preflight.json`, 161392 bytes, SHA-256 `63ba7e37d106f85ce305fe42259f6d507d098d6e28b974c6071e84df286e12e8`
- `repeatability.json`, 3243 bytes, SHA-256 `f3933737dbff14a1ea658ee4753092b97248348fbda482a3729b43d15d28e356`
- `repeated-native-predictions.json`, 2368527 bytes, SHA-256 `2e9c5ddf648b1694a3522968530e5da0fadc6c16a53a1931ced8d1ef5b11b6a6`
- `repeated-predictions.json`, 5411346 bytes, SHA-256 `bea085dd099aa78924b0898f42f91b2a19138b395c59ce6b8a2aef28ba9b9787`
- `run_flip_inference_r56.py`, 28781 bytes, SHA-256 `bf2ff9056f12976ca8c4aa971de2d2a7af1c679369c44dbf73de9f9d2542700d`
- `run_inference_ablation_r53.py`, 36334 bytes, SHA-256 `8c8a37c3dcaccde5c568e9df6f44651cdb05965ca5f5ed5b0f2978852387dcc1`
- `runtime.json`, 5381 bytes, SHA-256 `02d3f638f2dc6ac82caaa2bce2c50cdcf348d21e5d8b76bc181a73a570d12b2e`
- `summary.json`, 3425215 bytes, SHA-256 `3f39ec412d3d40fd17d8c629abb867af4a2025d0e351f6db12236adde73e542e`
- `timing.json`, 140733 bytes, SHA-256 `3037e15e74f1121d136711405e4eafc617d1954f3c90b282b9c4e3053366fc1b`

This records the returned archive. It makes no accuracy or performance claim. It is development evidence only, not a training result, and it is not an automatic model promotion. PR35 remains the selected deployment model.
