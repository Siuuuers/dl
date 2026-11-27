# Front-diagnosis-r49 T4 results

Provenance for the returned archive:

- Experiment: `front-diagnosis-r49`, frozen-model proposal diagnosis and Soft-NMS pilot on the exact PR137 candidate checkpoint and the original 20 development photos; not new training and not a promotion.
- `summary.json`: `training` false, `fresh_inference` true, `automatic_promotion` false, `same_tensor_comparison` true, `saved_baseline_exact_images` 20, `source` `8cf5a2afc9fe28ba0f4f4596c27944761a790b76`, `script_sha256` `188df92036d5dc8fd8e13fc8d8815728a56c33ced654de2cf4716788eac47811`, `archive_sha256` `fc8c40357c83831d17f88b0bb5a5ecd2e0cb1a8128be697e37e80620e4981023` (this is the PR137 input archive, 42395650 bytes, not the r49 output hash), `checkpoint_sha256` `222bf3c91a651183047a10eb1a50e16ae8fe999f15372afe005bfca2d0b157f9`.
- Frozen 666 stays primary and 663 historical/diagnostic. PR35 remains selected.
- Notebook: `notebooks/baomai_r49_diagnosis.ipynb` at checkout-fix commit `55a6f68993a63fcdf422014995fada33352720df` (PR 140), 28068 bytes, SHA-256 `63060941e7834a60562d97ddf3dd1bb46f2feb921b737ae744aaca0d329026d2`, run unchanged.
- Drive copy: https://colab.research.google.com/drive/1WiqNxoacCFHehgsJAmeTYxql7q_b7Mlm?authuser=1 (authuser=1, gloriawingyin@gmail.com / “in my heart”). Title: Copy of baomai_r49_diagnosis.ipynb.
- Runtime: Colab Tesla T4 High-RAM (Python 3). A100 then L4 were requested; Colab said the requested GPU was unavailable and connected T4, which is authorized. Hostname `080bf3c97061`. System RAM ~2.0/51.0 GB, GPU RAM 15.0 GB (Tesla T4 15360 MiB), Disk ~0.3/235.7 GB. `summary.json` runtime: Python 3.13.16, torch 2.14.1+cu130, torchvision 0.29.1+cu130, Ultralytics 8.4.182, device 0, gpu Tesla T4. Runtime released; Manage Sessions empty; Reconnect was not clicked.
- Settings from `summary.json`: imgsz 1280, conf 0.13, iou 0.7, matching_iou 0.5, endpoint_conf 0.5, max_det 300, nms true, batch 1. Soft-NMS: linear, IoU > .70, score *= 1-IoU, strict conf > .13, max_det 300, same postclip.
- Archive: `baomai-r49-diagnosis-results.zip`, 15631804 bytes, SHA-256 `892c8ee79ff470990bb47dc59f947ad6bd2577a86bc9827e9dd57f81c56213f3`, 28 members. This hash matches the Cell 8 Return line: Return: /content/baomai-r49/work/baomai-r49-diagnosis-results.zip SHA-256: 892c8ee79ff470990bb47dc59f947ad6bd2577a86bc9827e9dd57f81c56213f3. `unzip -t` and `ZipFile.testzip()` were clean. Committed as a single ordinary git `100644` blob, not Git LFS, not a split ZIP_STORED pack.
- This pull request is based on current `main` `b4df76935178de5a122119aea48e85d2d661effc`. This results pack does not merge, close, or edit earlier PRs, including draft PR 139 (https://github.com/Siuuuers/dl/pull/139) and PR 140 (https://github.com/Siuuuers/dl/pull/140).

Recorded counts, transcribed as printed, not as a selection or promotion. Cell 8 / `summary.json`:

- Fresh baseline exact-image count: 20 / 20
- Miss stages: confidence_gate 10, no_iou_eligible_decoded_box 3, ordinary_nms 2
- frozen666 saved missed 34 unmatched 91 duplicate subset 17 both within 5px 511 / 584
- frozen666 hard missed 34 unmatched 91 duplicate subset 17 both within 5px 511 / 584
- frozen666 soft missed 34 unmatched 90 duplicate subset 17 both within 5px 511 / 584
- historical663 saved missed 30 unmatched 90 duplicate subset 17 both within 5px 511 / 584
- historical663 hard missed 30 unmatched 90 duplicate subset 17 both within 5px 511 / 584
- historical663 soft missed 30 unmatched 89 duplicate subset 17 both within 5px 511 / 584
- r38 saved missed 15 unmatched 84 duplicate subset 14 both within 5px 566 / 583
- r38 hard missed 15 unmatched 84 duplicate subset 14 both within 5px 566 / 583
- r38 soft missed 15 unmatched 83 duplicate subset 13 both within 5px 566 / 583

Independently measured archive members (SHA-256 and size), hashed from the extracted zip members:

- `summary.json`, 3493779 bytes, SHA-256 `279e78ffa021037850a6cbe84b1c98ac025fcecfe86636eed8bfba2ce50a5f34`
- `predictions.json`, 2427401 bytes, SHA-256 `87a951076a9010bc771d59c19dc762b8f23e59897c0426c2ab3ab52db3df2345`
- `trace.json`, 9757 bytes, SHA-256 `11ed322cfc77295aee6abbb7df702f935ca9d4c374a3d29eae66ad0e4373ddb8`
- `missed-cases.json`, 67512 bytes, SHA-256 `fa27652f508b287deed89f9219631d3c45c10bcf544b415fd716e74061557f4a`
- `timing.json`, 3121 bytes, SHA-256 `546930b464290480813844f0da55c7bdda54ff26dc6b25d176c5f86febc50c88`
- `input-audit.json`, 44712 bytes, SHA-256 `2a29a78614b938abc2fc160459e445a5ffa6177e1d0a19f04ebb3bed8f373940`
- `decoded/new_000007.npy`, 850208 bytes, SHA-256 `c6f48bc5b0d1da669144dc20393150628c4ba2a58789c87420e8180b3bff2b03`
- `decoded/new_000008.npy`, 850208 bytes, SHA-256 `9fc972844693bf7b9378e8fa36f62df0248c21d951d4f7dee9bd52dacca14eab`
- `decoded/new_000009.npy`, 850208 bytes, SHA-256 `db6ecdc6eb8a073e4c68076793335375803d0faed8b8fae691784c876685de9a`
- `decoded/new_000010.npy`, 850208 bytes, SHA-256 `e5fbedd5943267e899e4c16d56ff4d427a3e342787679f8c9f717614506a59ad`
- `decoded/new_000011.npy`, 850208 bytes, SHA-256 `4e6159a8e22b843c56c273f5a4d0fed503b17902f16473cb3658d57bf1c7055b`
- `decoded/new_000053.npy`, 850208 bytes, SHA-256 `f2a254ab92f92016be9fd1fb96d1d615a17d6bc4a56e23c04a8d7ebb2f29b6c4`
- `decoded/new_000054.npy`, 850208 bytes, SHA-256 `14d2807a80bfcd0a7fe321379c26e1c738bb60df85d59dce2eb7774e7782fc50`
- `decoded/new_000055.npy`, 850208 bytes, SHA-256 `5d75ec973e58d6442ff1f9bda7e238b6a827e1e60c75e08ff983547ed6d7d435`
- `decoded/new_000056.npy`, 850208 bytes, SHA-256 `e3f64680626ae12d016caec65d0a066d5ebe212e429a8a47004cbd8c49a03fff`
- `decoded/new_000057.npy`, 850208 bytes, SHA-256 `32d0645a2a6fa6795f3fe4abd8471df7b1dd42797e12901d20f095aea1f808df`
- `decoded/new_000058.npy`, 850208 bytes, SHA-256 `6f160c0b785183e7e6865f85bf5bac7c059f82cd1f48e44c3eb408dccf5c7c77`
- `decoded/new_000059.npy`, 850208 bytes, SHA-256 `3903adb1f6bb3cb7206db49679063386aaaa3f0a14cdc0ea1e7de6d30eb97eae`
- `decoded/new_000086.npy`, 850208 bytes, SHA-256 `27c645a63c6deecc711e6b2f8c8290892aea8c229b0fa51c2e4bd6a27b056e90`
- `decoded/new_000087.npy`, 850208 bytes, SHA-256 `6a47d0babe0104b01fb6df1a00d72502873e6c4fd74319dd895f3416e1428588`
- `decoded/new_000088.npy`, 850208 bytes, SHA-256 `9d1153c0685f1095ebde07a9801bd9c2ba02de625c371bd6701539e8d3d4e15f`
- `decoded/new_000089.npy`, 850208 bytes, SHA-256 `f33f6fa7b79a1d0c9d0bc81990162d239f2d88e6233cd087554e125c96a47def`
- `decoded/new_000090.npy`, 850208 bytes, SHA-256 `0a062812dbed47547c57fb27191583345e6343b203c46ff731f9d5929c5b274b`
- `decoded/new_000146.npy`, 850208 bytes, SHA-256 `547a5c03efc797f309e18f5f8a3fa1fb229c4fd28f69c7a16c25493ac33e560c`
- `decoded/new_000147.npy`, 850208 bytes, SHA-256 `69d3f3094911da02fec878c318641df0b5e2d429972696dcafea1fc8860f46ff`
- `decoded/new_000148.npy`, 850208 bytes, SHA-256 `82c881e3ed78ded1212267ebe1bb6f102d9c3287cd24c4f9814d5a4b7bb6bf8b`
- `diagnose_r49.py`, 17299 bytes, SHA-256 `188df92036d5dc8fd8e13fc8d8815728a56c33ced654de2cf4716788eac47811`
- `inventory.json`, 3639 bytes, SHA-256 `691912e82aebec5fc50636d267df3ad9880d3ff3b935a6e4d8cd5e7a6417312c`

This records the returned archive. It makes no accuracy or performance claim. It is a diagnosis result, not a training result, and it is not an automatic model promotion. PR35 remains selected.
