# Front-diagnosis-r50 T4 results

Provenance for the returned archive:

- Experiment: `front-diagnosis-r50` on the unchanged notebook from commit `424b56b05dd1c3a7e43674947068e226783c9e04`. This is a diagnosis run, not new training and not a promotion.
- Recorded flags: `training` false, `automatic_promotion` false, `fresh_inference` true, `primary_reference` frozen666.
- `SOURCE_REVISION` / `source` `5aa5c04d91fb6e1b5a5cf9cce85215f7564238aa`, `SCRIPT_SHA256` `cba4f1df6dcf431c933af1506a4de15718636d9477406eb34779ac4d8ab6ee3e`, `R50_REFERENCE_SHA256` `f639459bdd986f6553b9e00f0ecfdf1eee3c7f5344e95c300a4a17a66352c04f`.
- Frozen 666 stays primary. PR35 remains the selected training comparison.
- Notebook: `notebooks/baomai_r50_diagnosis.ipynb` at commit `424b56b05dd1c3a7e43674947068e226783c9e04`, SHA-256 `203f2572bc3e363416b4538a0ab83001f475b6b114201c1960ca963a0ded3ca8`, run unchanged.
- Drive copy: https://colab.research.google.com/drive/1JQ02JZOq6u1s4_9u1_tmwzRDQevjHGtv?authuser=1
- Runtime: A100 High-RAM was requested; Colab connected authorized T4 High-RAM fallback (“The selected GPU type was not available. You are now connected to a T4.”). Hostname `bff438c4e1f3`. Tesla T4 15360 MiB, Driver 580.82.07, CUDA 13.0. UI RAM 50.99 GB (High-RAM), Disk 235.68 GB. PyTorch 2.14.1+cu130, inference device 0. Runtime released after the ZIP; Manage Sessions empty; Reconnect was not clicked.
- Archive: `baomai-r50-diagnosis-results.zip`, 15997796 bytes, SHA-256 `c22da73bf196121cc9090e19b443ae8aa1d7c89a6de5d18353e0d4857d65014f`, 29 members. This hash matches the Cell 6 Return line for `/content/baomai-r50-5aa5c04d91fb/run/baomai-r50-diagnosis-results.zip`. `ZipFile.testzip()` was clean. Committed as a single ordinary git `100644` blob, not Git LFS, not a split ZIP_STORED pack.
- This pull request is based on current `main` `b4df76935178de5a122119aea48e85d2d661effc`. This results pack does not merge, close, or edit earlier PRs, including draft PR 142 (https://github.com/Siuuuers/dl/pull/142), results PR 143 (https://github.com/Siuuuers/dl/pull/143), PR 140 (https://github.com/Siuuuers/dl/pull/140), draft PR 139 (https://github.com/Siuuuers/dl/pull/139), and PR 137 (https://github.com/Siuuuers/dl/pull/137).

Recorded counts, transcribed as printed, not as a selection or promotion. Cell 6:

- frozen666 saved tp632/fn34/fp91
- historical0663 tp633/fn30/fp90
- r38 tp639/fn15/fp84
- r50 saved tp639/fn15/fp84 (soft fp83)
- Fresh miss mechanisms identical for r38 and r50: confidence_gate 10, no_iou_eligible_decoded_box 3, ordinary_nms 2

Independently measured archive members (SHA-256 and size), hashed from the extracted zip members:

- `summary.json`, 4770402 bytes, SHA-256 `c81fd63f5a76f03c4ccecf7c1e673f81abc8c215f9ec6c7fc42762b08150df85`
- `predictions.json`, 2427401 bytes, SHA-256 `faf22be5852ff28eaf3f773bbfa847c2bfe7458a82d6af2a21a4d90bbfa00201`
- `trace.json`, 9757 bytes, SHA-256 `98560e2a332f0c584825509e5b89015af08926b8d50856523551f745cee37b83`
- `missed-cases.json`, 136189 bytes, SHA-256 `a00eab6d458526020484d4a6d093427d4b3702a135520db73e2f74072222e37b`
- `timing.json`, 3117 bytes, SHA-256 `842936c80bb2f01447dda2d5debd8bf865ee331a52e697fcbf86807fd82d2b5e`
- `input-audit.json`, 44712 bytes, SHA-256 `86e11ccff6c085b87c600723abf9015f993098b0120df3e04bd4b90f5296bef1`
- `preflight.json`, 156051 bytes, SHA-256 `d6757e85382f415b9a2b44091363ab2a201701943c7c84078f040c1c300ccbf3`
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
- `diagnose_fronts_r50.py`, 33386 bytes, SHA-256 `cba4f1df6dcf431c933af1506a4de15718636d9477406eb34779ac4d8ab6ee3e`
- `inventory.json`, 3777 bytes, SHA-256 `6a600e11d782fc62a7560a357711752951a45463a75415fe70b11d47325a7c02`

This records the returned archive. It makes no accuracy or performance claim. It is a diagnosis result, not a training result, and it is not an automatic model promotion. PR35 remains selected.
