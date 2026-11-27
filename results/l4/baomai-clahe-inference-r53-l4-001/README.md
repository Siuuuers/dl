# Clahe-inference-r53 L4 results

Provenance for the returned archive:

- Experiment: `clahe-inference-r53` on the unchanged notebook from commit `5001d83439eaf31c7ecae378c27a0de847e65686`. This is a development inference comparison, not new training and not a promotion.
- Recorded flags from `summary.json`: `status` completed, `interpretation_allowed` true, `baseline_passed` true, `repeated_outputs_stable` true, `fresh_inference` true, `training` false, `automatic_promotion` false, `treatment_executed` true.
- `SOURCE_REVISION` / `source` `7b36e23dbbc24769f92a88cb93ae44acfc0406c5`, notebook-publish commit `5001d83439eaf31c7ecae378c27a0de847e65686`, `DRIVER_SHA256` / `script_sha256` `8c8a37c3dcaccde5c568e9df6f44651cdb05965ca5f5ed5b0f2978852387dcc1`, checkpoint `222bf3c91a651183047a10eb1a50e16ae8fe999f15372afe005bfca2d0b157f9`, input archive `fc8c40357c83831d17f88b0bb5a5ecd2e0cb1a8128be697e37e80620e4981023` (this is the input archive, 42395650 bytes, not the r53 output hash).
- `summary.json` interpretation: "Development comparison only. No benefit is established until this returned evidence is reviewed; no model, label, filter or threshold is promoted." Frozen 666 stays primary. PR35 remains the selected training comparison.
- Notebook: `notebooks/baomai_clahe_inference_r53.ipynb` at commit `5001d83439eaf31c7ecae378c27a0de847e65686`, run unchanged.
- Drive copy: https://colab.research.google.com/drive/1XTc7Rs2YfaHBGamkUO4PNDIUYshltAFO?authuser=1
- Runtime: A100 was requested first; Colab connected authorized L4 High-RAM instead. Hostname `47d0ab4fdc62`. NVIDIA L4, 23034 MiB, GPU UUID `GPU-5cc1a4cc-aafd-3392-7657-277d75a0d6a7`. From `runtime.json`: CUDA 13.0, cuDNN 92800, Python 3.13.16, torch 2.14.1+cu130, torchvision 0.29.1+cu130, ultralytics 8.4.182, device `cuda:0`.
- Settings from `summary.json` contract: experiment `clahe`, control photometry `identity`, treatment photometry `lab_l_clahe` (clipLimit 2.0, tileGridSize 8x8), imgsz 1280, conf 0.13, iou 0.7, matching_iou 0.5, endpoint_conf 0.5, max_det 300, nms true, batch 1.
- Archive: `baomai-clahe-inference-r53-results.zip`, 2343077 bytes, SHA-256 `1e1dde4528729deced36a7943f1a3fc68e4e4993e05c0289e10fcf03f2cfa56d`, 12 ordinary members. `ZipFile.testzip()` was clean. Committed as a single ordinary git `100644` blob, not Git LFS, not a split ZIP_STORED pack.
- This pull request is based on current `main` `b4df76935178de5a122119aea48e85d2d661effc`. This results pack does not merge, close, or edit earlier PRs, including draft PR 148 (https://github.com/Siuuuers/dl/pull/148), draft PR 147 (https://github.com/Siuuuers/dl/pull/147), results PR 146 (https://github.com/Siuuuers/dl/pull/146), results PR 144 (https://github.com/Siuuuers/dl/pull/144), results PR 143 (https://github.com/Siuuuers/dl/pull/143), PR 140 (https://github.com/Siuuuers/dl/pull/140), draft PR 145 (https://github.com/Siuuuers/dl/pull/145), draft PR 142 (https://github.com/Siuuuers/dl/pull/142), draft PR 139 (https://github.com/Siuuuers/dl/pull/139), and PR 137 (https://github.com/Siuuuers/dl/pull/137).

Recorded counts, transcribed from `summary.json` postclip detection as printed, not as a selection or promotion. Cell 8 postclip counts match these figures:

- frozen666 control postclip tp632/fn34/fp91 duplicates 17 two-corner rows within5px 511/584
- frozen666 treatment postclip tp633/fn33/fp92 duplicates 19 two-corner rows within5px 506/584
- historical663 control postclip tp633/fn30/fp90 duplicates 17 two-corner rows within5px 511/584
- historical663 treatment postclip tp632/fn31/fp93 duplicates 18 two-corner rows within5px 506/584
- r38 control postclip tp639/fn15/fp84 duplicates 14 two-corner rows within5px 566/583
- r38 treatment postclip tp640/fn14/fp85 duplicates 23 two-corner rows within5px 564/583
- r50 control postclip tp639/fn15/fp84 duplicates 13 two-corner rows within5px 566/583
- r50 treatment postclip tp640/fn14/fp85 duplicates 22 two-corner rows within5px 564/583

Independently measured archive members (SHA-256 and size), hashed from the extracted zip members:

- `baseline-equivalence.json`, 517 bytes, SHA-256 `a4d081beb4e870dff5b798ade4e00793bb5b1975baa2bee953580f54138216f0`
- `cases.json`, 6311351 bytes, SHA-256 `4208adcb2fdfc9a0e0b59de37421fa09ff61675b1ebd03d28c5f57f32541d326`
- `input-audit.json`, 44712 bytes, SHA-256 `86e11ccff6c085b87c600723abf9015f993098b0120df3e04bd4b90f5296bef1`
- `inventory.json`, 1451 bytes, SHA-256 `936e1700c9a12f2a60230123ee9487179c798cdc28374f598a5fd19d7d6ed2b1`
- `predictions.json`, 1674974 bytes, SHA-256 `1be54e0ef3f087940014b8abb269dc9509524f9d573b225e4ec68bbf42d33bb2`
- `preflight.json`, 160769 bytes, SHA-256 `33d503d8cefab3fbb1323d1b167a6731c9acfd5301111e3d3b5e5dc7ebd1bc24`
- `repeatability.json`, 3243 bytes, SHA-256 `f3933737dbff14a1ea658ee4753092b97248348fbda482a3729b43d15d28e356`
- `repeated-predictions.json`, 5432970 bytes, SHA-256 `8d466836a0b99ea73c62cc45f5b047509c3868433d5a259ae1397e749f4b007f`
- `run_inference_ablation_r53.py`, 36334 bytes, SHA-256 `8c8a37c3dcaccde5c568e9df6f44651cdb05965ca5f5ed5b0f2978852387dcc1`
- `runtime.json`, 4571 bytes, SHA-256 `d2fd60f4f991ed5affbd816251513d6e49d80f8363afc4cd2a7254d6ce9920a4`
- `summary.json`, 3427288 bytes, SHA-256 `5cb6ebf574e85743ad404901e5ef4f96cc73075fa582d54f8adbc8e07f62aa1b`
- `timing.json`, 140332 bytes, SHA-256 `24dba16960c322c29830231b73466a6912f023a44fdf05dfdeb656881c18c833`

This records the returned archive. It makes no accuracy or performance claim. It is development evidence only, not a training result, and it is not an automatic model promotion. PR35 remains selected.
