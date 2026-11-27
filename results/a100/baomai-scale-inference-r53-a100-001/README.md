# Scale-inference-r53 A100 results

Provenance for the returned archive:

- Experiment: `scale-inference-r53` on the unchanged notebook from commit `5001d83439eaf31c7ecae378c27a0de847e65686`. This is a development inference comparison, not new training and not a promotion.
- Recorded flags from `summary.json`: `status` completed, `interpretation_allowed` true, `baseline_passed` true, `repeated_outputs_stable` true, `fresh_inference` true, `training` false, `automatic_promotion` false, `treatment_executed` true.
- `SOURCE_REVISION` / `source` `7b36e23dbbc24769f92a88cb93ae44acfc0406c5`, notebook-publish commit `5001d83439eaf31c7ecae378c27a0de847e65686`, `DRIVER_SHA256` / `script_sha256` `8c8a37c3dcaccde5c568e9df6f44651cdb05965ca5f5ed5b0f2978852387dcc1`, checkpoint `222bf3c91a651183047a10eb1a50e16ae8fe999f15372afe005bfca2d0b157f9`, input archive `fc8c40357c83831d17f88b0bb5a5ecd2e0cb1a8128be697e37e80620e4981023` (this is the input archive, 42395650 bytes, not the r53 output hash).
- `summary.json` interpretation: "Development comparison only. No benefit is established until this returned evidence is reviewed; no model, label, filter or threshold is promoted." Frozen 666 stays primary. PR35 remains the selected training comparison.
- Notebook: `notebooks/baomai_scale_inference_r53.ipynb` at commit `5001d83439eaf31c7ecae378c27a0de847e65686`, run unchanged.
- Drive copy: https://colab.research.google.com/drive/1sbHGW7UY_3Lg6SdnQCHX_-5IsweasDDI?authuser=1
- Runtime: A100 + High-RAM was requested, and this same pending request connected. Hostname `1840ee8bc271`. NVIDIA A100-SXM4-40GB, High-RAM 83.5 GB. From `runtime.json`: CUDA 13.0, cuDNN 92800, Python 3.13.16, torch 2.14.1+cu130, torchvision 0.29.1+cu130, ultralytics 8.4.182, device `cuda:0`. The runtime was released after the ZIP was saved.
- Settings from `summary.json` contract: experiment `scale`, control photometry `identity` with imgsz 1280, treatment photometry `identity` with imgsz 1600, conf 0.13, iou 0.7, matching_iou 0.5, endpoint_conf 0.5, max_det 300, nms true, batch 1.
- Archive: `baomai-scale-inference-r53-results.zip`, 2328146 bytes, SHA-256 `b1638b1eaf5e3877a77efb840c342bd1dfe7783ca5951041ed37b692df59abca`, 12 ordinary members. `ZipFile.testzip()` was clean. Committed as a single ordinary git `100644` blob, not Git LFS, not a split ZIP_STORED pack.
- This pull request is based on current `main` `b4df76935178de5a122119aea48e85d2d661effc`. This results pack does not merge, close, or edit earlier PRs, including results PR 149 (https://github.com/Siuuuers/dl/pull/149), draft PR 148 (https://github.com/Siuuuers/dl/pull/148), draft PR 147 (https://github.com/Siuuuers/dl/pull/147), results PR 146 (https://github.com/Siuuuers/dl/pull/146), results PR 144 (https://github.com/Siuuuers/dl/pull/144), results PR 143 (https://github.com/Siuuuers/dl/pull/143), PR 140 (https://github.com/Siuuuers/dl/pull/140), draft PR 145 (https://github.com/Siuuuers/dl/pull/145), draft PR 142 (https://github.com/Siuuuers/dl/pull/142), draft PR 139 (https://github.com/Siuuuers/dl/pull/139), and PR 137 (https://github.com/Siuuuers/dl/pull/137).

Recorded counts, transcribed from `summary.json` postclip detection as printed, not as a selection or promotion. Cell 8 postclip counts match these figures:

- frozen666 control postclip tp632/fn34/fp91 duplicates 17 two-corner rows within5px 511/584
- frozen666 treatment postclip tp633/fn33/fp82 duplicates 20 two-corner rows within5px 503/584
- historical663 control postclip tp633/fn30/fp90 duplicates 17 two-corner rows within5px 511/584
- historical663 treatment postclip tp634/fn29/fp81 duplicates 20 two-corner rows within5px 503/584
- r38 control postclip tp639/fn15/fp84 duplicates 14 two-corner rows within5px 566/583
- r38 treatment postclip tp639/fn15/fp76 duplicates 18 two-corner rows within5px 557/583
- r50 control postclip tp639/fn15/fp84 duplicates 13 two-corner rows within5px 566/583
- r50 treatment postclip tp638/fn16/fp77 duplicates 17 two-corner rows within5px 557/583

Independently measured archive members (SHA-256 and size), hashed from the extracted zip members:

- `baseline-equivalence.json`, 517 bytes, SHA-256 `a4d081beb4e870dff5b798ade4e00793bb5b1975baa2bee953580f54138216f0`
- `cases.json`, 6277739 bytes, SHA-256 `d13191aa79efbc5e3d71480e213180e7812cc3590dc5d8f37d8b764f41129397`
- `input-audit.json`, 44712 bytes, SHA-256 `86e11ccff6c085b87c600723abf9015f993098b0120df3e04bd4b90f5296bef1`
- `inventory.json`, 1451 bytes, SHA-256 `92dc6033482904561a7194a6cab954f3a9c426b7179ec83ee2942c1757e235f7`
- `predictions.json`, 1659638 bytes, SHA-256 `6d818ffbcf0a5e725d86fa880cc0fb20bc4f017359d911195df16ca5a11b8b67`
- `preflight.json`, 160768 bytes, SHA-256 `083c2a7329d5dd44c9acf694c82bdef8de1319b305f8ecf9fafebd2e9284f6de`
- `repeatability.json`, 3243 bytes, SHA-256 `f3933737dbff14a1ea658ee4753092b97248348fbda482a3729b43d15d28e356`
- `repeated-predictions.json`, 5383326 bytes, SHA-256 `527e56849154a3ede9106e7adbe0c801d62a72df7cd896e8fd1d1c8843e8b5f1`
- `run_inference_ablation_r53.py`, 36334 bytes, SHA-256 `8c8a37c3dcaccde5c568e9df6f44651cdb05965ca5f5ed5b0f2978852387dcc1`
- `runtime.json`, 4583 bytes, SHA-256 `a5af0ed9381769741a62d89654db3e93005d0480fd9c1af684b7cefa5fc00527`
- `summary.json`, 3422337 bytes, SHA-256 `2e269a8b620c7ac13b36fa83fa546f545d7bc071ea30fa342b25cf1407ecdf1f`
- `timing.json`, 140464 bytes, SHA-256 `f5b1617525abfae73b6f49e01c6661c93988d62e7d40aaae1196ae72354fbd9d`

This records the returned archive. It makes no accuracy or performance claim. It is development evidence only, not a training result, and it is not an automatic model promotion. PR35 remains selected.
