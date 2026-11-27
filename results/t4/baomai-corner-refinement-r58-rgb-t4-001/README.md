# Corner-refinement-r58-rgb T4 results

Provenance for the returned RGB (`hint_mode=none`) local-refiner pack. Facts below are transcribed from the returned files and the Colab run, not a claim of improvement. This records the archive. It makes no accuracy or promotion claim. It is development evidence only. The local refiner trained here is not a deployment model. PR35 remains the selected deployment model. Nothing is automatically promoted. The paired hints results are in PR 153 (https://github.com/Siuuuers/dl/pull/153). This note does not compare accuracy or recommend a winner.

- Pinned launcher commit `0d8f7c6c8a48a4d63ce6641bbed841a663c172e9` (`notebooks/baomai_corner_refinement_rgb_r58.ipynb`). `SOURCE_COMMIT` / `source` `5b1650b8fa3959a0917680305d7b499af783c5aa`. `DRIVER_SHA256` / `script_sha256` `80d52d827e3b0a63318ebb0a611356bd6534da7df1d9891a3f409f57076841b6` (`scripts/run_corner_refinement_r58.py`; the `run_corner_refinement_r58.py` member of the evidence ZIP matches that hash).
- Recorded flags from `summary.json`: `status` completed, `interpretation_allowed` true, `baseline_passed` true, `training` true, `automatic_promotion` false, `fresh_inference` true, `development_only` true, `hint_mode` none, `schema` `baomai-local-refiner-r58-results-v1`. This trained the local refiner; it is not a deployment model.
- Hardware: T4 High-RAM was requested to match the completed hints arm. Hostname `7efd7dc77cbe`. Tesla T4, CUDA 13.0, device `cuda:0`, System RAM 51.0 GB. Receipts in `detector-runtime.json`: Python 3.13.16, torch 2.14.1+cu130, torchvision 0.29.1+cu130, ultralytics 8.4.182, opencv-python 5.0.0.93, numpy 2.4.6, Pillow 13.0.0, GPU Tesla T4, device `cuda:0`, CUDA 13.0. This matches the hints arm's recorded Torch/TorchVision and GPU type.
- The notebook was run unchanged on a new Drive copy. No `BAOMAI_*` environment variables were set. Run all started after hardware proof. Cell 6 finished with `Real CPU mechanics check: passed` and initial-state SHA-256 `8978a47c2b71b3a5aa1f4752e322cf4d91c947afe263c75baa673f4c7bb6ddc8` (`model-smoke.json` `initial_model.state_sha256`; the same initial-state hash as the hints arm). Drive copy: https://colab.research.google.com/drive/1q_qxznv4WKhIuFCE6IkVlb7KsYSA7YhB?authuser=1
- Baseline-equivalence passed (`passed` true, `exact_json_equal` true, maximum coordinate-component / endpoint-error / probability drift 0.0).
- Input archive is the PR137 candidate: archive SHA-256 `fc8c40357c83831d17f88b0bb5a5ecd2e0cb1a8128be697e37e80620e4981023`, checkpoint SHA-256 `222bf3c91a651183047a10eb1a50e16ae8fe999f15372afe005bfca2d0b157f9`.
- Preflight SHA-256 `fa3435ecfa740b89aba7afa5812acaac7f4c2374386493292ea5f8dc05b1ebba` (the `preflight.json` evidence member, also `training-receipt.json` `preflight_sha256`).
- Outer upload ZIP `baomai-corner-refinement-r58-none-upload.zip`, 129765273 bytes, SHA-256 `84082233c55141aefe9bb5b26b5ac4677756c8a3ff54d8553cc725d6d37ebe6a`. That wrapper is not committed here (GitHub's ordinary file-size limit is 100 MB). This pack contains the compact evidence ZIP, `return-manifest.json`, and the nine wrapped checkpoint parts extracted from it.
- Evidence ZIP `baomai-corner-refinement-r58-none-results.zip`, 3117954 bytes, SHA-256 `1f66eae2410c9e9d866f924956e1dc94916c86dd115ec2e43a0c27f5c3fa0b60`.
- Final refiner `refiner-final.pt`: 136332779 bytes, `model_sha256` `6dcc57f1d9b807e472c62557e8a85d0dfd456e418957737323fea52b0c00fbb6`. Training receipt: 30 epochs, selection fixed final epoch, updates 3600.
- This pull request is based on current `main` `b4df76935178de5a122119aea48e85d2d661effc`. This results pack does not merge, close, or edit earlier PRs, including results PR 153 (https://github.com/Siuuuers/dl/pull/153), draft PR 151 (https://github.com/Siuuuers/dl/pull/151), results PR 152 (https://github.com/Siuuuers/dl/pull/152), results PR 150 (https://github.com/Siuuuers/dl/pull/150), results PR 149 (https://github.com/Siuuuers/dl/pull/149), draft PR 148 (https://github.com/Siuuuers/dl/pull/148), draft PR 147 (https://github.com/Siuuuers/dl/pull/147), results PR 146 (https://github.com/Siuuuers/dl/pull/146), results PR 144 (https://github.com/Siuuuers/dl/pull/144), results PR 143 (https://github.com/Siuuuers/dl/pull/143), PR 140 (https://github.com/Siuuuers/dl/pull/140), draft PR 145 (https://github.com/Siuuuers/dl/pull/145), draft PR 142 (https://github.com/Siuuuers/dl/pull/142), draft PR 139 (https://github.com/Siuuuers/dl/pull/139), and PR 137 (https://github.com/Siuuuers/dl/pull/137).

Recorded Cell 8 postclip counts, already checked equal to evidence `summary.json` detection as `fn` / unmatched `fp` / `duplicate_fp` and two-corner `within5`/`eligible`. These are transcribed only and are not a selection or promotion:

- frozen666 control 34 91 17 511/584 vs treatment 34 91 17 501/584
- historical663 control 30 90 17 511/584 vs treatment 30 90 17 501/584
- r38 control 15 84 14 566/583 vs treatment 15 84 14 547/583
- r50 control 15 84 13 566/583 vs treatment 15 84 13 547/583

Matching convention, transcribed from `pair-accounting.json` (Cell 8 matches): "Match all fixed detector boxes first, then account for offered pairs under those same assignments. Accepted subsets are not rematched."

frozen666 / r50 pair-accounting figures, transcribed from `pair-accounting.json` (Cell 8 matches):

- frozen666 control: confirmed_within5 511, known_pair_errors_over5 66, matched_incomplete_reference 51, offered_pairs 712, unmatched_offered_pairs 84
- frozen666 treatment: confirmed_within5 501, known_pair_errors_over5 76, matched_incomplete_reference 54, offered_pairs 722, unmatched_offered_pairs 91
- r50 control: confirmed_within5 566, known_pair_errors_over5 17, matched_incomplete_reference 51, offered_pairs 712, unmatched_offered_pairs 78
- r50 treatment: confirmed_within5 547, known_pair_errors_over5 36, matched_incomplete_reference 55, offered_pairs 722, unmatched_offered_pairs 84

How to reassemble the model from the wrapped parts (do not unwrap the parts in this repository). From `return-manifest.json`: read `refiner-final.pt.part00.zip` through `refiner-final.pt.part08.zip` in that listed order; verify each wrapper SHA-256 and size; extract and verify each listed member SHA-256 and size; concatenate the member bytes in that same order; verify the complete model is 136332779 bytes with SHA-256 `6dcc57f1d9b807e472c62557e8a85d0dfd456e418957737323fea52b0c00fbb6`.

Wrapper hashes transcribed from `return-manifest.json`:

- `refiner-final.pt.part00.zip`, 15576436 bytes, SHA-256 `bff0fbdd336bdc752769926e38fb0766ef96ff649c76d9249629e9b8c1ee1318`, member `refiner-final.pt.part00` 16777216 bytes, member SHA-256 `746f0527bdf1cdb72b803e5c9a11a37add3d5387d0b9d1829baf3c417f09a612`
- `refiner-final.pt.part01.zip`, 15607345 bytes, SHA-256 `ccdb564141b6506dd82bafb596cf7d60339d540dad2e075fa61003025599f020`, member `refiner-final.pt.part01` 16777216 bytes, member SHA-256 `fda208dbc318022b4d04b761bce528ca85f69e22744aa4f67bc957304f536948`
- `refiner-final.pt.part02.zip`, 15585677 bytes, SHA-256 `b9ab1d660790ba290316ec5d8e6308f02f8ca1736388c12c0f76a36277ae6ec5`, member `refiner-final.pt.part02` 16777216 bytes, member SHA-256 `b99cb45d97bafd3e985567fdcf1b6b9fe801ff12b40484ff72cdbde136d374e0`
- `refiner-final.pt.part03.zip`, 15589395 bytes, SHA-256 `ba2afa918905acad1c11edea890a04c474ae00abd9fc09fbc2af28bfcdd5533b`, member `refiner-final.pt.part03` 16777216 bytes, member SHA-256 `9d9bc777bde34bb86777ed4f4d5f424c81e1e95f3a97385fed2b57d4676b33ec`
- `refiner-final.pt.part04.zip`, 15587908 bytes, SHA-256 `fefa68f201dda1b15bc9d3a3fe96af13d97064981cddec77d804fc56a0113217`, member `refiner-final.pt.part04` 16777216 bytes, member SHA-256 `8e7ff9b18fab36d1eb66689db88ed407c8080026b72f217c437487f82d2b72b2`
- `refiner-final.pt.part05.zip`, 15584338 bytes, SHA-256 `3a9c0f58088748c81766908c150c337e2b2356e5a895b8f088512dcc9b8fea67`, member `refiner-final.pt.part05` 16777216 bytes, member SHA-256 `e8ff1c236943be4e9b45cae43271979e77180566922e5ca7d857d798bc4d3236`
- `refiner-final.pt.part06.zip`, 15588956 bytes, SHA-256 `cd15be3b805171bff522569856ee84e134f3ddf0c7dbec6a7fcfdee5f27fa446`, member `refiner-final.pt.part06` 16777216 bytes, member SHA-256 `4c71db92322daf7f971989c968e305034a5021b618d690e0e0d03b88a444763b`
- `refiner-final.pt.part07.zip`, 15577022 bytes, SHA-256 `8efa117722affa1f714a962c54ba7e49f39c1590d70ee680ce5acaaf085dd914`, member `refiner-final.pt.part07` 16777216 bytes, member SHA-256 `a2d4ee2799d9ff81b89b86ce1f1645fe56ac3a29f57a659926e496afc04ca847`
- `refiner-final.pt.part08.zip`, 1945279 bytes, SHA-256 `00abaf37acd13c149e325914c4b01eca3656b380d61b639acd9d9f5ff98cd3cb`, member `refiner-final.pt.part08` 2115051 bytes, member SHA-256 `301de9374f10d625e2220adaba2eab3023d61a6066d1b174782800147bc63f95`

All pack files are ordinary git `100644` blobs, not Git LFS, and the full outer upload ZIP is not committed.

This records the returned archive. It makes no accuracy or performance claim. It is development evidence only, not an automatic model promotion. PR35 remains the selected deployment model.
