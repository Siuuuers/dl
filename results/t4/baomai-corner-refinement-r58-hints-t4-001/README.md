# Corner-refinement-r58-hints T4 results

Provenance for the returned initial-corner (`hint_mode=initial`) local-refiner pack. Facts below are transcribed from the returned files and the Colab run, not a claim of improvement. This records the archive. It makes no accuracy or promotion claim. It is development evidence only. The local refiner trained here is not a deployment model. PR35 remains the selected deployment model. Nothing is automatically promoted.

- Pinned launcher commit `0d8f7c6c8a48a4d63ce6641bbed841a663c172e9` (`notebooks/baomai_corner_refinement_hints_r58.ipynb`). `SOURCE_COMMIT` / `source` `5b1650b8fa3959a0917680305d7b499af783c5aa`. `DRIVER_SHA256` / `script_sha256` `80d52d827e3b0a63318ebb0a611356bd6534da7df1d9891a3f409f57076841b6` (`scripts/run_corner_refinement_r58.py`; the `run_corner_refinement_r58.py` member of the evidence ZIP matches that hash).
- Recorded flags from `summary.json`: `status` completed, `interpretation_allowed` true, `baseline_passed` true, `training` true, `automatic_promotion` false, `fresh_inference` true, `development_only` true, `hint_mode` initial, `schema` `baomai-local-refiner-r58-results-v1`. This trained the local refiner; it is not a deployment model.
- Hardware: A100 + High-RAM was requested; A100 was unavailable and the authorized T4 High-RAM runtime was kept. Hostname `23dbef4acf41`. Tesla T4, CUDA 13.0, device `cuda:0`, System RAM 51.0 GB. From `detector-runtime.json` and the training receipts: Python 3.13.16, torch 2.14.1+cu130, torchvision 0.29.1+cu130, ultralytics 8.4.182, opencv-python 5.0.0.93, numpy 2.4.6, Pillow 13.0.0, GPU Tesla T4, device `cuda:0`, CUDA 13.0.
- The notebook was run unchanged on a new Drive copy. No `BAOMAI_*` environment variables were set. Drive copy: https://colab.research.google.com/drive/1BN36M2mfgUs30R_2Sy745w7yiz95kS1H?authuser=1
- Baseline-equivalence passed (`passed` true, `exact_json_equal` true, maximum coordinate-component / endpoint-error / probability drift 0.0).
- Input archive is the PR137 candidate: archive SHA-256 `fc8c40357c83831d17f88b0bb5a5ecd2e0cb1a8128be697e37e80620e4981023`, checkpoint SHA-256 `222bf3c91a651183047a10eb1a50e16ae8fe999f15372afe005bfca2d0b157f9`.
- Preflight SHA-256 `5e9900c34dcfa02154d645c2e91f94ca1b1aa67923478b7a20728ab8a3f96341` (the `preflight.json` evidence member, also `training-receipt.json` `preflight_sha256`) matches the published hints bootstrap receipt.
- Outer upload ZIP `baomai-corner-refinement-r58-initial-upload.zip`, 129790926 bytes, SHA-256 `713dca5626c11de4528091df1af579c2de986a34e71b2ff6a0bc0b7381a85e70`. That wrapper is not committed here (GitHub's ordinary file-size limit is 100 MB). This pack contains the compact evidence ZIP, `return-manifest.json`, and the nine wrapped checkpoint parts extracted from it.
- Evidence ZIP `baomai-corner-refinement-r58-initial-results.zip`, 3118004 bytes, SHA-256 `f57e0a53a24a6159e4cb9b91613099146f05e01a483b02e304419a520e69202b`.
- Final refiner `refiner-final.pt`: 136332779 bytes, `model_sha256` `a389d44a04366529ebf8ec35b76f23467d02ee0cbe515f375b64fc3eb51df69c`. Initial-state SHA-256 `8978a47c2b71b3a5aa1f4752e322cf4d91c947afe263c75baa673f4c7bb6ddc8` (`model-smoke.json` `initial_model.state_sha256`).
- This pull request is based on current `main` `b4df76935178de5a122119aea48e85d2d661effc`. This results pack does not merge, close, or edit earlier PRs, including draft PR 151 (https://github.com/Siuuuers/dl/pull/151), results PR 152 (https://github.com/Siuuuers/dl/pull/152), results PR 150 (https://github.com/Siuuuers/dl/pull/150), results PR 149 (https://github.com/Siuuuers/dl/pull/149), draft PR 148 (https://github.com/Siuuuers/dl/pull/148), draft PR 147 (https://github.com/Siuuuers/dl/pull/147), results PR 146 (https://github.com/Siuuuers/dl/pull/146), results PR 144 (https://github.com/Siuuuers/dl/pull/144), results PR 143 (https://github.com/Siuuuers/dl/pull/143), PR 140 (https://github.com/Siuuuers/dl/pull/140), draft PR 145 (https://github.com/Siuuuers/dl/pull/145), draft PR 142 (https://github.com/Siuuuers/dl/pull/142), draft PR 139 (https://github.com/Siuuuers/dl/pull/139), and PR 137 (https://github.com/Siuuuers/dl/pull/137).

Recorded Cell 8 postclip counts, already checked equal to evidence `summary.json` detection as `fn` / unmatched `fp` / `duplicate_fp` and two-corner `within5`/`eligible`. These are transcribed only and are not a selection or promotion:

- frozen666 control 34 91 17 511/584 vs treatment 34 91 17 505/584
- historical663 control 30 90 17 511/584 vs treatment 30 90 17 505/584
- r38 control 15 84 14 566/583 vs treatment 15 84 14 558/583
- r50 control 15 84 13 566/583 vs treatment 15 84 13 558/583

Matching convention, transcribed from `pair-accounting.json` (Cell 8 matches): "Match all fixed detector boxes first, then account for offered pairs under those same assignments. Accepted subsets are not rematched."

frozen666 / r50 pair-accounting figures, transcribed from `pair-accounting.json` (Cell 8 matches):

- frozen666 control: confirmed_within5 511, known_pair_errors_over5 66, matched_incomplete_reference 51, offered_pairs 712, unmatched_offered_pairs 84
- frozen666 treatment: confirmed_within5 505, known_pair_errors_over5 72, matched_incomplete_reference 54, offered_pairs 722, unmatched_offered_pairs 91
- r50 control: confirmed_within5 566, known_pair_errors_over5 17, matched_incomplete_reference 51, offered_pairs 712, unmatched_offered_pairs 78
- r50 treatment: confirmed_within5 558, known_pair_errors_over5 25, matched_incomplete_reference 55, offered_pairs 722, unmatched_offered_pairs 84

How to reassemble the model from the wrapped parts (do not unwrap the parts in this repository). From `return-manifest.json`: read `refiner-final.pt.part00.zip` through `refiner-final.pt.part08.zip` in that listed order; verify each wrapper SHA-256 and size; extract and verify each listed member SHA-256 and size; concatenate the member bytes in that same order; verify the complete model is 136332779 bytes with SHA-256 `a389d44a04366529ebf8ec35b76f23467d02ee0cbe515f375b64fc3eb51df69c`.

Wrapper hashes transcribed from `return-manifest.json`:

- `refiner-final.pt.part00.zip`, 15599308 bytes, SHA-256 `09a57fb76067962cd293380f461ebf807bf4caaa9f635da4ccb234ad99d0cf9c`, member `refiner-final.pt.part00` 16777216 bytes, member SHA-256 `8c40cb2cf9d20577350afebdf53cbb52b06c30a2e84311e4a45bf5ddf74f19e6`
- `refiner-final.pt.part01.zip`, 15607348 bytes, SHA-256 `d70c7a29fc48766a06987068aed5d3f4e92fcd0d57f8103cbcfa310541cbe2cb`, member `refiner-final.pt.part01` 16777216 bytes, member SHA-256 `8fab4c777db43c98523415938525d0c0639ce92a451e90824fea9aa08a608dd5`
- `refiner-final.pt.part02.zip`, 15585520 bytes, SHA-256 `427904d18b90d028f38f84d7fbab6457514f28f08d8b12cfa5e11fcee60f4038`, member `refiner-final.pt.part02` 16777216 bytes, member SHA-256 `5c8159f4ca4f06f1a03b5fe8a9372c2e1accc26def4bf8548a3e487eb988845e`
- `refiner-final.pt.part03.zip`, 15589704 bytes, SHA-256 `ddce2c1dbd47b820f4f946845e5a872babe3942e53eca22911a8412b4e537cb0`, member `refiner-final.pt.part03` 16777216 bytes, member SHA-256 `81389f8fcafd0fcaffe8475dd2fc029a97639767453e9125be1895739f8cbb0e`
- `refiner-final.pt.part04.zip`, 15588141 bytes, SHA-256 `a2d8606efc0ad8cdc27b3706e1545e52af57cabc45cafcd3e734ad783c649300`, member `refiner-final.pt.part04` 16777216 bytes, member SHA-256 `c5bf69fb5439baab1d6ad3ef7c521eaa0579596f35410b0ab8ee89e50041589c`
- `refiner-final.pt.part05.zip`, 15585384 bytes, SHA-256 `6e837636375a5e5d00f2a0a8326747e73067393faf0e484640cd367dfaf5d50d`, member `refiner-final.pt.part05` 16777216 bytes, member SHA-256 `42f19c3363789784aab150a46d4ed79231cdd80c4d0eddaa868fd0a7946367a6`
- `refiner-final.pt.part06.zip`, 15589427 bytes, SHA-256 `f606589f9bf61bff74d955cb17bdb29a180c3cb53aa9ca687668510cc3435505`, member `refiner-final.pt.part06` 16777216 bytes, member SHA-256 `0a2b536cb64823b8e5702e62f3630acea33b54f3ba0b38877c0ae14018bc1286`
- `refiner-final.pt.part07.zip`, 15577876 bytes, SHA-256 `374d105f35a33aafc3d5cd240dcbf4fb97be8ae0b05851a726c568bd57c5f025`, member `refiner-final.pt.part07` 16777216 bytes, member SHA-256 `d59ddc3fb8468a65362ec85d6dc71f65c23a59b6a962a535dcaff59b5c3f0969`
- `refiner-final.pt.part08.zip`, 1945242 bytes, SHA-256 `ed0e0959fe083cd15a0943ac7f7fec2f9c8a2c3f7b29c7c7cb562f8d7b2c0eb9`, member `refiner-final.pt.part08` 2115051 bytes, member SHA-256 `e77b5d59158bcd30f81dcd043658c5cf136acf59e12dc71eebf56720cd556fc1`

All pack files are ordinary git `100644` blobs, not Git LFS, and the full outer upload ZIP is not committed.

This records the returned archive. It makes no accuracy or performance claim. It is development evidence only, not an automatic model promotion. PR35 remains the selected deployment model.
