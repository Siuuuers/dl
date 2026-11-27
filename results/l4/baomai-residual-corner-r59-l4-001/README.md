# Residual-corner-r59 L4 results

Provenance for the returned residual-corner-r59 pack. Facts below are transcribed from the returned files and the Colab run, not a claim of improvement. This records the archive. It makes no accuracy or promotion claim. It is development evidence only. The local residual corrector trained here is not a deployment model. PR35 remains the selected deployment model. Nothing is automatically promoted.

- Pinned launcher commit `a48e286342bd9c0d45a459831a86ef7866347fcb` (`notebooks/baomai_residual_corner_r59.ipynb`). `SOURCE_COMMIT` / `source` `06e55d9c1b2557014aff7ff0d68b5096890fe546`. `DRIVER_SHA256` / `script_sha256` `54cdf49281abdc5409c9a8828d1a3bb00c5f9da8610e8767645b69315a78e5d3` (`scripts/run_residual_corner_r59.py`; the `run_residual_corner_r59.py` member of the evidence ZIP matches that hash).
- Recorded flags from `summary.json`: `status` completed, `interpretation_allowed` true, `baseline_passed` true, `training` true, `automatic_promotion` false, `fresh_inference` true, `development_only` true, `schema` `baomai-residual-corner-r59-results-v1`. This trained the local residual corrector; it is not a deployment model.
- Hardware: A100 High-RAM was requested but unavailable; the authorized L4 High-RAM runtime was kept. Hostname `5a9832a687a3`. NVIDIA L4, CUDA 13.0, device `cuda:0`, System RAM 53.0 GB. From `detector-runtime.json`: Python 3.13.16, torch 2.14.1+cu130, torchvision 0.29.1+cu130, ultralytics 8.4.182, opencv-python 5.0.0.93, numpy 2.4.6, Pillow 13.0.0, GPU NVIDIA L4, device `cuda:0`, CUDA 13.0.
- The notebook was run unchanged on a new Drive copy. No `BAOMAI_*` environment variables were set. Run all started after hardware proof.
- Cell 6: pretrained CPU check passed (`model-smoke.json` `status` passed, `device` cpu). Exact zero initial offsets `True` (`initial_offsets_exactly_zero`). Initial-state SHA-256 `2982de6f862dbbb1c40819b0155475cbdbe842012eaa50851ddf5f200875a959` (`model-smoke.json` `initial_model.state_sha256`; matches published-bootstrap). `zero-output-gate.json` `passed` true.
- Baseline-equivalence passed (`passed` true, `exact_json_equal` false). Recorded maximum drift: coordinate-component 0.0001220703125 px, endpoint-error 6.766371575572805e-05 px, probability 1.1622905731201172e-06, against tolerances 0.05 px / 0.075 px / 1e-05.
- Input archive is the PR137 candidate: archive SHA-256 `fc8c40357c83831d17f88b0bb5a5ecd2e0cb1a8128be697e37e80620e4981023`, checkpoint SHA-256 `222bf3c91a651183047a10eb1a50e16ae8fe999f15372afe005bfca2d0b157f9`.
- Preflight SHA-256 `28e87eed7640df59907dc44e5342c09e94c2344eab3e59c17a352709351c23f9` (the `preflight.json` evidence member, also `training-receipt.json` `preflight_sha256`).
- Outer upload ZIP `baomai-residual-corner-r59-upload.zip`, 91693769 bytes, SHA-256 `d02ebbe236b26b668283cd312455ec8d5df71e7b958a5c441ecc63cb842699c1`. That wrapper is not committed here. This pack follows the notebook return layout: the compact evidence ZIP, `return-manifest.json`, and the six wrapped checkpoint parts extracted from the outer ZIP.
- Evidence ZIP `baomai-residual-corner-r59-results.zip`, 3971513 bytes, SHA-256 `dbf5057d765c2f38bc2a28fcf98104c51c49a09afa5a83a2268495784e08b79c`.
- Final residual corrector `correction-final.pt`: 94404981 bytes, `model_sha256` `19c7e95aca0019c3b1eb60de4c36c6ecd30d35fdcaaa103606cdd7d553f7006a`. Training receipt: 30 epochs, selection fixed final epoch, `expected_updates` 3600, updates 3600.
- This pull request is based on current `main` `b4df76935178de5a122119aea48e85d2d661effc`. This results pack does not merge, close, or edit earlier PRs, including results PR 154 (https://github.com/Siuuuers/dl/pull/154), results PR 153 (https://github.com/Siuuuers/dl/pull/153), results PR 152 (https://github.com/Siuuuers/dl/pull/152), draft PR 151 (https://github.com/Siuuuers/dl/pull/151), results PR 150 (https://github.com/Siuuuers/dl/pull/150), results PR 149 (https://github.com/Siuuuers/dl/pull/149), draft PR 148 (https://github.com/Siuuuers/dl/pull/148), draft PR 147 (https://github.com/Siuuuers/dl/pull/147), results PR 146 (https://github.com/Siuuuers/dl/pull/146), draft PR 145 (https://github.com/Siuuuers/dl/pull/145), results PR 144 (https://github.com/Siuuuers/dl/pull/144), results PR 143 (https://github.com/Siuuuers/dl/pull/143), draft PR 142 (https://github.com/Siuuuers/dl/pull/142), PR 140 (https://github.com/Siuuuers/dl/pull/140), draft PR 139 (https://github.com/Siuuuers/dl/pull/139), and PR 137 (https://github.com/Siuuuers/dl/pull/137).

Recorded Cell 8 postclip counts from the common-confidence table (`summary.json` `reports_at_common_confidence_025`; already checked equal). Detection as `fn` / unmatched `fp` / `duplicate_fp` and two-corner `within5`/`eligible`. Cell 8 used this common-confidence table, not the native-threshold `reports` object. These are transcribed only and are not a selection or promotion:

- frozen666 control 34 55 9 511/584 vs treatment 34 55 9 522/584
- historical663 control 31 55 9 511/584 vs treatment 31 55 9 522/584
- r38 control 18 51 7 566/583 vs treatment 18 51 7 564/583
- r50 control 18 51 5 566/583 vs treatment 18 51 5 564/583

Matching convention, transcribed from both `pair-accounting-common025.json` and `pair-accounting.json` (the wording is the same in each file): "Match all fixed detector boxes first, then account for offered pairs under those same assignments. Accepted subsets are not rematched."

frozen666 / r50 pair-accounting figures from `pair-accounting-common025.json` (the common-confidence table; this is the table that belongs with the Cell 8 counts above):

- frozen666 control: confirmed_within5 511, known_pair_errors_over5 66, matched_incomplete_reference 51, offered_pairs 678, unmatched_offered_pairs 50
- frozen666 treatment: confirmed_within5 522, known_pair_errors_over5 55, matched_incomplete_reference 51, offered_pairs 678, unmatched_offered_pairs 50
- r50 control: confirmed_within5 566, known_pair_errors_over5 17, matched_incomplete_reference 48, offered_pairs 678, unmatched_offered_pairs 47
- r50 treatment: confirmed_within5 564, known_pair_errors_over5 19, matched_incomplete_reference 48, offered_pairs 678, unmatched_offered_pairs 47

frozen666 / r50 pair-accounting figures from `pair-accounting.json` (the other pair-accounting table in the evidence ZIP; not the Cell 8 common-confidence table):

- frozen666 control: confirmed_within5 511, known_pair_errors_over5 66, matched_incomplete_reference 51, offered_pairs 712, unmatched_offered_pairs 84
- frozen666 treatment: confirmed_within5 522, known_pair_errors_over5 55, matched_incomplete_reference 51, offered_pairs 712, unmatched_offered_pairs 84
- r50 control: confirmed_within5 566, known_pair_errors_over5 17, matched_incomplete_reference 51, offered_pairs 712, unmatched_offered_pairs 78
- r50 treatment: confirmed_within5 564, known_pair_errors_over5 19, matched_incomplete_reference 51, offered_pairs 712, unmatched_offered_pairs 78

The two tables are transcribed separately. This note does not interpret a winner.

How to reassemble the model from the wrapped parts (do not unwrap the parts in this repository). From `return-manifest.json`: read `correction-final.pt.part00.zip` through `correction-final.pt.part05.zip` in that listed order; verify each wrapper SHA-256 and size; extract and verify each listed member SHA-256 and size; concatenate the member bytes in that same order; verify the complete model is 94404981 bytes with SHA-256 `19c7e95aca0019c3b1eb60de4c36c6ecd30d35fdcaaa103606cdd7d553f7006a`.

Wrapper hashes transcribed from `return-manifest.json`:

- `correction-final.pt.part00.zip`, 15598552 bytes, SHA-256 `6dc480c2c7aac50826b16f3bc72ea658ade298392e25d3b6aaecd8103996bc9f`, member `correction-final.pt.part00` 16777216 bytes, member SHA-256 `ec8eec0a8b1a499d404b0e0c08009d550c3f8c29999607d689a3c19f9a4601a8`
- `correction-final.pt.part01.zip`, 15606222 bytes, SHA-256 `3ca1ff3561cc0e7ad151cdf69b4ba9e109e51058dbf59c9c90ad760b4f91f690`, member `correction-final.pt.part01` 16777216 bytes, member SHA-256 `2ddbaab2cfe8e24500feba16fbaf41a05caaa50aedf996ca649f2d0ed0736a50`
- `correction-final.pt.part02.zip`, 15585843 bytes, SHA-256 `7a51d01e55ba24276a25aa44e4414e48a042e64641238f1acbeba34277073da0`, member `correction-final.pt.part02` 16777216 bytes, member SHA-256 `f6583c710058b6da190a77f60f93ad588842a966ba212e3e6981dda44860c4db`
- `correction-final.pt.part03.zip`, 15589261 bytes, SHA-256 `c6b04387b5c544d6cdcd646d967d596edb573f5fcd373590fbd6d3ccc3ec0ef5`, member `correction-final.pt.part03` 16777216 bytes, member SHA-256 `03359a6336633d7a752845e73fe3b06bbff68b8aea9937f6483e9d060ef4fdf4`
- `correction-final.pt.part04.zip`, 15587380 bytes, SHA-256 `49c8b288e962067bb70bf6bcf73cd791b1e9f0db2fdd845749194afcf034099d`, member `correction-final.pt.part04` 16777216 bytes, member SHA-256 `e28c30d97dd13b1c7e88b6df24bb35adbd2a0e354662aa95c0789883be0f5c64`
- `correction-final.pt.part05.zip`, 9751388 bytes, SHA-256 `a1e09a9a404c54cddb6d949bcc085cc04746335b2350e7a18ed8d1d78929f07f`, member `correction-final.pt.part05` 10518901 bytes, member SHA-256 `c35f2ae23aaebb1460b7c37293282a9b83b12c1eb82ce2a1eb2286811510f1f8`

All pack files are ordinary git `100644` blobs, not Git LFS, and the full outer upload ZIP is not committed.

This records the returned archive. It makes no accuracy or performance claim. It is development evidence only, not an automatic model promotion. PR35 remains the selected deployment model.
