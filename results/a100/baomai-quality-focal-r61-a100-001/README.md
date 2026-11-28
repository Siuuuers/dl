# Quality-focal-r61 A100 results

Provenance for the returned quality-focal-r61 pack. Facts below are transcribed from the returned files and the Colab run, not a claim of improvement. This records the archive. It makes no accuracy or promotion claim. It is development evidence only. PR35 remains the selected deployment model. Nothing is automatically promoted.

- Official run of the PR 156 fixed notebook, unchanged Run all, with no `BAOMAI_*` environment variables set. Drive copy https://colab.research.google.com/drive/1-KKV24z8hoA3LNdKNWMKKH0qUSt9GUJx?authuser=1, account gloriawingyin@gmail.com.
- `SOURCE_COMMIT` / `source` `acaf5d4b4043583fdd81f5ea848b3a6a68ada728`. `DRIVER_SHA256` `6e740c65f23e3eed3537eb00966cdea9202eccd7303b3d7b7259f181c368ebf5` (`scripts/run_quality_focal_r61.py`; the `source/scripts/run_quality_focal_r61.py` member of the evidence ZIP matches that hash). Notebook pin `587514e557db13be287a5483eaf952dfd5e14761`.
- Recorded flags: `staging.json` / evaluation schema `baomai-quality-focal-r61-v1` and `baomai-quality-focal-evaluation-r61-v1`; `treatment.automatic_promotion` false; `measurement.protocol.development_only` true; `measurement.protocol.box_confidence_strictly_greater` 0.25. `cpu-loader-preflight.json` `status` passed (device cpu). `loss-guard.json` `status` passed (`scalar_gradient_guard.status` passed). Both arm receipts `status` complete.
- Hardware: NVIDIA A100-SXM4-40GB High-RAM was requested and connected. Hostname `f04a05887725`. Driver 580.82.07, CUDA 13.0, System RAM 83.5 GB. From `evaluation/summary.json` `runtime` and both arm receipts: Python 3.13.16, torch 2.14.1+cu130, torchvision 0.29.1+cu130, ultralytics 8.4.182, opencv-python 5.0.0.93, numpy 2.4.6, Pillow 13.0.0, GPU NVIDIA A100-SXM4-40GB, CUDA 13.0, cuDNN 92800. Connected about 4:28–4:48 PM HKT on 28 Nov 2026. Cell 6 about 14 minutes; Cell 8 about 1 minute. The runtime has since been released; Manage Sessions was empty after release.
- Training, transcribed from both arm receipts and `evaluation/summary.json` budgets: 142 photographs / 3999 fronts, identical annotations (`actual_loader.train.counts` [142, 3999, 7259, 739, 13], `exact_image_and_label_bytes` true, same `entries_sha256` on both arms). Budget 30 epochs / 1080 batches / 80 steps / 4260 presentations per arm. reference complete, epochs 30, updates 80, `focal_beta` 0; candidate complete, epochs 30, updates 80, `focal_beta` 2. `staging_sha256` `7914e2e622b76b4947067b8b8abeee8e01efdd4c76c013e8c6850fa93be69c45`. `initial_state_sha256` `5268194c14264aed09912980a0f9678c4705836c8cf864c5db9d28953fc00011`. Evaluate: 120 forward calls + 3 warmup.
- Outer upload ZIP `baomai-quality-focal-r61-upload.zip`, 47153085 bytes, SHA-256 `2096451712ea841c7c4da002a7ef147f6d5eaf1c7d910626cca3d57b2dd12dc1`. That wrapper is not committed here. This pack follows the notebook return layout: the compact evidence ZIP, `return-manifest.json`, and the wrapped checkpoint parts extracted from the outer ZIP. Schema `paired-model-return-r61-v1`, `status` completed.
- Evidence ZIP `baomai-quality-focal-r61-results.zip`, 7528011 bytes, SHA-256 `cf7b7b71349b9412bd5272001f5071afcf8d3b356423deb6c233cb1e6c326c4a`.
- Complete models, transcribed from `return-manifest.json` and verified by concatenating listed members: `reference-last.pt` 21704750 bytes, SHA-256 `15d7c97d5c58f16e40462807231c7bb01a5ee9a719225cb756493e1b41714d67`; `candidate-last.pt` 21704750 bytes, SHA-256 `8760602df814f47685dde25a2e389a554f623a7280473c90d5e94a0f30fe519f`.
- This pull request is based on current `main` `b4df76935178de5a122119aea48e85d2d661effc`. This results pack does not merge, close, or edit earlier PRs, including source PR 156 (https://github.com/Siuuuers/dl/pull/156), results PR 155 (https://github.com/Siuuuers/dl/pull/155), results PR 154 (https://github.com/Siuuuers/dl/pull/154), results PR 153 (https://github.com/Siuuuers/dl/pull/153), results PR 152 (https://github.com/Siuuuers/dl/pull/152), draft PR 151 (https://github.com/Siuuuers/dl/pull/151), results PR 150 (https://github.com/Siuuuers/dl/pull/150), results PR 149 (https://github.com/Siuuuers/dl/pull/149), draft PR 148 (https://github.com/Siuuuers/dl/pull/148), draft PR 147 (https://github.com/Siuuuers/dl/pull/147), results PR 146 (https://github.com/Siuuuers/dl/pull/146), draft PR 145 (https://github.com/Siuuuers/dl/pull/145), results PR 144 (https://github.com/Siuuuers/dl/pull/144), results PR 143 (https://github.com/Siuuuers/dl/pull/143), draft PR 142 (https://github.com/Siuuuers/dl/pull/142), PR 140 (https://github.com/Siuuuers/dl/pull/140), draft PR 139 (https://github.com/Siuuuers/dl/pull/139), and PR 137 (https://github.com/Siuuuers/dl/pull/137).

Fixed-threshold scores (confidence > 0.25), transcribed from `evaluation/summary.json` `measurement` (`detection.fn` / `detection.fp` as misses / extras, and `pairs`). These are transcribed only and are not a selection or promotion:

- frozen666, supplied original: misses 26, extras 41; eligible 584, returned 582, missed_box 2; within5 534, within10 580, over5 48
- frozen666, reference two_view: misses 34, extras 53; eligible 584, returned 577, missed_box 7; within5 522, within10 574, over5 55
- frozen666, candidate two_view: misses 32, extras 91; eligible 584, returned 577, missed_box 7; within5 525, within10 572, over5 52
- r50, supplied original: misses 34, extras 61; eligible 583, returned 580, missed_box 3; within5 555, within10 577, over5 25
- r50, reference two_view: misses 18, extras 49; eligible 583, returned 583, missed_box 0; within5 568, within10 583, over5 15
- r50, candidate two_view: misses 12, extras 83; eligible 583, returned 583, missed_box 0; within5 570, within10 582, over5 13
- supplied664, supplied original: misses 39, extras 56; eligible 584, returned 570, missed_box 14; within5 521, within10 568, over5 49
- supplied664, reference two_view: misses 46, extras 67; eligible 584, returned 568, missed_box 16; within5 511, within10 565, over5 57
- supplied664, candidate two_view: misses 44, extras 105; eligible 584, returned 568, missed_box 16; within5 514, within10 563, over5 54

AP50 (confidence > 0.001, 101 recall points), transcribed from `evaluation/summary.json` `detection_ranking`:

- r50 supplied 0.9586496689180926 / reference 0.9838649186792631 / candidate 0.9839533627248876
- supplied664 supplied 0.9457373004177758 / reference 0.9249216469949534 / candidate 0.9279691793895595
- frozen666 supplied 0.9748923608261791 / reference 0.9543564579671548 / candidate 0.9578883221736856

How to reassemble the models from the wrapped parts (do not unwrap the parts in this repository). From `return-manifest.json`: for each of `reference-last.pt` and `candidate-last.pt`, read the listed `part00.zip` then `part01.zip` in that order; verify each wrapper SHA-256 and size; extract and verify each listed member SHA-256 and size; concatenate the member bytes in that same order; verify each complete model is 21704750 bytes with the SHA-256 below.

Wrapper hashes transcribed from `return-manifest.json`:

- `reference-last.pt.part00.zip`, 15374799 bytes, SHA-256 `b587bfa756832eb9d4f1cdea7cd0657737b7bdad3dfa7095e5ddbe89a2c85ea0`, member `reference-last.pt.part00` 16777216 bytes, member SHA-256 `5497e7f20d8dd07b516c3b2c47fa1d5527f95f53eba264d20178e7a74b2a76c4`
- `reference-last.pt.part01.zip`, 4435788 bytes, SHA-256 `680516e8d11b32e57786f7d602fe8888e84d0050a2368077b5cb4ca04e804bd2`, member `reference-last.pt.part01` 4927534 bytes, member SHA-256 `ac9db9450a269b46b466dc1f94e72427c9c6d2fef767e6b4fc249f65531c6818`
- `candidate-last.pt.part00.zip`, 15375190 bytes, SHA-256 `fa01fc6a7e20e89fa2bc9e86d9cf9e16c8ee400475a9527b85707ff18d782311`, member `candidate-last.pt.part00` 16777216 bytes, member SHA-256 `6dbb610201745e55fa346c1c13ff1fd231210cd8e23f7e8bc6d7a1f50c733f17`
- `candidate-last.pt.part01.zip`, 4436189 bytes, SHA-256 `2dfa01b0be4ce21568fedd8dc4b186e504cc6f68bbde58f8c01b4bc746cf8b9a`, member `candidate-last.pt.part01` 4927534 bytes, member SHA-256 `e7f9147d23e41de6cc7e6c391b2bf882a62e91faa5e5ef20063da416a1c82418`

Complete model hashes: `reference-last.pt` SHA-256 `15d7c97d5c58f16e40462807231c7bb01a5ee9a719225cb756493e1b41714d67`; `candidate-last.pt` SHA-256 `8760602df814f47685dde25a2e389a554f623a7280473c90d5e94a0f30fe519f`.

All pack files are ordinary git `100644` blobs, not Git LFS, and the full outer upload ZIP is not committed.

This records the returned archive. It makes no accuracy or performance claim. It is development evidence only, not an automatic model promotion. PR35 remains the selected deployment model.
