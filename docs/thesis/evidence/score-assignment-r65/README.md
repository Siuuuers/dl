# Completed frozen score and assignment diagnostic

All four actual cells of `notebooks/baomai_score_assignment_r65.ipynb` completed unchanged in one local IPython namespace. The notebook fetched a fresh, clean checkout of published source `7528f2f3980b7abaef0f2644c0a98d5163440f11`; its driver SHA-256 is `340db638b96c21bbac4c1d81ed058a7d9541c6933f21099f8fad114eda41ee15`.

## Verified scope

The complete pass used the same 142 unchanged training photographs and 3,999 annotated fronts for the two final checkpoints from the paired classification-loss experiment. It performed 284 forward calls and recorded both native prediction heads and both native-overlap and box-only-overlap assignment variants. The independent execution checks verified 568 image/head records, 31,992 unique target records, 24 summary groups, and source target strata of 137 fronts with no known corners, 465 with one and 3,397 with two. The separate three-photo smoke performed six forward calls and is explicitly distinguished from the complete pass.

Both models remained in evaluation mode without gradients or optimizer updates. Their parameter and buffer hashes before and after observation agree. Cached and staged checkpoint bytes remain unchanged. The native input/assignment observer was checked on zero-, one- and two-known-corner targets and empty scenes. The seven native implementation files, the 96-file published source closure, all staged image and label hashes, and all ten output ZIP members were verified.

## Evidence

- [notebook-execution.json](notebook-execution.json): all four cell statuses, unchanged cell hashes, runtime, timing, frozen-state guards and final package verification.
- [executed-notebook.ipynb](executed-notebook.ipynb): the actual notebook with captured stream and rich-display outputs; source cells remain unchanged.
- `notebook-cell-1.txt` through `notebook-cell-4.txt`: Python and native/subprocess output logs.
- [results/summary.json](results/summary.json), [results/groups.csv](results/groups.csv) and [results/results.md](results/results.md): group-level observations.
- [results/targets.json.gz](results/targets.json.gz): target identity, eligibility, selection, assignment, score and supervision components. It is compressed once in this evidence directory.
- [results/images.json](results/images.json): all model/image/head records, including empty scenes.
- [results/manifest.json](results/manifest.json): exact hashes and sizes of the five analytic outputs.
- [package-manifest.json](package-manifest.json): independently verified membership and hashes of the complete ten-member return ZIP.
- [plan.json](plan.json), [inputs/pr157-return-manifest.json](inputs/pr157-return-manifest.json), [source/audit_score_assignment_r65.py](source/audit_score_assignment_r65.py) and [smoke/summary.json](smoke/summary.json): fixed input/source identities and the separate smoke result.
- [execute_notebook.py](execute_notebook.py): the runner used for this local execution and independent package checks. It preserves failures and refuses to replace an existing execution receipt.
- [inventory.json](inventory.json): hashes and sizes of this evidence directory, excluding the inventory itself.

The verified ZIP is `baomai-score-assignment-r65-results.zip`, 30,651,413 bytes, SHA-256 `1d75827ca9c6532f2648133caf62789294c6dabcd1982acb08987a058f661c29`. Its members are retained here once; a duplicate ZIP is not included in repository evidence. The executed notebook contains no embedded photographs or checkpoint files.

## Execution conditions and limits

Execution used Python 3.12.14, IPython 9.18.1, Torch 2.10.0+cpu, torchvision 0.25.0+cpu, Ultralytics 8.4.182, NumPy 2.4.6, OpenCV 5.0.0.93 and Pillow 13.0.0. It used two CPU threads, deterministic algorithms, and disabled TensorFloat32 settings. The four cells took 18.782, 4.338, 211.274 and 4.968 seconds respectively in this environment; these are execution records, not hardware speed comparisons.

This is a frozen, unaugmented diagnostic on training images. It does not replay historical stochastic training, measure recognition accuracy on an independent set, calibrate scores to physical error, or establish the performance of a model retrained with box-only assignment. The synthetic native-conflict fixture is explicitly distinguished from observed checkpoint cases. No annotations, checkpoints, confidence thresholds or training objectives were changed. The local FileLink fallback completed; a hosted Colab session and its browser download interaction were not exercised.

Interpretation is maintained in [the score-assignment research note](../../research-score-assignment.md). The manuscript was not rebuilt for this diagnostic execution.
