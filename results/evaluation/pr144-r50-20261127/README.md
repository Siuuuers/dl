# PR144: verified R50 return confirms the existing diagnosis

The [PR144 return](https://github.com/Siuuuers/dl/pull/144) passed its original source-pinned R50 verifier and an independent CPU replay using the existing R51 audit helpers. **It corroborates the PR143/R51 findings without changing the next experiment.** No inference, training, annotation amendment or model promotion was performed by this audit.

The machine-readable result is [verification.json](verification.json). Reproduce it with [audit_front_diagnosis_r50_results.py](../../../scripts/audit_front_diagnosis_r50_results.py), which delegates original source/input verification to the pinned R50 script in a separate process and reuses the existing tensor/scoring helpers. The frozen R51 training implementation, notebook pins and PR143-focused PDF are unchanged.

## Exact identities

| Item | Identity |
| --- | --- |
| PR144 head | `9e6841592bba4a3053b6e47360b9793b29a6d0ef` |
| Archive | `results/t4/baomai-r50-diagnosis-t4-001/baomai-r50-diagnosis-results.zip` |
| Archive bytes | 15,997,796 |
| Archive SHA256 | `c22da73bf196121cc9090e19b443ae8aa1d7c89a6de5d18353e0d4857d65014f` |
| Original R50 source commit | `5aa5c04d91fb6e1b5a5cf9cce85215f7564238aa` |
| Original R50 script SHA256 | `cba4f1df6dcf431c933af1506a4de15718636d9477406eb34779ac4d8ab6ee3e` |
| Archived source/input contract SHA256 | `ab773117b34850b091f527e73ca50eda5d772a8d2659dfda931ab10885740c0c` |
| R50 reference manifest SHA256 | `f639459bdd986f6553b9e00f0ecfdf1eee3c7f5344e95c300a4a17a66352c04f` |

The original source/input contract is stored in the returned `preflight.json` and `summary.json`. It binds the source closure, four score references, stable row map, image hashes, PR137 input archive, checkpoint and inference settings. The verifier checked 658 source/data files, the real PR137 receipt, all 20 original PNGs, and all 28 inventory members plus the inventory itself.

## What agrees

All **20 decoded NPY files are byte-for-byte identical** to PR143. The complete prediction, trace and PR137 input-audit JSON objects are equal after parsing; their serialized text bytes differ. All 18 previously returned PR143 score tables and the six R50 tables previously replayed in R51 are exactly equal, including per-image matches and endpoint errors.

The independent replay also reproduced 20 native selections and output reconstructions, 20 linear Soft-NMS outputs, 40 postclip filters, and 30 missed-target traces across R38 and R50. No matched-target anchor assignment changes or recovered targets appear in the hard/Soft comparison.

| Reference | Hard FN / FP / duplicate FP | Soft FN / FP / duplicate FP | Both corners within 5px, hard and Soft |
| --- | --- | --- | --- |
| frozen666, primary | 34 / 91 / 17 | 34 / 90 / 17 | 511 / 584 |
| historical663 | 30 / 90 / 17 | 30 / 89 / 17 | 511 / 584 |
| R38 | 15 / 84 / 14 | 15 / 83 / 13 | 566 / 583 |
| R50 | 15 / 84 / 13 | 15 / 83 / 12 | 566 / 583 |

R38 and R50 both retain the same **10 confidence-gate misses, three misses without an IoU-eligible decoded box, and two ordinary-NMS misses**. These are pipeline locations rather than established physical causes. Raw Soft-NMS also recovers no target before postclip. The final comparison removes one proposal and adds none; its duplicate/other-unmatched category remains reference dependent.

The returned runtime identifies a Tesla T4, Python3.13.16, PyTorch2.14.1+cu130, torchvision0.29.1+cu130 and Ultralytics8.4.182, matching PR143. Returned timings cover suppression and construction only; they do not establish deployment latency. Repeating the same checkpoint and development scenes does not create independent test evidence. PR35 remains selected, and the R51 classification-only training question remains the planned next controlled experiment.

## Reproduce the original verifier

Prepare a **clean checkout of the original R50 source commit** with the required sparse files, as done by the first cell of [the published R50 notebook](../../../notebooks/baomai_r50_diagnosis.ipynb). Set the following shell variables to local paths: `R50_SOURCE_DIR`, `R50_CHECK_DIR` (a fresh work directory), `PR137_ZIP`, `PR144_ZIP`, and `ORIGINALS_DIR` (the 20 original PNGs named by image ID).

First create and verify the input contract before placing the returned result in the work directory:

```bash
python -B "$R50_SOURCE_DIR/scripts/diagnose_fronts_r50.py" \
  --root "$R50_SOURCE_DIR" --work "$R50_CHECK_DIR" \
  --source-revision 5aa5c04d91fb6e1b5a5cf9cce85215f7564238aa \
  --script-sha256 cba4f1df6dcf431c933af1506a4de15718636d9477406eb34779ac4d8ab6ee3e \
  --archive "$PR137_ZIP" --originals-dir "$ORIGINALS_DIR" \
  --preflight-only
```

Copy the PR144 ZIP to `$R50_CHECK_DIR/baomai-r50-diagnosis-results.zip`, preserving any existing file. Then run:

```bash
python -B "$R50_SOURCE_DIR/scripts/diagnose_fronts_r50.py" \
  --root "$R50_SOURCE_DIR" --work "$R50_CHECK_DIR" \
  --source-revision 5aa5c04d91fb6e1b5a5cf9cce85215f7564238aa \
  --script-sha256 cba4f1df6dcf431c933af1506a4de15718636d9477406eb34779ac4d8ab6ee3e \
  --archive "$PR137_ZIP" --originals-dir "$ORIGINALS_DIR" \
  --verify-return
```

For the original verifier plus the cross-return tensor, trace and score comparison, also set `CURRENT_REPO` to the current repository and `PR143_ZIP` to the verified PR143 return, then run:

```bash
python -B "$CURRENT_REPO/scripts/audit_front_diagnosis_r50_results.py" \
  --root "$CURRENT_REPO" --source-root "$R50_SOURCE_DIR" \
  --work "$R50_CHECK_DIR" --archive "$PR144_ZIP" \
  --pr143-archive "$PR143_ZIP" --prior-archive "$PR137_ZIP" \
  --originals "$ORIGINALS_DIR" --output /tmp/pr144-r50-verification.json
```

These commands use cached results and CPU arithmetic. They do not invoke model prediction or training.
