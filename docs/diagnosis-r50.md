# R50: complete the frozen-model diagnosis

**The inference question is now complete.** [PR143](https://github.com/Siuuuers/dl/pull/143)
returned the successful R49 experiment with all decoded tensors. The
[R51 audit](front-diagnosis-r51.md) independently verifies it and additionally
replays its outputs on the R50 reference. A fresh R50 run is unnecessary just
to reproduce this result. The next prepared experiment is the
[R51 classification-only training comparison](partial-classification-r51.md).
The original source pins and run instructions below remain as historical,
reproducible evidence.

## Historical R50 handoff

Open [baomai_r50_diagnosis.ipynb](../notebooks/baomai_r50_diagnosis.ipynb) in Colab, select a T4 GPU and Run all. Return `baomai-r50-diagnosis-results.zip`. CPU execution is supported. The notebook is pinned to published source commit `5aa5c04d91fb6e1b5a5cf9cce85215f7564238aa` and the exact script SHA256 `cba4f1df6dcf431c933af1506a4de15718636d9477406eb34779ac4d8ab6ee3e`.

## What this measures

PR137 contains a completed training result. PR140 fixes one source-checkout dependency in the R49 notebook; it does not contain a completed R49 diagnostic return. R50 finishes that controlled inference experiment using the exact PR137 candidate checkpoint and the existing 20 development photographs. No new training treatment or acquisition of photographs is required.

The predeclared intervention replaces ordinary hard NMS with linear Soft-NMS: for IoU greater than 0.70, multiply a remaining proposal's score by `1 − IoU`, keep score strictly greater than 0.13 and retain at most 300 detections. Both arms consume the same freshly decoded FP32 tensor, preserve the keypoints attached to each proposal and apply the same existing postclip filter. Native NMS indices must reconstruct the fresh predictor's outputs exactly. Saved-versus-fresh differences are reported separately from the hard-versus-soft comparison.

Four annotation versions are scored separately on the same 20 photographs:

| Reference | Objects | Role |
|---|---:|---|
| frozen666 | 666 | Unchanged primary reference |
| historical663 | 663 | Historical comparison |
| R38 | 654 | Existing diagnostic reference and stable row identities |
| R50 proposal | 654 | Current development-only annotation proposal |

R50 inherits R48 and changes only `new_000010 / R029` physical bottom to original-image `ymax = 714`; both keypoints are unchanged. The R50 manifest SHA256 is `f639459bdd986f6553b9e00f0ecfdf1eee3c7f5344e95c300a4a17a66352c04f`. The reference versions are not independent test sets. The remaining held annotation is not certified by this experiment.

For R38 and R50 separately, trace every row missed by either the saved candidate or fresh hard-NMS arm. The stages are no IoU-eligible decoded box, confidence gate, ordinary NMS or possible output cap, postclip suppression, and retained-but-unmatched assignment. These locate the loss within the pipeline; they do not prove that object size, shadow, label error or occlusion caused it. The return includes all decoded tensors, exact proposal indices, per-image scores and recovered/newly missed row identities. Corner success is stratified by independently locatable slots. An apparent miss improvement must be read alongside unmatched predictions, their duplicate subset, other unmatched predictions and both-corner 5/10-pixel success.

## Setup and recovery

The source setup checks the exact revision and clean working tree before changing an existing checkout. It reapplies sparse membership on **every** run, fixing the PR140 recovery blind spot where an existing `.git` directory skipped the new row-map pattern. It then verifies the R38 row-map SHA256, every tracked repository Python source file against its Git object, all required source labels, overlays, review evidence, reference manifests and scored labels. This happens before installing packages or importing the inference runtime.

The second cell runs a CPU preflight that restores and audits the exact PR137 archive, replays its saved scores, extracts and hashes the frozen candidate checkpoint, and verifies all 20 original PNG hashes and 1280 × 720 dimensions. A persisted input contract binds the source, script, archive, saved prediction cache, saved summary, checkpoint, originals, all reference labels and inference settings. A mismatch stops and preserves existing data. Choose a fresh `BAOMAI_R50_BASE` directory to keep separate attempts.

Only then does the third cell install `ultralytics==8.4.182`, keeping the Colab PyTorch/torchvision versions constrained to those already installed. The final cell runs the comparison. A completed return is checked against the current contract, its exact ZIP membership and member hashes, native traces, saved cache, replayed postclip outputs and all 24 score tables before reuse. Completed compatible returns are reused; incompatible ones are preserved. No model is automatically promoted.

The CPU preflight is also available directly:

```bash
python -B scripts/diagnose_fronts_r50.py \
  --root /path/to/pinned/source --work /path/to/separate/run \
  --source-revision 5aa5c04d91fb6e1b5a5cf9cce85215f7564238aa \
  --script-sha256 cba4f1df6dcf431c933af1506a4de15718636d9477406eb34779ac4d8ab6ee3e \
  --preflight-only
```

Optional `--archive`, `--cache-dir` and `--originals-dir` reuse already downloaded inputs after verification. The notebook exposes equivalent `BAOMAI_R50_ARCHIVE`, `BAOMAI_R50_CACHE` and `BAOMAI_R50_ORIGINALS` environment variables. `--source-check-only` checks source closure without downloads; `--verify-return` audits an existing completed bundle without importing Ultralytics or using a GPU. `--selftest` checks the fixed Soft-NMS rule and stage classifier with synthetic arrays.

## Return and interpretation

The ZIP contains 20 decoded tensors, the exact diagnosis script, prediction/trace/miss/timing JSON, the PR137 input audit, the immutable preflight receipt, the score summary and a member inventory. It contains no images or model weights. Timing covers suppression and result construction only; it is not an end-to-end benchmark, and CPU NumPy Soft-NMS timing is not an optimized deployment comparison.

Until this notebook returns successfully, no measured claim is made that Soft-NMS reduces misses or unmatched predictions. A filter can select or rescore proposals already produced by the model; it cannot establish the missing cassette geometry when no suitable proposal exists. If the fixed comparison helps, any later model-selection decision must also consider precision, corner placement and the development-only status of these repeatedly reviewed scenes.

Publication verification: the pinned notebook’s actual first two cells passed against source commit `5aa5c04d91fb6e1b5a5cf9cce85215f7564238aa` fetched directly from GitHub. Fresh setup and CPU preflight verified 658 source/data files, the PR137 archive and candidate checkpoint, all 20 original PNGs and all four score references. The test then excluded the R38 row map through sparse checkout and reran the same two cells: setup restored the file, and the complete preflight receipt was byte-identical. No package installation or inference cells were executed. The readable [publication receipt](../results/review/front-review-r50-20261127/notebook-publication-check.json) records the exact source, script, notebook, input-contract and preflight hashes.

Authoring verification also compiled all four notebook code cells and passed the fixed Soft-NMS and stage-classification checks. An isolated clean Git fixture checked preservation of wrong revisions and dirty source, missing-row-map rejection and repair, repeated preflight identity, and rejection without overwrite of an incompatible completed return. The return-reuse negative check did not import Torch or Ultralytics. These setup and integrity checks do not constitute a fresh inference result.

The historical R49 notebook is preserved. Fresh inference and training are not performed while preparing these source changes.
