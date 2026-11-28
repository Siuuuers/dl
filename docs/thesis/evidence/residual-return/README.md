# Residual correction return evidence

The PR155 residual-correction return is fully audited. The implementation and recorded training recipe completed successfully, but its reviewed five-pixel pair count is 564/583, below the research control's 566/583 and current two-view method's 569/583. Mean maximum-corner error improves slightly relative to the control. See the [result explanation](../../residual-results.md).

## Files

| File | Content |
| --- | --- |
| [audit.json](audit.json) | Source, archive, model, arithmetic, scoring, training-mask/order and budget checks; paired comparisons |
| [paired-case-changes.json.gz](paired-case-changes.json.gz) | Every complete reference pair, its arm-specific errors and threshold states |
| [selected-cpu-forward.json](selected-cpu-forward.json) | Actual returned-model CPU inference on 71 proposals from two diagnostic photographs |
| [return-manifest.json](return-manifest.json) | Exact original evidence/checkpoint member hashes |
| [manifest.json](manifest.json) | Hashes and sizes of the evidence files in this folder |

The original compact ZIP has 3,971,513 bytes and SHA-256 `dbf5057d765c2f38bc2a28fcf98104c51c49a09afa5a83a2268495784e08b79c`. Its six wrapped parts reconstruct the 94,404,981-byte model with SHA-256 `19c7e95aca0019c3b1eb60de4c36c6ecd30d35fdcaaa103606cdd7d553f7006a`.

The immutable return is at `244e553b2ea0ae6fe35d518c33da8998c2a31e5e`, under `results/l4/baomai-residual-corner-r59-l4-001/`. Preserve the compact evidence ZIP, return manifest and every wrapped checkpoint part. The original 30-epoch training ran on L4 with Torch 2.14.1+cu130. The independent sample-order and selected-model replay use CPU Torch 2.9.0 and TorchVision 0.24.0.

## Reproduce

Fetch the immutable return, place its nine files in a separate input directory, and obtain the hash-verified prior training archive using the existing `audit_additional_front_pilot_r47_results.py` helper. The completed original-model comparison ZIP is already stored in the adjacent `original-baseline/fresh/` folder.

The source contract contains 697 files. Two historical inputs may require an overlay when using the current report checkout: `results/review/front-review-20261126-r38/row-maps.json` and `scripts/build_thesis_docx_r56.py`. Read their exact bytes from implementation commit `06e55d9c1b2557014aff7ff0d68b5096890fe546` into the same relative paths beneath a separate overlay directory. The audit accepts only the recorded hashes; it does not modify current source files or relax source identity checks.

```bash
python scripts/audit_residual_return_r60.py \
  --directory /path/to/pr155-return \
  --original-archive /path/to/baomai-additional-front-pilot-r47-results.zip \
  --comparison-archive docs/thesis/evidence/original-baseline/fresh/baomai-original-comparison-r59-results.zip \
  --pinned-sources /path/to/historical-source-overlay \
  --output-dir /path/to/replayed-audit
```

For the independent model-forward check, concatenate the verified checkpoint members in manifest order, extract the compact evidence ZIP into a separate directory, and obtain `new_000008.png` and `new_000010.png`. Their expected hashes are recorded in the return's input contract.

```bash
python scripts/replay_residual_model_r60.py \
  --return-dir /path/to/extracted-evidence \
  --model /path/to/correction-final.pt \
  --images-dir /path/to/original-pngs \
  --output /path/to/selected-cpu-forward.json
```

The two images were chosen after the result to inspect a regression and an improvement. The replay checks numerical continuity and the unchanged threshold outcomes; it does not provide a new independent accuracy estimate. No new training, checkpoint selection or automatic model promotion is performed.
