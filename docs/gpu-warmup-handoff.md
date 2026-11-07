# Next GPU experiment: test the bias warmup rate

**Keep epoch173 as the working model.** The [first completed comparison](gpu-comparison-2026-11-07.md) found that both fine-tunes increased extra detections and tight corner error. This experiment tests one possible shared cause: the initial learning rate for bias parameters. It does not assume that the cause has been established.

**Colab handoff:** use [baomai_bias_warmup.ipynb](../notebooks/baomai_bias_warmup.ipynb) in a GPU runtime and run its cells in order. It downloads the inputs directly, runs the trial and both recipes, verifies the pair, and downloads `baomai-warmup-results.zip`. The notebook pins training source to `679b87b38e7b0bca5fbd02fb1519a90fb2379541`. The shell route below uses the same revision.

[Implementation verification](warmup-execution-verification-2026-11-07.md): both real four-batch CPU trials and all 96 focused checks pass. These are execution checks; completed GPU fine-tunes and their accuracy comparison are still pending.

## The comparison declared before training

Run both recipes consecutively in one GPU session, using this same clean source revision and software environment. Both initialize freshly from the verified epoch173 checkpoint and use the **original reference labels**.

| Setting | Fresh control | Lower-bias-warmup experiment |
| --- | --- | --- |
| Recipe ID | `paired-finetune-v1` | `warmup-bias-lr-1e-4-v1` |
| Label arm | `reference` | `reference` |
| `warmup_bias_lr` | `0.1` | `0.0001` |
| Nominal `lr0` | `0.0001` | `0.0001` |
| Schedule | 30 epochs; 3 warmup epochs | Same |
| Training | Batch 4, image size 1280, FP32, AdamW, seed 42 | Same |
| Selection | Finalized epoch-30 `last.pt` | Same |

Every other declared scientific setting remains equal, including the full dataset, gradient accumulation, augmentation, losses and RLE gain. Both use 137 training images, 3,772 object labels and 6,839 visible training endpoints; the same 20 validation images remain frozen. The deletion-only candidate from the first experiment is not an arm of this comparison.

In [Ultralytics 8.4.182's pinned trainer](https://github.com/ultralytics/ultralytics/blob/v8.4.182/ultralytics/engine/trainer.py), explicit AdamW retains the configured bias warmup value. The original first bias learning rate is therefore 1,000 times nominal `lr0`. The revised positive value tests that difference while preserving the existing real-update witness. It changes neither the warmup duration nor the rest of the schedule. A resource trial also retains the 30-epoch schedule; it stops after a few actual batches instead of shortening `epochs`.

The fresh control is necessary for the declared pair: hosted GPU environments can change between sessions. The old returned M1 remains historical evidence; this packer requires both new runs to share recorded source, software, GPU name, backend state and input hashes.

## Prepare the GPU checkout

Use Linux with an available CUDA GPU and Python 3.11 or newer. Start a new checkout so existing runs and results remain available:

```bash
GIT_LFS_SKIP_SMUDGE=1 git clone \
  --branch codex/baomai-bias-warmup-20261107 --single-branch \
  https://github.com/Siuuuers/dl.git dl-warmup
cd dl-warmup
GIT_LFS_SKIP_SMUDGE=1 git checkout --detach 679b87b38e7b0bca5fbd02fb1519a90fb2379541
git lfs install --local
git lfs pull \
  --include='data/images/train/*.png,data/images/val/*.png,weights/cassette_lower_epoch173.pt' \
  --exclude=''
git status --short
git rev-parse HEAD
```

Keep the checkout clean and at the same revision through both runs. Git supplies labels and code; selective Git LFS supplies the original image and checkpoint bytes. Camera access is not required.

Use working CUDA-enabled PyTorch and compatible torchvision. The [official PyTorch selector](https://pytorch.org/get-started/locally/) describes installation for a particular machine. Preserve those installed versions while adding the pinned trainer:

```bash
python - <<'PY'
from importlib.metadata import version
from pathlib import Path
import torch
import torchvision
assert torch.cuda.is_available(), "Select a GPU runtime first."
print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))
Path('../baomai-warmup-torch-constraints.txt').write_text(
    ''.join(f'{name}=={version(name)}\n' for name in ('torch', 'torchvision'))
)
PY
python -m pip install -c ../baomai-warmup-torch-constraints.txt \
  'ultralytics==8.4.182' psutil
```

Do not install the camera-machine lock file for this experiment. Resolve dependency conflicts before starting the runs. The notebook accompanying the handoff performs these steps in order and preserves CUDA PyTorch in the same way.

## Run the resource trial and matched pair

The following examples allow 12,000 MiB of **host RAM**, 600 seconds for the trial and six hours for each full run. These are stop budgets, not expected usage or a speed promise. Choose a host budget appropriate to available memory. GPU VRAM is separate; the notebook prints `nvidia-smi`. The short trial does not qualify later validation or checkpoint finalization memory.

All output directories must be new and outside the checkout:

```bash
python scripts/train_pose_comparison.py \
  --trial --arm reference --recipe warmup-bias-lr-1e-4-v1 \
  --device 0 --threads 4 --trial-batches 4 \
  --max-seconds 600 --max-rss-mib 12000 \
  --output ../baomai-warmup-trial-001
```

Continue when the trial exits successfully and its receipt reports `purpose: resource-trial`, `status: trial-complete`, positive optimizer updates and an observed parameter change. It is not a completed model and has no eligible checkpoint manifest.

```bash
python scripts/train_pose_comparison.py \
  --train --arm reference --recipe paired-finetune-v1 \
  --device 0 --threads 4 \
  --max-seconds 21600 --max-rss-mib 12000 \
  --output ../baomai-warmup-control-001

python scripts/train_pose_comparison.py \
  --train --arm reference --recipe warmup-bias-lr-1e-4-v1 \
  --device 0 --threads 4 \
  --max-seconds 21600 --max-rss-mib 12000 \
  --output ../baomai-warmup-low-bias-001
```

Run them sequentially in the same session. Both start from epoch173, not from the trial or the first run. If a run fails, keep its receipt and raw log, resolve the failure and use a fresh output directory. A change to batch size, image size, losses or augmentation is a different experiment.

New receipts and checkpoint manifests use schema 2, with an explicit recipe ID and SHA-256 of its declared configuration. They retain actual resolved settings, source hashes, input hashes, real optimizer updates, per-epoch completion and final checkpoint provenance. Runtime evidence records the resolved device name, effective model memory format and PyTorch backend state without altering those settings. Recipe or environment mismatches cannot silently become an accepted pair.

## Return one results ZIP

After both full runs finish successfully:

```bash
python scripts/package_warmup_results.py \
  --control ../baomai-warmup-control-001 \
  --warmup ../baomai-warmup-low-bias-001 \
  --output ../baomai-warmup-results.zip
```

Return `baomai-warmup-results.zip` as a chat attachment or through a new result PR, as with the first GPU return. Preserve the ZIP unchanged. The packer includes `warmup-results-manifest.json` and six files under each of `control/` and `warmup/`: checkpoint manifest, training receipt, staging manifest, finalized `fit/weights/last.pt`, `fit/results.csv` and raw `train.log`. It checks the pair and all archived bytes before publishing the ZIP. Existing outputs are never overwritten.

The original `package_gpu_results.py` still serves the reference/candidate label experiment. Use the new packer for this experiment; the roles describe two recipes on the same reference labels.

## How the returned experiment will be assessed

Run fresh inference on **M0, the new control and the new warmup model together** in one evaluator environment. Use the unchanged frozen validation images and current evaluator/scorer: image size 1280, rectangular preprocessing, batch 1, FP32, ordinary NMS at IoU 0.70, `max_det=300`, box thresholds 0.15/0.25/0.40, matching IoU 0.50 and endpoint confidence 0.50. Select each trained model through its returned checkpoint manifest.

At the predeclared primary box threshold 0.25, require all five unrounded measures to be no worse and at least one to improve: **FN, total FP, missing visible endpoints, conditional median error and conditional p95 error**. Missing statistics do not pass. Assess the warmup model against both M0 and the matched fresh control before adopting the revised recipe. Improving against the control while still failing against M0 does not qualify it to replace M0.

Retain both-corner radius success with missed objects in the denominator, common-endpoint comparisons, empty-image false detections and reference-unlocatable output counts with their denominators. The trainer's automatic `best.pt` console score is not the final-epoch direct-pixel comparison. No threshold tuning, coordinate offset, label correction or second training change is part of this test.

The current 20-image validation set has already been used for development. This one-seed comparison tests a bounded hypothesis on that set; it does not establish repeatability or independent generalization. An untouched test set remains a later requirement for a final assessment.

## Historical result compatibility

Schema-1 checkpoint bundles from PR #13 retain their original identities and validation behavior. They are never relabeled as the new experiment. Exact historical comparison replay additionally checks implementation hashes, so reproduce the old report with its recorded code revision `484f5671456f004e4198173537fd847e0a56db90`, following [the historical report](https://github.com/Siuuuers/dl/blob/155f1118f9773d22af944c4b5774fc406d51d943/docs/gpu-comparison-2026-11-07.md). The new recipe-aware `pose_checkpoint.py` has a new source hash; do not weaken that historical replay check.
