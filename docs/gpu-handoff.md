# GPU handoff: train the baomai comparison

**First returned pair:** both GPU runs completed and were evaluated on 2026-11-07. [The measured comparison](gpu-comparison-2026-11-07.md) retains the original epoch173 model. Read that result before starting another identical run. The commands below reproduce the original declared recipe; the proposed next warmup experiment needs explicit versioned support.

The current task is to find cassettes in the supplied baomai D405 images. Camera settings are unknown; robot calibration is outside this experiment. Keep the existing one-class, two-corner target. This handoff trains two comparable models so we can measure whether the reviewed label changes help.

## 1. Put the inputs on the GPU machine

Choose **one** route. Use Linux, Python 3.11 or newer (3.12 matches the tested environment), Git and Git LFS. Allow several GB of free disk space for separate training copies and checkpoints.

### Route A: transfer the prepared ZIP

Save the supplied archive as `baomai-gpu-inputs.zip`, then run from its parent directory:

```bash
unzip baomai-gpu-inputs.zip -d baomai-gpu-package
python baomai-gpu-package/restore_gpu_inputs.py --output dl-gpu
cd dl-gpu
```

The output directory must be new and outside the extracted package. Restoration needs no network or machine-learning packages. It restores the recorded Git revision, copies the real image/checkpoint bytes, verifies hashes and proves the checkout is clean.

| Input | Restored location |
| --- | --- |
| 137 original training PNGs and 20 original validation PNGs | `data/images/train/`, `data/images/val/` |
| Reference labels, frozen split and metadata | `data/labels/`, `data/data.yaml`, `data/manifest.json` |
| Reviewed five-file label candidate | `data/candidates/dedup-2026-11-06/` |
| Original initialization checkpoint | `weights/cassette_lower_epoch173.pt` |
| Training, checking, evaluation and packaging code | `scripts/`, with instructions in `docs/` |

The archive contains these large assets under `assets/`, plus `source.bundle`, `manifest.json`, `manifest.sha256` and the restore helper. Labels and code are inside the Git bundle. Train from the restored checkout. Annotated previews and a Python environment are intentionally excluded.

### Route B: download directly on the GPU machine

This avoids transferring the archive through another device:

```bash
GIT_LFS_SKIP_SMUDGE=1 git clone \
  --branch codex/baomai-gpu-handoff-20261107 --single-branch \
  https://github.com/Siuuuers/dl.git dl-gpu
cd dl-gpu
git lfs install --local
git lfs pull \
  --include='data/images/train/*.png,data/images/val/*.png,weights/cassette_lower_epoch173.pt' \
  --exclude=''
```

Ordinary Git download supplies the labels and code; selective LFS download supplies the actual PNG and checkpoint bytes. Keep this revision unchanged through both runs.

## 2. Prepare the GPU environment

Activate an environment with working CUDA-enabled PyTorch and compatible torchvision. If these are missing, follow the [official PyTorch installation selector](https://pytorch.org/get-started/locally/) for this machine first. The [Ultralytics installation guide](https://docs.ultralytics.com/quickstart/) also recommends choosing PyTorch for the hardware before installing Ultralytics.

From `dl-gpu`, preserve those installed versions while adding the project's pinned trainer:

```bash
python - <<'PY'
from importlib.metadata import version
from pathlib import Path
import torch
import torchvision
assert torch.cuda.is_available(), "Activate a working CUDA PyTorch environment."
print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))
Path("../baomai-torch-constraints.txt").write_text(
    "".join(f"{name}=={version(name)}\n" for name in ("torch", "torchvision"))
)
PY
python -m pip install -c ../baomai-torch-constraints.txt \
  'ultralytics==8.4.182' psutil
python - <<'PY'
import torch
import ultralytics
assert ultralytics.__version__ == "8.4.182"
assert torch.cuda.is_available()
print("CUDA calculation:", (torch.ones(1, device="cuda:0") * 2).item())
PY
```

Resolve dependency conflicts before proceeding; do not remove the constraints to silently replace the GPU runtime. Do not install the whole `requirements.lock.txt`: it includes the original camera-machine stack. These steps do not require connecting a D405.

## 3. Run the short resource trial

All output directories must be new and outside the checkout:

```bash
python scripts/train_pose_comparison.py \
  --trial --arm reference --device 0 --threads 4 \
  --trial-batches 4 --max-seconds 600 --max-rss-mib 6000 \
  --output ../baomai-gpu-trial-001
```

Inspect its `training-receipt.json` and `train.log`. Continue when the receipt reports `purpose: resource-trial`, `status: trial-complete`, positive optimizer updates and a parameter change. This is a resource check, not a trained model.

The RSS limit measures **host RAM**, not GPU VRAM. Observe VRAM with `nvidia-smi`. The trial stops before validation and finalization; those phases can use more memory. No completed GPU training has been established yet. Choose host-memory and time budgets appropriate to this machine; 6,000 MiB and the six-hour limits below are examples, not measured requirements or duration estimates.

## 4. Train both arms sequentially

The runner fixes 30 epochs, batch 4, image size 1280, FP32, AdamW and seed 42. Both arms start freshly from epoch173. Reference labels contain 3,772 training objects; the candidate removes eleven duplicate rows, leaving 3,761. Validation is unchanged. See `docs/train-pose-comparison.md` for the full experiment.

```bash
python scripts/train_pose_comparison.py \
  --train --arm reference --device 0 --threads 4 \
  --max-seconds 21600 --max-rss-mib 6000 \
  --output ../baomai-reference-001

python scripts/train_pose_comparison.py \
  --train --arm candidate --device 0 --threads 4 \
  --max-seconds 21600 --max-rss-mib 6000 \
  --output ../baomai-candidate-001
```

Use the same environment, device and settings for both. Keep source files clean until both runs finish. The runner stages and verifies each dataset automatically. If a run fails, retain its log and receipt, diagnose the failure, and use a fresh output directory for a retry. Do not quietly reduce batch size or image size; that changes the comparison.

## 5. Pack and return the results

After both runs finish successfully:

```bash
python scripts/package_gpu_results.py \
  --reference ../baomai-reference-001 \
  --candidate ../baomai-candidate-001 \
  --output ../baomai-gpu-results.zip
```

The packer checks eligibility and hashes before creating the ZIP. It includes a results manifest and, under each of `reference/` and `candidate/`, these six files:

```text
checkpoint-manifest.json
training-receipt.json
staging-manifest.json
fit/weights/last.pt
fit/results.csv
train.log
```

To return the results through your repository, run the following from `dl-gpu` **only after both runs and packaging succeed**. Choose another unused suffix if this branch or destination already exists:

```bash
(
set -e
git switch -c gpu-return/baomai-gpu-001
git remote set-url origin https://github.com/Siuuuers/dl.git
git lfs install --local
mkdir -p results/gpu
mkdir results/gpu/baomai-gpu-001
unzip ../baomai-gpu-results.zip -d results/gpu/baomai-gpu-001
git add results/gpu/baomai-gpu-001
git add -f \
  results/gpu/baomai-gpu-001/reference/train.log \
  results/gpu/baomai-gpu-001/candidate/train.log
git commit -m "Return baomai GPU reference and candidate models"
git push -u origin gpu-return/baomai-gpu-001
)
```

This preserves both arm folders and `gpu-results-manifest.json`. The existing `.pt` rule sends weights through Git LFS; the explicit command includes otherwise ignored logs. The remote change is necessary for an offline restore because its original `origin` points to the local bundle. GitHub authentication must be available on the GPU machine. Send the pushed branch link in the next session; you can also attach the results ZIP directly instead.

Completion means **both eligible 30-epoch bundles**, not just weights. Do not return staged datasets, caches or intermediate checkpoints. `last.pt` is the predeclared final-epoch model; the trainer's final console score for `best.pt` is not the comparison result. Next, evaluate the original, reference and candidate models under matching settings. Training completion alone does not establish improved accuracy.
