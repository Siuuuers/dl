#!/usr/bin/env python3
"""Build two source-pinned local-refinement training notebooks."""
import argparse
import hashlib
import json
from pathlib import Path
import re

from build_inference_ablation_r53_notebooks import SPARSE, cell

MODES = {'none': ('rgb', 'RGB-only local corner refinement'),
         'initial': ('hints', 'RGB plus initial-corner hints')}


def build(source_commit, driver_sha, hint_mode):
    if re.fullmatch('[0-9a-f]{40}', source_commit) is None or re.fullmatch('[0-9a-f]{64}', driver_sha) is None:
        raise ValueError('Use a complete published source commit and driver SHA256')
    if hint_mode not in MODES: raise ValueError('Unknown fixed hint mode')
    suffix, title = MODES[hint_mode]
    name = f'baomai_corner_refinement_{suffix}_r58.ipynb'
    treatment = ('Both extra input channels remain zero. The network must locate the two lower corners from the RGB crop.'
                 if hint_mode == 'none' else
                 'Two extra input channels contain Gaussian hints centred on the frozen detector\'s available corner estimates. Hint positions receive fixed 4-canvas-pixel standard-deviation noise during training only.')
    cells = [cell('markdown', f'''
        # {title}

        **Choose a GPU runtime, then Run all.** This notebook trains one fixed
        local refiner and compares its final checkpoint with the unchanged
        detector on the existing 20 development photographs. The detector boxes,
        box confidence, suppression decisions and original corner scores stay
        fixed. Every raw detection receives the same refinement procedure.

        {treatment}

        The paired notebook uses the identical five-channel architecture, exact
        pretrained weights, seed, training examples, order, optimizer and schedule.
        Its only intervention is the information supplied in the two hint channels.
        Run both with the same Torch/TorchVision versions and GPU type. Their
        receipts record initialization hashes and per-epoch sample-order hashes.

        **This is a research experiment, not a deployment model.** Original corner
        scores are retained for a controlled comparison; they are not calibrated
        probabilities for the refined coordinates. Neither notebook changes
        annotations or selects an epoch from development accuracy.
        '''), cell('markdown', '''
        ## 1 Load the immutable implementation

        Source, cache and generated results use separate folders. The two notebooks
        can share a base folder while keeping separate work directories. An
        incompatible source or changed input is preserved and reported. Select a
        fresh `BAOMAI_R58_REFINER_BASE` for a new independent run.
        '''), cell('code', f'''
        from pathlib import Path
        import hashlib, json, os, re, subprocess, sys

        SOURCE_COMMIT = "{source_commit}"
        DRIVER_SHA256 = "{driver_sha}"
        HINT_MODE = "{hint_mode}"
        BASE = Path(os.environ.get("BAOMAI_R58_REFINER_BASE", "/content/baomai-corner-refinement-r58")).resolve()
        SOURCE, WORK, CACHE = BASE / "source", BASE / "{suffix}", BASE / "cache"
        BASE.mkdir(parents=True, exist_ok=True)
        ENV = {{**os.environ, "GIT_LFS_SKIP_SMUDGE": "1", "GIT_TERMINAL_PROMPT": "0"}}
        def run(command, cwd=None):
            subprocess.run(list(map(str, command)), cwd=cwd, env=ENV, check=True)
        def git(*args):
            return subprocess.check_output(["git", *args], cwd=SOURCE, env=ENV, text=True).strip()
        if not (SOURCE / ".git").exists():
            if SOURCE.exists() and any(SOURCE.iterdir()):
                raise RuntimeError("Nonempty source preserved. Choose a fresh BAOMAI_R58_REFINER_BASE.")
            run(["git", "init", SOURCE])
            run(["git", "remote", "add", "origin", "https://github.com/Siuuuers/dl.git"], SOURCE)
            run(["git", "fetch", "--depth=1", "--filter=blob:none", "origin", SOURCE_COMMIT], SOURCE)
            run(["git", "sparse-checkout", "init", "--no-cone"], SOURCE)
            run(["git", "sparse-checkout", "set", *{SPARSE!r}], SOURCE)
            run(["git", "checkout", "--detach", SOURCE_COMMIT], SOURCE)
        if git("rev-parse", "HEAD") != SOURCE_COMMIT or git("status", "--porcelain", "--untracked-files=all"):
            raise RuntimeError("Changed or incompatible source preserved. Choose a fresh BAOMAI_R58_REFINER_BASE.")
        run(["git", "sparse-checkout", "init", "--no-cone"], SOURCE)
        run(["git", "sparse-checkout", "set", *{SPARSE!r}], SOURCE)
        SCRIPT = SOURCE / "scripts/run_corner_refinement_r58.py"
        assert hashlib.sha256(SCRIPT.read_bytes()).hexdigest() == DRIVER_SHA256
        def command(action, *extra):
            return [sys.executable, "-B", SCRIPT, action, "--source-commit", SOURCE_COMMIT,
                    "--script-sha256", DRIVER_SHA256, "--root", SOURCE, "--work", WORK,
                    "--cache-dir", CACHE, "--hint-mode", HINT_MODE, *extra]
        print("Source:", SOURCE_COMMIT, "Hint mode:", HINT_MODE)
        print("Work:", WORK)
        '''), cell('markdown', '''
        ## 2 Verify images, annotations, detector and external initialization

        The preflight verifies the immutable annotated-data archive and detector
        checkpoint, 142 training photographs, 20 development photographs and all
        four evaluation references. The training set contains 3,999 annotated
        fronts. Whole photographs retain their existing split membership.
        Independent capture-group labels are not available, so the evaluation is
        reported as development evidence.

        The external initializer is TorchVision's ResNet50 ImageNet1K V2 file,
        checked using its complete SHA256 and file length. No model is loaded in
        this cell. The next cell runs an actual pretrained CPU forward/backward
        check before detector gating and experiment training.
        '''), cell('code', '''
        run(command("prepare"))
        preflight = json.loads((WORK / "preflight.json").read_text())
        staging = json.loads((WORK / "staging-r58.json").read_text())
        print("Training photos:", staging["training_photographs"], "Training fronts:", staging["training_fronts"])
        print("Development photos:", staging["evaluation_photographs"])
        print("Backbone:", preflight["contract"]["pretrained_backbone"])
        '''), cell('markdown', '''
        ## 3 Check the real model, reproduce the baseline and train once

        The native inference packages are pinned; the runtime's compatible Torch
        and TorchVision pair is retained and recorded. The real model check uses
        two training photographs, verifies the output shape and a finite masked
        loss, performs one disposable optimization step, and checks that an
        unavailable endpoint receives no coordinate-loss gradient. Experiment
        training reinitializes the model from exactly the recorded initial state.

        Before training, the frozen detector must reproduce the cached development
        predictions within 0.05 image pixel and 1e-5 confidence, with unchanged
        matched/missed/duplicate identities and 5/10-pixel success states. A failed
        gate produces a diagnostic download and stops the experiment.

        Successful gating is followed by frozen inference on training photographs.
        Only one-to-one box matches at intersection-over-union at least 0.50 provide
        coordinate supervision. Missing reference corners and corners outside the
        fixed predicted crop are masked. Unmatched proposals supply no coordinate
        target, but all proposals are accounted for during evaluation.

        Fixed recipe: a 1.5-times predicted-box crop, aspect-preserving resize into
        a 256×256 canvas, two 64×64 heatmaps, three 256-channel deconvolution stages,
        seed 46, 30 epochs, batch 32, FP32 AdamW at 1e-4 with cosine decay to 1e-6.
        RGB augmentation is disabled. The final epoch is used automatically.
        Padding is excluded from loss and decoding. The decoder uses a valid-region
        maximum plus a fixed quarter-cell offset toward the larger neighbour.
        Failures remain failures; no ground-truth-based fallback is used.

        Completed epochs can resume in the same work folder and runtime. Memory
        failures preserve the checkpoint and do not automatically alter the recipe.
        '''), cell('code', '''
        from importlib.metadata import version
        import torch
        DEVICE = os.environ.get("BAOMAI_R58_REFINER_DEVICE", "0")
        assert DEVICE == "cpu" or torch.cuda.is_available(), "Select a GPU runtime. CPU is supported only by explicit configuration."
        constraints = BASE / "torch-constraints.txt"
        constraints.write_text("".join(f"{name}=={version(name)}\\n" for name in ("torch", "torchvision")))
        run([sys.executable, "-m", "pip", "install", "--quiet", "-c", constraints,
             "ultralytics==8.4.182", "opencv-python==5.0.0.93", "numpy==2.4.6", "Pillow==13.0.0"])
        run(command("run", "--device", DEVICE))
        smoke = json.loads((WORK / "model-smoke.json").read_text())
        print("Real CPU mechanics check:", smoke["status"], "Initial-state SHA256:", smoke["initial_model"]["state_sha256"])
        '''), cell('markdown', '''
        ## 4 Replay the scores and download the complete evidence

        Verification reconstructs suppression, crop transforms, prediction records,
        all four score tables and paired case identities. It checks the training
        crop masks, exact epoch exposure, sample-order hashes, update count and final
        model identity. The outer download contains a compact evidence ZIP and
        exact FP32 checkpoint parts wrapped in small ZIP files, with a manifest
        that verifies their reassembly. Keep the entire returned upload ZIP.

        **To return the result through a GitHub pull request:** extract the outer
        upload ZIP, then upload its compact evidence ZIP, `return-manifest.json`
        and every small wrapped checkpoint part together. Do not unwrap the
        individual checkpoint parts. The outer download can exceed GitHub's
        ordinary file-size limits; its extracted members are the upload units.

        Two-corner success uses every eligible reference row, including missed
        detections. Offered-pair accounting matches all fixed detector boxes first
        and retains those assignments when classifying available pairs. This is
        distinct from rematching only an accepted subset. Detection counts should
        stay identical because this experiment changes corner coordinates only.
        A development improvement is evidence for a later independent capture test;
        this notebook does not automatically adopt a model.
        '''), cell('code', '''
        run(command("verify"))
        summary = json.loads((WORK / "summary.json").read_text())
        print("Status:", summary["status"], "Interpretation allowed:", summary["interpretation_allowed"])
        print("Reference | arm | misses | unmatched | duplicates | two known corners within 5 px")
        for reference in ("frozen666", "historical663", "r38", "r50"):
            for arm, reports in summary["reports"].items():
                report = reports[reference]["postclip"]
                d, c = report["detection"], report["corners_by_locatable_count"]["2"]
                print(reference, arm, d["fn"], d["fp"], d["duplicate_fp"], f"{c['within5']}/{c['eligible']}")
        if summary["status"] == "completed":
            accounting = json.loads((WORK / "pair-accounting.json").read_text())
            print(accounting["matching_convention"])
            for reference in ("frozen666", "r50"):
                print(reference, accounting["references"][reference])
        ARCHIVE = WORK / f"baomai-corner-refinement-r58-{HINT_MODE}-upload.zip"
        assert ARCHIVE.is_file()
        from IPython.display import FileLink, display
        try:
            from google.colab import files
        except ImportError:
            display(FileLink(str(ARCHIVE)))
        else:
            files.download(str(ARCHIVE))
        '''), cell('markdown', '''
        ## Method sources

        1. B. Xiao, H. Wu and Y. Wei, “Simple Baselines for Human Pose Estimation
           and Tracking,” ECCV, 2018, pp. 466–481.
           https://arxiv.org/abs/1804.06208
        2. G. Moon, J. Y. Chang and K. M. Lee, “PoseFix: Model-agnostic General
           Human Pose Refinement Network,” CVPR, 2019, pp. 7773–7781.
           https://openaccess.thecvf.com/content_CVPR_2019/html/Moon_PoseFix_Model-Agnostic_General_Human_Pose_Refinement_Network_CVPR_2019_paper.html
        3. TorchVision, “resnet50,” official model documentation.
           https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet50.html

        This cassette experiment adapts the deconvolutional heatmap idea and tests
        whether initial estimates add useful information. It does not reproduce
        the full PoseFix architecture, corruption distribution or benchmark recipe.
        ''')]
    return {'cells':cells,'nbformat':4,'nbformat_minor':5,
        'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},
                    'language_info':{'name':'python'},'colab':{'name':name,'provenance':[]},
                    'source_commit':source_commit,'driver_sha256':driver_sha,'hint_mode':hint_mode,
                    'experiment':'local_corner_refinement','training':True,'fresh_inference':True,
                    'automatic_promotion':False,'accelerator':'GPU'}}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-commit',required=True)
    parser.add_argument('--output-dir',type=Path,default=Path(__file__).resolve().parents[1]/'notebooks')
    args=parser.parse_args()
    driver_sha=hashlib.sha256(Path(__file__).with_name('run_corner_refinement_r58.py').read_bytes()).hexdigest()
    args.output_dir.mkdir(parents=True,exist_ok=True)
    for mode,(suffix,_) in MODES.items():
        path=args.output_dir/f'baomai_corner_refinement_{suffix}_r58.ipynb'
        path.write_text(json.dumps(build(args.source_commit,driver_sha,mode),indent=1)+'\n')
        print(json.dumps({'path':str(path),'source_commit':args.source_commit,'driver_sha256':driver_sha,'hint_mode':mode}))


if __name__=='__main__':main()
