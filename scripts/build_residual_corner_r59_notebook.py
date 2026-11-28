#!/usr/bin/env python3
"""Build one immutable-source, fixed-budget residual corner correction notebook."""
import argparse
import hashlib
import json
from pathlib import Path
import re

from build_inference_ablation_r53_notebooks import SPARSE, cell


def build(source_commit, driver_sha):
    if re.fullmatch('[0-9a-f]{40}', source_commit) is None or re.fullmatch('[0-9a-f]{64}', driver_sha) is None:
        raise ValueError('A full published source commit and driver SHA256 are required')
    name = 'baomai_residual_corner_r59.ipynb'
    cells = [cell('markdown', '''
        # Learn a correction while preserving the initial corner estimate

        **Choose a GPU runtime, then Run all.** This notebook trains one fixed
        model to adjust the two lower cassette corners. It receives a colour crop
        and two maps showing the detector's available corner estimates. A
        zero-initialized output layer predicts two coordinate corrections, so the
        untrained model preserves the original estimates exactly.

        The experiment follows the observation that replacing the original points
        with newly located heatmap peaks reduced accuracy. It tests a different
        learning target: a correction to each available original point. An
        improvement is a hypothesis, not an assumed result.

        The control is the fixed project research detector. It is distinct from
        the original supplied baomai model, whose direct comparison has a separate
        notebook. Neither corner confidence nor availability is learned here;
        the original scores are retained and are not calibrated probabilities
        for the corrected coordinates. This is a development experiment.
        '''), cell('markdown', '''
        ## 1 Load the fixed implementation

        Source, cache and outputs occupy separate directories. Incompatible
        existing files are preserved. Select a fresh `BAOMAI_R59_RESIDUAL_BASE`
        to start a different independent run.
        '''), cell('code', f'''
        from pathlib import Path
        import hashlib, json, os, subprocess, sys

        SOURCE_COMMIT = "{source_commit}"
        DRIVER_SHA256 = "{driver_sha}"
        BASE = Path(os.environ.get("BAOMAI_R59_RESIDUAL_BASE", "/content/baomai-residual-corner-r59")).resolve()
        SOURCE, WORK, CACHE = BASE / "source", BASE / "work", BASE / "cache"
        BASE.mkdir(parents=True, exist_ok=True)
        ENV = {{**os.environ, "GIT_LFS_SKIP_SMUDGE": "1", "GIT_TERMINAL_PROMPT": "0"}}
        def run(command, cwd=None):
            subprocess.run(list(map(str, command)), cwd=cwd, env=ENV, check=True)
        def git(*args):
            return subprocess.check_output(["git", *args], cwd=SOURCE, env=ENV, text=True).strip()
        if not (SOURCE / ".git").exists():
            if SOURCE.exists() and any(SOURCE.iterdir()):
                raise RuntimeError("Nonempty source preserved. Choose a fresh BAOMAI_R59_RESIDUAL_BASE.")
            run(["git", "init", SOURCE])
            run(["git", "remote", "add", "origin", "https://github.com/Siuuuers/dl.git"], SOURCE)
            run(["git", "fetch", "--depth=1", "--filter=blob:none", "origin", SOURCE_COMMIT], SOURCE)
            run(["git", "sparse-checkout", "init", "--no-cone"], SOURCE)
            run(["git", "sparse-checkout", "set", *{SPARSE!r}], SOURCE)
            run(["git", "checkout", "--detach", SOURCE_COMMIT], SOURCE)
        if git("rev-parse", "HEAD") != SOURCE_COMMIT or git("status", "--porcelain", "--untracked-files=all"):
            raise RuntimeError("Changed source preserved. Choose a fresh BAOMAI_R59_RESIDUAL_BASE.")
        run(["git", "sparse-checkout", "init", "--no-cone"], SOURCE)
        run(["git", "sparse-checkout", "set", *{SPARSE!r}], SOURCE)
        SCRIPT = SOURCE / "scripts/run_residual_corner_r59.py"
        assert hashlib.sha256(SCRIPT.read_bytes()).hexdigest() == DRIVER_SHA256
        def command(action, *extra):
            return [sys.executable, "-B", SCRIPT, action, "--source-commit", SOURCE_COMMIT,
                    "--script-sha256", DRIVER_SHA256, "--root", SOURCE, "--work", WORK,
                    "--cache-dir", CACHE, *extra]
        print("Source:", SOURCE_COMMIT)
        print("Work:", WORK)
        '''), cell('markdown', '''
        ## 2 Verify the images, labels, detector and initialization

        The data check verifies 142 training photographs, their 3,999 annotated
        fronts, 20 development photographs, the frozen detector checkpoint and
        all evaluation references. Whole photographs retain their established
        split. Independent capture-group labels are unavailable, so these
        development photographs do not establish performance on a new session.

        The ResNet50 ImageNet1K V2 initializer is checked using its complete file
        hash and size. This cell downloads and checks data; it does not run or
        train a model.
        '''), cell('code', '''
        run(command("prepare"))
        preflight = json.loads((WORK / "preflight.json").read_text())
        staging = json.loads((WORK / "staging-r58.json").read_text())
        print("Training photos:", staging["training_photographs"], "Fronts:", staging["training_fronts"])
        print("Development photos:", staging["evaluation_photographs"])
        print("Initializer:", preflight["contract"]["pretrained_backbone"])
        '''), cell('markdown', '''
        ## 3 Check identity, reproduce the detector and train once

        The actual pretrained network first runs on two training photographs on
        CPU. Its four outputs must be exactly zero, and applying them must
        preserve the original coordinates. A disposable learning step verifies
        a finite loss, nonzero head gradient and zero gradient for an unavailable
        reference endpoint. Experiment training starts again from the recorded
        initial model state.

        The frozen detector must reproduce the archived development predictions
        within 0.05 image pixel and 1e-5 confidence, with unchanged matching and
        success states. A failed gate returns diagnostics and prevents training.

        The model has a five-channel ResNet50, global average pooling and a
        zero-initialized four-output linear layer. Inputs are predicted-box crops
        expanded by a factor of 1.5 and resized with aspect ratio preserved into
        a 256×256 canvas. Gaussian input maps use an 8-canvas-pixel standard
        deviation. **Hint jitter is disabled** so that each hint, correction
        origin and target refer to the same initial point. This differs from the
        earlier heatmap experiment and is recorded explicitly.

        Training uses one-to-one detector-box matches at intersection-over-union
        at least 0.50. A corner contributes a target only when its reference is
        known and its original estimate is finite, in the image and has score
        at least 0.50. The target is the reference-minus-original offset scaled
        into the resized canvas. Known targets outside a predicted crop remain
        valid regression targets and are counted. Unavailable initial points are
        preserved unchanged at inference. Unmatched training proposals have no
        coordinate target; every evaluation proposal is still accounted for.

        Fixed recipe: seed 46, 30 epochs, batch size 32, FP32 AdamW at 1e-4,
        cosine decay to 1e-6, SmoothL1 loss with beta equal to one canvas pixel,
        and no RGB augmentation. Corrections have no learned or manual size
        bound. A nonfinite correction or a corrected point outside the image is
        marked unavailable. The final epoch is used; no development score selects
        a checkpoint. Completed epochs may resume only with the same inputs and
        runtime. No setting changes automatically after a memory failure.
        '''), cell('code', '''
        from importlib.metadata import version
        import torch
        DEVICE = os.environ.get("BAOMAI_R59_RESIDUAL_DEVICE", "0")
        assert DEVICE == "cpu" or torch.cuda.is_available(), "Select a GPU runtime. CPU requires explicit configuration."
        constraints = BASE / "torch-constraints.txt"
        constraints.write_text("".join(f"{name}=={version(name)}\\n" for name in ("torch", "torchvision")))
        run([sys.executable, "-m", "pip", "install", "--quiet", "-c", constraints,
             "ultralytics==8.4.182", "opencv-python==5.0.0.93", "numpy==2.4.6", "Pillow==13.0.0"])
        run(command("run", "--device", DEVICE))
        smoke = json.loads((WORK / "model-smoke.json").read_text())
        print("Pretrained CPU check:", smoke["status"], "Exact zero initial offsets:", smoke["initial_offsets_exactly_zero"])
        if (WORK / "training-crops.json").exists():
            crops = json.loads((WORK / "training-crops.json").read_text())
            print("Training supervision:", crops["counts"])
            print("Required correction sizes:", crops["target_residual_distribution"])
        '''), cell('markdown', '''
        ## 4 Verify the result and return the evidence

        Verification replays every saved offset and its coordinate transform,
        suppression decisions, score table and case classification. It also
        reconstructs the training targets and masks, epoch sample order, training
        budget and model parts. A second set of scores uses the common strict
        box-confidence threshold greater than 0.25. This remains a comparison
        with the research detector; it does not substitute for running the
        original supplied model in the separate direct-comparison notebook.

        Two-corner success includes missed detections in its denominator. The
        offered-pair audit fixes all detector-box matches before classifying
        available pairs. It reports successful pairs, coordinate errors,
        incomplete references and unmatched pairs separately.

        Keep the full downloaded ZIP. To return it through GitHub, extract only
        the outer ZIP and upload the compact evidence ZIP, manifest and every
        small wrapped checkpoint part together. Do not unwrap the checkpoint
        parts. A future independent-capture evaluation is needed before a
        development improvement can justify automatic use.
        '''), cell('code', '''
        run(command("verify"))
        summary = json.loads((WORK / "summary.json").read_text())
        print("Status:", summary["status"], "Interpretation allowed:", summary["interpretation_allowed"])
        print("Reference | arm | misses | unmatched | duplicates | two known corners within 5 px")
        tables = summary.get("reports_at_common_confidence_025", summary["reports"])
        for reference in ("frozen666", "r50"):
            for arm, reports in tables.items():
                report = reports[reference]["postclip"]
                detection, corners = report["detection"], report["corners_by_locatable_count"]["2"]
                print(reference, arm, detection["fn"], detection["fp"], detection["duplicate_fp"],
                      f"{corners['within5']}/{corners['eligible']}")
        ARCHIVE = WORK / "baomai-residual-corner-r59-upload.zip"
        assert ARCHIVE.is_file()
        from IPython.display import FileLink, display
        try:
            from google.colab import files
        except ImportError:
            display(FileLink(str(ARCHIVE)))
        else:
            files.download(str(ARCHIVE))
        '''), cell('markdown', '''
        ## Method source and interpretation

        J. Carreira, P. Agrawal, K. Fragkiadaki and J. Malik, “Human Pose Estimation
        with Iterative Error Feedback,” in *Proceedings of the IEEE Conference
        on Computer Vision and Pattern Recognition*, 2016, pp. 4733–4742.
        https://www.cv-foundation.org/openaccess/content_cvpr_2016/html/Carreira_Human_Pose_Estimation_CVPR_2016_paper.html

        That work feeds images and rendered pose estimates into a network that
        predicts corrections. This notebook adapts the idea to a single step and
        two cassette corners. Its zero-initialized output, ResNet50 backbone,
        unconstrained targets and SmoothL1 loss are explicit design choices; it
        does not reproduce the paper's iterative training or claim its published
        results. The shared training detector has already seen the training
        photographs, so their residuals may be easier than new-session errors.
        ''')]
    return {'cells': cells, 'nbformat': 4, 'nbformat_minor': 5,
        'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
            'language_info': {'name': 'python'}, 'colab': {'name': name, 'provenance': []},
            'source_commit': source_commit, 'driver_sha256': driver_sha,
            'experiment': 'residual_corner_correction', 'training': True, 'fresh_inference': True,
            'automatic_promotion': False, 'accelerator': 'GPU'}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-commit', required=True)
    parser.add_argument('--output-dir', type=Path, default=Path(__file__).resolve().parents[1]/'notebooks')
    args = parser.parse_args()
    digest = hashlib.sha256(Path(__file__).with_name('run_residual_corner_r59.py').read_bytes()).hexdigest()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = args.output_dir/'baomai_residual_corner_r59.ipynb'
    path.write_text(json.dumps(build(args.source_commit, digest), indent=1)+'\n')
    print(json.dumps({'path': str(path), 'source_commit': args.source_commit, 'driver_sha256': digest}))


if __name__ == '__main__':
    main()
