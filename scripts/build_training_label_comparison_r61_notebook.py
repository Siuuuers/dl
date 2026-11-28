#!/usr/bin/env python3
"""Build the controlled training-label notebook from published implementation."""
import argparse
import hashlib
import json
from pathlib import Path
import re

from build_inference_ablation_r53_notebooks import cell

SPARSE = ['/scripts/', '/data/manifest.json', '/data/labels/',
    '/data/candidates/front-review-2026-11-26-r37/',
    '/data/reviews/fronts-2026-11-26-r37/',
    '/data/reviews/fronts-2026-11-27-r50/',
    '/data/references/sequence-refined-reference-2026-11-18/']


def build(source_commit, driver_sha):
    if not re.fullmatch('[0-9a-f]{40}', source_commit) or not re.fullmatch('[0-9a-f]{64}', driver_sha):
        raise ValueError('A complete published source commit and driver hash are required')
    cells = [cell('markdown', '''
        # Does changing the training annotation improve recognition?

        **Choose a GPU runtime, then Run all.** Run the model-improvement
        notebook first if GPU time is limited. This separate experiment answers
        the annotation question with two independent training arms.

        Both arms use the same 137 original training photographs, the same
        supplied epoch-173 checkpoint, seed 46, and the current 30-epoch recipe.
        Only the training labels change: supplied labels versus the frozen
        effective labels used by the current detector. The five additional
        training photographs are excluded from both arms.

        Both trained detectors use identical original-view and two-view corner
        averaging inference. The reviewed evaluation reference is primary;
        supplied and earlier corrected references are reported separately.
        These references describe the same development photographs.
        '''), cell('markdown', '''
        ## 1 Load the fixed implementation

        Source, downloaded assets and generated results occupy separate folders.
        To retain files across sessions, set `BAOMAI_TRAINING_LABEL_BASE` to an
        already mounted persistent directory before running. The shared
        `BAOMAI_R61_CACHE` directory reuses the model-improvement notebook's
        verified photographs and checkpoint. Set it to a persistent directory
        too if those downloads should survive a runtime restart. Different
        existing files are preserved; choose a new folder for a different run.
        '''), cell('code', f'''
        from pathlib import Path
        import hashlib, json, os, subprocess, sys

        SOURCE_COMMIT = "{source_commit}"
        DRIVER_SHA256 = "{driver_sha}"
        BASE = Path(os.environ.get("BAOMAI_TRAINING_LABEL_BASE", "/content/baomai-training-label-comparison-r61")).resolve()
        SOURCE, WORK = BASE / "source", BASE / "work"
        CACHE = Path(os.environ.get("BAOMAI_R61_CACHE", "/content/baomai-r61-cache")).resolve()
        ASSETS = CACHE / "assets"
        BASE.mkdir(parents=True, exist_ok=True)
        CACHE.mkdir(parents=True, exist_ok=True)
        ENV = {{**os.environ, "GIT_LFS_SKIP_SMUDGE": "1", "GIT_TERMINAL_PROMPT": "0", "PYTHONDONTWRITEBYTECODE": "1"}}
        def run(command, cwd=None):
            subprocess.run(list(map(str, command)), cwd=cwd, env=ENV, check=True)
        def git(*args):
            return subprocess.check_output(["git", *args], cwd=SOURCE, env=ENV, text=True).strip()
        if not (SOURCE / ".git").exists():
            if SOURCE.exists() and any(SOURCE.iterdir()):
                raise RuntimeError("Nonempty source preserved; choose a fresh BAOMAI_TRAINING_LABEL_BASE.")
            run(["git", "init", SOURCE])
            run(["git", "remote", "add", "origin", "https://github.com/Siuuuers/dl.git"], SOURCE)
            run(["git", "fetch", "--depth=1", "--filter=blob:none", "origin", SOURCE_COMMIT], SOURCE)
            run(["git", "sparse-checkout", "init", "--no-cone"], SOURCE)
            run(["git", "sparse-checkout", "set", *{SPARSE!r}], SOURCE)
            run(["git", "checkout", "--detach", SOURCE_COMMIT], SOURCE)
        if git("rev-parse", "HEAD") != SOURCE_COMMIT or git("status", "--porcelain", "--untracked-files=all"):
            raise RuntimeError("Changed source preserved; choose a fresh BAOMAI_TRAINING_LABEL_BASE.")
        run(["git", "sparse-checkout", "init", "--no-cone"], SOURCE)
        run(["git", "sparse-checkout", "set", *{SPARSE!r}], SOURCE)
        SCRIPT = SOURCE / "scripts/run_training_label_comparison_r61.py"
        assert hashlib.sha256(SCRIPT.read_bytes()).hexdigest() == DRIVER_SHA256
        def command(action, *extra):
            return [sys.executable, "-B", SCRIPT, action, "--source-commit", SOURCE_COMMIT,
                    "--script-sha256", DRIVER_SHA256, "--root", SOURCE, "--work", WORK,
                    "--cache-dir", ASSETS, *extra]
        run(command("plan"))
        '''), cell('markdown', '''
        ## 2 Verify and stage the same photographs

        All 157 original photograph hashes and the shared checkpoint are checked
        before training. The two training folders contain identical photographs:
        137 for training and 20 for development. Trainer-validation labels are
        identical in both arms. Evaluation labels never alter training labels.

        Supplied training labels contain 3,772 fronts and 6,839 known corners;
        reviewed training labels contain 3,784 fronts and 6,837 known corners.
        Ninety-two training label files differ numerically. Both arms retain the
        same 13 empty training photographs.
        '''), cell('code', '''
        run(command("prepare"))
        preflight = json.loads((WORK / "preflight.json").read_text())
        print("Input verification:", preflight["status"])
        print("Counts:", preflight["counts"])
        print("Numeric label files changed:", preflight["numeric_changed_training_files"])
        print("Budget per arm:", preflight["budget_per_arm"])
        '''), cell('markdown', '''
        ## 3 Train the fixed pair

        A four-batch resource trial checks real finite gradients and parameter
        updates; its weights are discarded. Each full arm then starts afresh
        from the identical supplied checkpoint.

        Recipe: 30 epochs, 1280-pixel input, batch size 4, FP32 AdamW, seed 46,
        the established augmentations, and unmodified native pose loss. Each
        arm receives 4,110 image presentations, 1,050 batches and 78 expected
        optimizer updates. Actual initialization, sample order and optimizer
        update indices are checked across the arms. Both use final-epoch
        weights; no evaluation score chooses a seed, checkpoint or threshold.

        Completed arms can be reused only when their hashes and runtime match.
        Incomplete runs are preserved; this workflow does not claim to resume
        optimizer state from a weights-only checkpoint.
        '''), cell('code', '''
        from importlib.metadata import version
        import torch
        assert torch.cuda.is_available(), "Select a GPU runtime before training."
        constraints = BASE / "torch-constraints.txt"
        constraints.write_text("".join(f"{name}=={version(name)}" + chr(10) for name in ("torch", "torchvision")))
        run([sys.executable, "-m", "pip", "install", "--quiet", "-c", constraints,
             "ultralytics==8.4.182", "opencv-python==5.0.0.93", "numpy==2.4.6", "Pillow==13.0.0", "psutil"])
        run(command("trial"))
        run(command("train", "--arm", "original_labels"))
        run(command("train", "--arm", "reviewed_labels"))
        '''), cell('markdown', '''
        ## 4 Compare, verify and download

        Both models run on the same photographs in the same runtime. Original
        and reflected views are filtered identically before corner averaging.
        All original-view boxes and confidence values are preserved by averaging.
        Ground-truth coordinates never choose a reflected match or an average.

        Report missed fronts, extra predictions, the duplicate subset, and both
        known corners within five and ten pixels together. All complete reference
        pairs remain in the denominator, including missed detections. Alternative
        annotation references are sensitivity checks, not extra test images.

        Keep the full download. To return through GitHub, extract only the outer
        upload ZIP and upload its compact result ZIP, return manifest and every
        wrapped checkpoint part. Do not unwrap the checkpoint parts.
        '''), cell('code', '''
        run(command("evaluate"))
        run(command("verify"))
        summary = json.loads((WORK / "summary.json").read_text())
        print("Reference | labels | output | misses | extras | duplicates | within 5 px | within 10 px")
        for reference, arms in summary["measurement"]["references"].items():
            for arm, outputs in arms.items():
                for output, scores in outputs.items():
                    d, p = scores["detection"], scores["pairs"]
                    print(reference, arm, output, d["fn"], d["fp"], d["duplicate_fp"],
                          f"{p['within5']}/{p['eligible']}", f"{p['within10']}/{p['eligible']}", sep=" | ")
        print(summary["interpretation"])
        ARCHIVE = WORK / "baomai-training-label-comparison-r61-upload.zip"
        from IPython.display import FileLink, display
        try:
            from google.colab import files
        except ImportError:
            display(FileLink(str(ARCHIVE)))
        else:
            files.download(str(ARCHIVE))
        '''), cell('markdown', '''
        ## Interpretation

        This pair measures one frozen training-label intervention under one
        paired seed. It excludes the five extra photographs so that data exposure
        is identical. It does not measure every rule in a newly written annotation
        definition: those rules need a separately reviewed label release before
        any future training experiment can claim to test them.

        Both arms start from a checkpoint already trained on the supplied data.
        The question is the effect of the subsequent matched fine-tuning labels,
        not learning either dataset from an independent random initialization.
        These 20 photographs have guided project development. No improvement is
        assumed, and no returned model is selected automatically.
        ''')]
    return {'cells': cells, 'nbformat': 4, 'nbformat_minor': 5,
        'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
            'language_info': {'name': 'python'}, 'colab': {'name': 'baomai_training_label_comparison_r61.ipynb', 'provenance': []},
            'source_commit': source_commit, 'driver_sha256': driver_sha, 'experiment': 'controlled_training_label_comparison',
            'training': True, 'fresh_inference': True, 'automatic_promotion': False, 'accelerator': 'GPU'}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-commit', required=True)
    parser.add_argument('--output-dir', type=Path, default=Path(__file__).resolve().parents[1] / 'notebooks')
    args = parser.parse_args()
    driver_sha = hashlib.sha256(Path(__file__).with_name('run_training_label_comparison_r61.py').read_bytes()).hexdigest()
    target = args.output_dir / 'baomai_training_label_comparison_r61.ipynb'
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(build(args.source_commit, driver_sha), indent=1) + '\n')
    print(json.dumps({'path': str(target), 'source_commit': args.source_commit, 'driver_sha256': driver_sha}))


if __name__ == '__main__':
    main()
