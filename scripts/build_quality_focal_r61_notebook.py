#!/usr/bin/env python3
"""Build the fixed-source paired quality-focal training notebook."""
import argparse
import hashlib
import json
from pathlib import Path
import re

from build_inference_ablation_r53_notebooks import SPARSE, cell


def build(source_commit, driver_sha):
    if not re.fullmatch('[0-9a-f]{40}', source_commit) or not re.fullmatch('[0-9a-f]{64}', driver_sha):
        raise ValueError('Use a full published source commit and driver SHA-256')
    cells = [cell('markdown', '''
        # Improve cassette detection with quality-focal training

        **Run this model experiment first. Select one GPU runtime, then Run all.**
        Two detectors start independently from the same supplied checkpoint and
        receive the same 142 training photographs and exact annotations. The
        candidate changes only the classification objective: native binary
        cross-entropy is multiplied by `abs(probability - native_target)**2`.
        The control keeps the original loss. This is a Quality Focal Loss-inspired
        adaptation, not a complete reproduction of the GFL detector.

        This tests whether different weighting of classification errors improves
        recognition. It does not assume a benefit. Assignment rules, soft-target construction and
        box/corner/visibility loss formulas stay unchanged. Augmentation, seed,
        image exposure and update budget stay fixed. The modulation also changes classification-loss scale;
        this pilot assesses that complete objective change.

        The supplied model and both trained arms are evaluated together. Report
        missed fronts, extra predictions, duplicates, and complete-corner success
        at a fixed score above 0.25. A prespecified AP50 ranking measure uses a
        separate score floor of 0.001 and cap of 300 predictions per image. It does
        not choose a deployment threshold. All three annotation references remain
        visible; these are the same development photographs, not independent tests.
        '''), cell('markdown', '''
        ## 1 Load the fixed implementation

        Existing incompatible source or work is preserved. A fresh
        `BAOMAI_QUALITY_FOCAL_BASE` starts a separate run. The optional
        `BAOMAI_R61_CACHE` directory can be shared with the label-comparison notebook.
        '''), cell('code', f'''
        from pathlib import Path
        import hashlib, json, os, subprocess, sys

        SOURCE_COMMIT = "{source_commit}"
        DRIVER_SHA256 = "{driver_sha}"
        BASE = Path(os.environ.get("BAOMAI_QUALITY_FOCAL_BASE", "/content/baomai-quality-focal-r61")).resolve()
        SOURCE, WORK = BASE / "source", BASE / "work"
        CACHE = Path(os.environ.get("BAOMAI_R61_CACHE", "/content/baomai-r61-cache")).resolve()
        ASSETS = CACHE / "assets"
        BASE.mkdir(parents=True, exist_ok=True)
        CACHE.mkdir(parents=True, exist_ok=True)
        ENV = {{**os.environ, "GIT_LFS_SKIP_SMUDGE": "1", "GIT_TERMINAL_PROMPT": "0"}}
        def run(command, cwd=None):
            subprocess.run(list(map(str, command)), cwd=cwd, env=ENV, check=True)
        def git(*args):
            return subprocess.check_output(["git", *args], cwd=SOURCE, env=ENV, text=True).strip()
        if not (SOURCE / ".git").exists():
            if SOURCE.exists() and any(SOURCE.iterdir()):
                raise RuntimeError("Nonempty source preserved; choose a fresh BAOMAI_QUALITY_FOCAL_BASE.")
            run(["git", "init", SOURCE])
            run(["git", "remote", "add", "origin", "https://github.com/Siuuuers/dl.git"], SOURCE)
            run(["git", "fetch", "--depth=1", "--filter=blob:none", "origin", SOURCE_COMMIT], SOURCE)
            run(["git", "sparse-checkout", "init", "--no-cone"], SOURCE)
            run(["git", "sparse-checkout", "set", *{SPARSE!r}], SOURCE)
            run(["git", "checkout", "--detach", SOURCE_COMMIT], SOURCE)
        if git("rev-parse", "HEAD") != SOURCE_COMMIT or git("status", "--porcelain", "--untracked-files=all"):
            raise RuntimeError("Changed source preserved; choose a fresh BAOMAI_QUALITY_FOCAL_BASE.")
        run(["git", "sparse-checkout", "init", "--no-cone"], SOURCE)
        run(["git", "sparse-checkout", "set", *{SPARSE!r}], SOURCE)
        SCRIPT = SOURCE / "scripts/run_quality_focal_r61.py"
        assert hashlib.sha256(SCRIPT.read_bytes()).hexdigest() == DRIVER_SHA256
        def command(action, *extra):
            return [sys.executable, "-B", SCRIPT, action, "--root", SOURCE, "--work", WORK,
                    "--source-revision", SOURCE_COMMIT, "--script-sha256", DRIVER_SHA256, *extra]
        run(command("source-check"))
        print("Fixed source:", SOURCE_COMMIT)
        print("Generated results:", WORK)
        '''), cell('markdown', '''
        ## 2 Verify data and the actual loss implementation

        This cell can run on CPU. It checks all source, photograph, annotation and
        initializer hashes, constructs the real loaders, and differentiates the
        pretrained model on two training photographs. The control must reproduce
        native losses and gradients. The candidate must preserve assignments,
        soft targets, nonclassification gradients and missing-corner behavior.
        The candidate's classification gradient is checked against an independent
        analytic derivative. These checks perform no optimizer updates.
        '''), cell('code', '''
        from importlib.metadata import version
        import torch
        constraints = BASE / "torch-constraints.txt"
        constraints.write_text("".join(f"{name}=={version(name)}\\n" for name in ("torch", "torchvision")))
        run([sys.executable, "-m", "pip", "install", "--quiet", "-c", constraints,
             "ultralytics==8.4.182", "opencv-python==5.0.0.93", "numpy==2.4.6", "Pillow==13.0.0"])
        sys.path.insert(0, str(SOURCE / "scripts"))
        import audit_additional_front_pilot_r47_results as prior
        ARCHIVE = prior.obtain_archive(CACHE / "pr137")
        run(command("prepare", "--archive", ARCHIVE, "--inputs-dir", ASSETS, "--download-inputs"))
        run(command("cpu-loader", "--archive", ARCHIVE))
        if not (WORK / "loss-guard.json").exists():
            run([sys.executable, "-B", SOURCE / "scripts/check_quality_focal_r61.py", "--root", SOURCE,
                 "--inputs-dir", ASSETS, "--archive", ARCHIVE, "--output", WORK / "loss-guard.json"])
        run(command("verify-cpu", "--archive", ARCHIVE))
        plan = json.loads((WORK / "staging.json").read_text())
        guard = json.loads((WORK / "loss-guard.json").read_text())
        print("Both arms: 142 training photographs, 3999 fronts, identical annotations.")
        print("CPU differentiation guard:", guard["status"])
        print("Budget per arm:", {k: v for k, v in plan["budgets"]["reference"].items()
                                   if k != "optimizer_batch_indices"})
        '''), cell('markdown', '''
        ## 3 Train the matched pair

        A four-batch disposable trial checks finite gradients and real updates on
        the selected GPU. Each complete arm then starts again from epoch 173.
        Both use seed 46, 30 epochs, batch 4, input size 1280, FP32 AdamW and final
        epoch selection: 1080 batches, 4260 image presentations and 80 optimizer
        updates. Gradient accumulation explains the difference between batches and
        updates. The control and candidate use the same GPU/software environment.

        Completed compatible arms are verified and reused. An interrupted partial
        run is preserved for inspection; it is not silently restarted or resized.
        Optionally set `BAOMAI_R61_BACKUP` to an already mounted persistent folder
        to copy completed arms. Without that setting, keep the runtime alive until
        the final download finishes. No Google account authorization is required.
        '''), cell('code', '''
        import torch
        assert torch.cuda.is_available(), "Select a GPU runtime before training."
        backup = os.environ.get("BAOMAI_R61_BACKUP")
        extra = ["--backup-dir", str(Path(backup).resolve() / "quality-focal-r61")] if backup else []
        run(command("run-pair", "--archive", ARCHIVE, *extra))
        for role in ("reference", "candidate"):
            receipt = json.loads((WORK / role / "receipt.json").read_text())
            print(role, receipt["status"], "epochs", len(receipt["epochs"]),
                  "updates", receipt["steps"], "focal exponent", receipt["focal_beta"])
        '''), cell('markdown', '''
        ## 4 Measure, verify and download

        All three models process the same 20 original photographs and their
        explicit horizontal reflections. Reflection coordinates and corner slots
        are returned to the original frame. For the fixed-threshold comparison,
        both pools are filtered above 0.25 before the unchanged mutual-box matching
        and equal coordinate average. Boxes and scores are preserved by averaging.

        Detection AP50 uses the original-view outputs retained above 0.001 after
        the common suppression, with at most 300 predictions per image. This is a
        stated, capped ranking metric, not COCO mAP or a new threshold search.
        Corner success includes every eligible reference pair, including misses.

        The upload ZIP contains compact results, exact labels/source/provenance,
        and both checkpoints in verified small modelpart ZIPs. Extract it and
        upload its contents together to the result PR. A saved-output replay and
        full package verification run before download. No model is auto-selected.
        '''), cell('code', '''
        run(command("evaluate", "--archive", ARCHIVE))
        run(command("verify-evaluation", "--archive", ARCHIVE))
        run(command("package", "--archive", ARCHIVE))
        run(command("verify-return", "--archive", ARCHIVE))
        result = json.loads((WORK / "evaluation/summary.json").read_text())
        for reference in ("r50", "supplied664", "frozen666"):
            print("Annotation reference:", reference)
            for arm, output in (("supplied", "original"), ("reference", "two_view"), ("candidate", "two_view")):
                row = result["measurement"]["references"][reference][arm][output]
                print(arm, output, "misses", row["detection"]["fn"], "extras", row["detection"]["fp"],
                      "complete-pair result", row["pairs"])
        RETURN = WORK / "baomai-quality-focal-r61-upload.zip"
        from IPython.display import FileLink, display
        try:
            from google.colab import files
        except ImportError:
            display(FileLink(str(RETURN)))
        else:
            files.download(str(RETURN))
        ''')]
    return dict(cells=cells, nbformat=4, nbformat_minor=5, metadata={
        'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
        'language_info': {'name': 'python'}, 'source_commit': source_commit,
        'driver_sha256': driver_sha, 'experiment': 'quality_focal_classification',
        'training_required': True, 'accuracy_result_available': False,
        'colab': {'name': 'baomai_quality_focal_r61.ipynb', 'provenance': []}})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-commit', required=True)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    driver_sha = hashlib.sha256((root / 'scripts/run_quality_focal_r61.py').read_bytes()).hexdigest()
    target = args.output or root / 'notebooks/baomai_quality_focal_r61.ipynb'
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(build(args.source_commit, driver_sha), indent=1) + '\n')
    print(json.dumps({'path': str(target), 'source_commit': args.source_commit, 'driver_sha256': driver_sha}))


if __name__ == '__main__':
    main()
