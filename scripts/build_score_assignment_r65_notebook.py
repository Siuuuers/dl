#!/usr/bin/env python3
"""Build the four-cell frozen-checkpoint score/assignment diagnostic notebook."""
import argparse
import hashlib
import json
from pathlib import Path
import re

from build_inference_ablation_r53_notebooks import SPARSE, cell


def build(source_commit, driver_sha):
    if not re.fullmatch("[0-9a-f]{40}", source_commit) or not re.fullmatch("[0-9a-f]{64}", driver_sha):
        raise ValueError("A complete source commit and driver SHA256 are required")
    cells = [cell("markdown", """
        # Why do the frozen cassette models produce these scores?

        **Run all. No model is trained and no optimizer is created.**
        The notebook compares the native control and QFL final checkpoints from
        the completed paired experiment on their same 142 training photographs
        and unchanged labels. It uses a connected GPU if available, otherwise CPU.
        A GPU runtime is optional; the notebook does not allocate or purchase one.

        Deterministic, unaugmented images pass through the model in evaluation
        mode. Both raw heads are inspected before filtering or suppression.
        Targets are grouped by zero, one and two known corners. The audit retains
        native candidate eligibility, pre-conflict selection, final assignment,
        predicted score p, supervision target q, box CIoU and known-corner OKS.

        A separate box-only-overlap counterfactual reuses the same predictions.
        It is descriptive and changes neither the checkpoints nor annotations.
        This is not a repeat of historical training or an accuracy experiment.
        """), cell("markdown", """
        ## 1 Load the fixed source

        Existing incompatible source and outputs are preserved. For an independent
        rerun, set `BAOMAI_SCORE_ASSIGNMENT_BASE` to a fresh folder. The optional
        `BAOMAI_R61_CACHE` setting reuses the verified original photograph cache.
        """), cell("code", f"""
        from pathlib import Path
        import hashlib, json, os, subprocess, sys
        sys.dont_write_bytecode = True
        SOURCE_COMMIT = "{source_commit}"
        DRIVER_SHA256 = "{driver_sha}"
        assert SOURCE_COMMIT != "0" * 40, "This draft needs its published source pin."
        BASE = Path(os.environ.get("BAOMAI_SCORE_ASSIGNMENT_BASE", "/content/baomai-score-assignment-r65")).resolve()
        SOURCE, WORK = BASE / "source", BASE / "work"
        CACHE = Path(os.environ.get("BAOMAI_R61_CACHE", "/content/baomai-r61-cache")).resolve()
        ASSETS, MODELS = CACHE / "assets", CACHE / "quality-focal-pr157"
        BASE.mkdir(parents=True, exist_ok=True)
        CACHE.mkdir(parents=True, exist_ok=True)
        ENV = {{**os.environ, "GIT_LFS_SKIP_SMUDGE": "1", "GIT_TERMINAL_PROMPT": "0"}}
        def run(command, cwd=None):
            subprocess.run(list(map(str, command)), cwd=cwd, env=ENV, check=True)
        def git(*args):
            return subprocess.check_output(["git", *args], cwd=SOURCE, env=ENV, text=True).strip()
        if not (SOURCE / ".git").exists():
            if SOURCE.exists() and any(SOURCE.iterdir()):
                raise RuntimeError("Nonempty source preserved; choose a fresh BAOMAI_SCORE_ASSIGNMENT_BASE.")
            run(["git", "init", SOURCE])
            run(["git", "remote", "add", "origin", "https://github.com/Siuuuers/dl.git"], SOURCE)
            run(["git", "fetch", "--depth=1", "--filter=blob:none", "origin", SOURCE_COMMIT], SOURCE)
            run(["git", "sparse-checkout", "init", "--no-cone"], SOURCE)
            run(["git", "sparse-checkout", "set", *{SPARSE!r}], SOURCE)
            run(["git", "checkout", "--detach", SOURCE_COMMIT], SOURCE)
        if git("rev-parse", "HEAD") != SOURCE_COMMIT or git("status", "--porcelain", "--untracked-files=all"):
            raise RuntimeError("Changed source preserved; choose a fresh BAOMAI_SCORE_ASSIGNMENT_BASE.")
        run(["git", "sparse-checkout", "init", "--no-cone"], SOURCE)
        run(["git", "sparse-checkout", "set", *{SPARSE!r}], SOURCE)
        SCRIPT = SOURCE / "scripts/audit_score_assignment_r65.py"
        assert hashlib.sha256(SCRIPT.read_bytes()).hexdigest() == DRIVER_SHA256
        def command(action, *extra):
            return [sys.executable, "-B", SCRIPT, action, "--root", SOURCE, "--work", WORK,
                    "--source-revision", SOURCE_COMMIT, "--script-sha256", DRIVER_SHA256, *extra]
        run(command("source-check"))
        print("Source:", SOURCE_COMMIT)
        print("Outputs:", WORK)
        """), cell("markdown", """
        ## 2 Verify the checkpoints, native code and training inputs

        Existing Torch and torchvision installations are retained. If absent,
        CPU-only versions are installed. The exact seven native source files used
        by this audit are hash-checked before any checkpoint is loaded.

        The compact returned evidence archive supplies the exact training labels
        and image hashes. Whole checkpoint hashes, wrapper/member hashes when
        downloading, and every staged photograph and label are verified.
        No development photograph is staged for this diagnostic.
        """), cell("code", """
        from importlib.util import find_spec
        from importlib.metadata import version
        if find_spec("torch") is None or find_spec("torchvision") is None:
            run([sys.executable, "-m", "pip", "install", "--quiet", "--index-url",
                 "https://download.pytorch.org/whl/cpu", "torch==2.10.0+cpu", "torchvision==0.25.0+cpu"])
        constraints = BASE / "torch-constraints.txt"
        constraints.write_text("".join(f"{name}=={version(name)}\\n" for name in ("torch", "torchvision")))
        run([sys.executable, "-m", "pip", "install", "--quiet", "-c", constraints,
             "ultralytics==8.4.182", "opencv-python==5.0.0.93", "numpy==2.4.6", "Pillow==13.0.0"])
        run(command("native-check"))
        sys.path.insert(0, str(SOURCE / "scripts"))
        import audit_score_assignment_r65 as audit_driver
        ARCHIVE = audit_driver.obtain_evidence(MODELS)
        run(command("prepare", "--archive", ARCHIVE, "--inputs-dir", ASSETS,
                    "--models-dir", MODELS, "--download-inputs"))
        print("Verified 142 unchanged training photographs and two complete frozen checkpoints.")
        """), cell("markdown", """
        ## 3 Check observation, then run the complete audit

        The three-photo smoke contains zero-, one- and two-known-corner targets
        plus an empty scene. On guarded examples, diagnostic inputs are compared
        directly with the native PoseLoss26 call and logging must preserve every
        returned assignment tensor exactly. The box/OKS decomposition is checked
        against native overlap values. Parameter and buffer hashes must remain
        unchanged, with every module in evaluation mode and no gradients.

        The full pass then audits all 142 photographs for both checkpoints.
        Results distinguish a target without an eligible center, a target without
        a surviving assignment, and a surviving anchor with zero target score.
        Progress is printed every ten photographs. A completed smoke is never
        reported as completion of the full pass.
        """), cell("code", """
        run(command("audit", "--archive", ARCHIVE, "--device", "auto", "--smoke"))
        run(command("verify", "--archive", ARCHIVE, "--smoke"))
        run(command("audit", "--archive", ARCHIVE, "--device", "auto"))
        run(command("verify", "--archive", ARCHIVE))
        result = json.loads((WORK / "results/summary.json").read_text())
        assert result["full142_completed"] and result["photographs_per_model"] == 142
        print("Full diagnostic completed:", result["forward_calls"], "forward calls.")
        print(result["interpretation"])
        """), cell("markdown", """
        ## 4 Read and return the evidence

        The table averages over surviving anchors. Per-target records preserve
        identities, all eligible anchor IDs, selected anchors, p/q, CIoU/OKS and
        the two assignment variants. They allow us to distinguish target quality
        from a prediction below its target without guessing from aggregate loss.

        These are training-image diagnostics at frozen final checkpoints.
        They cannot establish historical learning causes, calibrate a score as a
        physical error bound, or show that box-only assignment would improve a
        retrained model. The return contains no photographs or checkpoint files.
        """), cell("code", """
        run(command("package", "--archive", ARCHIVE))
        from IPython.display import Markdown, FileLink, display
        display(Markdown((WORK / "results/results.md").read_text()))
        RETURN = WORK / "baomai-score-assignment-r65-results.zip"
        try:
            from google.colab import files
        except ImportError:
            display(FileLink(str(RETURN)))
        else:
            files.download(str(RETURN))
        """)]
    return {"cells": cells, "nbformat": 4, "nbformat_minor": 5, "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"}, "source_commit": source_commit, "driver_sha256": driver_sha,
        "source_published": source_commit != "0" * 40, "training_required": False,
        "experiment": "frozen_score_assignment_audit", "colab": {"name": "baomai_score_assignment_r65.ipynb", "provenance": []}}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256((root / "scripts/audit_score_assignment_r65.py").read_bytes()).hexdigest()
    path = args.output or root / "notebooks/baomai_score_assignment_r65.ipynb"
    path.write_text(json.dumps(build(args.source_commit, digest), indent=1) + "\n")
    print(json.dumps({"path": str(path), "source_commit": args.source_commit, "driver_sha256": digest}))


if __name__ == "__main__":
    main()
