#!/usr/bin/env python3
"""Build a four-cell, immutable-source CPU augmentation audit notebook."""
import argparse
import hashlib
import json
from pathlib import Path
import re

from build_inference_ablation_r53_notebooks import SPARSE, cell


def build(source_commit, driver_sha):
    if not re.fullmatch('[0-9a-f]{40}', source_commit) or not re.fullmatch('[0-9a-f]{64}', driver_sha):
        raise ValueError('Use a complete source commit and driver SHA256')
    cells = [cell('markdown', '''
        # Check what training augmentation does to our cassette labels

        **Use a CPU runtime and Run all. No model is trained or loaded.**
        This diagnostic uses the same 142 training photographs and frozen labels
        as the quality-focal experiment. It compares the unchanged source labels
        with native letterboxing and with the existing training augmentation.

        Three fixed passes produce 426 image presentations. For every original
        label row, the notebook records clipping, native box-filter decisions,
        source and transformed corner availability, horizontal slot exchange,
        and final target geometry. It creates at most six review panels.

        A removed target is not automatically a mistake: the transformed object
        may be outside the image or too small to identify. A box-area ratio is
        not the physical fraction of visible cassette material. Review the
        transformed pixels before deciding whether an augmentation should change.
        These are new diagnostic draws, not a reconstruction of the random draws
        in a historical training run. No evaluation photographs enter this audit.
        '''), cell('markdown', '''
        ## 1 Load the fixed source

        Existing incompatible source and outputs are preserved. The optional
        `BAOMAI_R61_CACHE` setting shares verified source assets with the existing
        training notebooks. Set `BAOMAI_AUGMENTATION_AUDIT_BASE` to a fresh folder
        for an independent rerun.
        '''), cell('code', f'''
        from pathlib import Path
        import hashlib, json, os, subprocess, sys
        sys.dont_write_bytecode = True

        SOURCE_COMMIT = "{source_commit}"
        DRIVER_SHA256 = "{driver_sha}"
        assert SOURCE_COMMIT != "0" * 40, "This draft needs its published source pin."
        BASE = Path(os.environ.get("BAOMAI_AUGMENTATION_AUDIT_BASE", "/content/baomai-augmentation-audit-r63")).resolve()
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
                raise RuntimeError("Nonempty source preserved; choose a fresh BAOMAI_AUGMENTATION_AUDIT_BASE.")
            run(["git", "init", SOURCE])
            run(["git", "remote", "add", "origin", "https://github.com/Siuuuers/dl.git"], SOURCE)
            run(["git", "fetch", "--depth=1", "--filter=blob:none", "origin", SOURCE_COMMIT], SOURCE)
            run(["git", "sparse-checkout", "init", "--no-cone"], SOURCE)
            run(["git", "sparse-checkout", "set", *{SPARSE!r}], SOURCE)
            run(["git", "checkout", "--detach", SOURCE_COMMIT], SOURCE)
        if git("rev-parse", "HEAD") != SOURCE_COMMIT or git("status", "--porcelain", "--untracked-files=all"):
            raise RuntimeError("Changed source preserved; choose a fresh BAOMAI_AUGMENTATION_AUDIT_BASE.")
        run(["git", "sparse-checkout", "init", "--no-cone"], SOURCE)
        run(["git", "sparse-checkout", "set", *{SPARSE!r}], SOURCE)
        SCRIPT = SOURCE / "scripts/audit_training_augmentation_r63.py"
        assert hashlib.sha256(SCRIPT.read_bytes()).hexdigest() == DRIVER_SHA256
        def command(action, *extra):
            return [sys.executable, "-B", SCRIPT, action, "--root", SOURCE, "--work", WORK,
                    "--source-revision", SOURCE_COMMIT, "--script-sha256", DRIVER_SHA256, *extra]
        run(command("source-check"))
        print("Fixed source:", SOURCE_COMMIT)
        print("Audit outputs:", WORK)
        '''), cell('markdown', '''
        ## 2 Verify the native code and prepare training inputs

        Existing Torch/torchvision installations are retained. If absent, the
        cell installs CPU-only versions from the official PyTorch index. It then
        verifies the exact five native source files controlling augmentation,
        fixtures for box filtering and corner availability, archived training
        labels, and every photograph hash. No checkpoint is downloaded for use
        by this audit; the existing PR137 evidence archive includes model files
        that are never loaded.
        '''), cell('code', '''
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
        import audit_additional_front_pilot_r47_results as prior
        ARCHIVE = prior.obtain_archive(CACHE / "pr137")
        run(command("prepare", "--archive", ARCHIVE, "--inputs-dir", ASSETS, "--download-inputs"))
        print("142 training photographs are verified. The 20 evaluation photographs are not staged.")
        '''), cell('markdown', '''
        ## 3 Audit the actual training transforms

        The audit executes the real dataset and native transformation pipeline
        at the existing 1280 setting, with rectangular batches. Its observation
        hooks preserve the tensors exactly, checked against uninstrumented native
        calls. Native row filtering is also recomputed independently.

        Each sample has a fixed, recorded seed. Both existing source-missing
        points and newly cropped-out known points are tracked separately. The
        source files remain unchanged. Optional Albumentations presence and its
        actual transform list are recorded; an unexpected spatial transform
        stops the audit rather than guessing row lineage.
        '''), cell('code', '''
        run(command("audit", "--archive", ARCHIVE))
        run(command("verify", "--archive", ARCHIVE))
        result = json.loads((WORK / "results/summary.json").read_text())
        print(json.dumps(result["counts"], indent=2))
        print(result["interpretation"])
        '''), cell('markdown', '''
        ## 4 Read, inspect and download

        Review the original and transformed pixels side by side. Red highlights
        rows removed by the native filter; orange highlights retained clipped
        rows or a known corner that moved outside the canvas. These are review
        categories, not a verdict that the labels are wrong. The return includes
        every source-row lineage, aggregate counts, source/runtime identities,
        unaugmented controls, and the small panel selection.

        This diagnostic can motivate one later augmentation-policy comparison.
        It cannot demonstrate that augmentation caused the detector's current
        misses or that a new policy improves recognition.
        '''), cell('code', '''
        run(command("package", "--archive", ARCHIVE))
        from IPython.display import Image, Markdown, FileLink, display
        display(Markdown((WORK / "results/results.md").read_text()))
        for panel in result["panels"]:
            display(Markdown("**" + panel["sample_id"] + "** — " + panel["category"]))
            display(Image(filename=str(WORK / "results" / panel["path"]), width=1400))
        RETURN = WORK / "baomai-augmentation-audit-r63-results.zip"
        try:
            from google.colab import files
        except ImportError:
            display(FileLink(str(RETURN)))
        else:
            files.download(str(RETURN))
        ''')]
    return {'cells': cells, 'nbformat': 4, 'nbformat_minor': 5, 'metadata': {
        'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
        'language_info': {'name': 'python'}, 'source_commit': source_commit, 'driver_sha256': driver_sha,
        'experiment': 'native_training_augmentation_audit', 'training_required': False,
        'model_accuracy_result_available': False, 'source_published': source_commit != '0' * 40,
        'colab': {'name': 'baomai_augmentation_audit_r63.ipynb', 'provenance': []}}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-commit', required=True)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256((root / 'scripts/audit_training_augmentation_r63.py').read_bytes()).hexdigest()
    path = args.output or root / 'notebooks/baomai_augmentation_audit_r63.ipynb'
    path.write_text(json.dumps(build(args.source_commit, digest), indent=1) + '\n')
    print(json.dumps({'path': str(path), 'source_commit': args.source_commit, 'driver_sha256': digest}))


if __name__ == '__main__':
    main()
