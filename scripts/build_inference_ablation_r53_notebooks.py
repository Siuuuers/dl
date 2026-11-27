#!/usr/bin/env python3
"""Build two four-cell R53 inference notebooks from one published source commit."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import textwrap

SPARSE = [
    '/scripts/', '/metadata/model_card.json', '/models/cassette_lower_yolo26s_pose.yaml',
    '/data/manifest.json', '/data/labels/', '/data/candidates/front-review-2026-11-26-r37/',
    '/data/references/sequence-refined-reference-2026-11-18/',
    '/data/references/visible-front-completion-2026-11-14/',
    '/data/reviews/fronts-2026-11-26-r38/', '/data/reviews/fronts-2026-11-27-r48/',
    '/data/reviews/fronts-2026-11-27-r50/', '/data/reviews/additional-fronts-2026-11-27-r47/',
    '/results/review/front-review-20261126-r38/row-maps.json',
]
TITLES = {'clahe': 'Fixed luminance CLAHE versus unchanged images',
          'scale': 'Image size1600 versus1280'}


def cell(kind, source):
    source = textwrap.dedent(source).strip() + '\n'
    result = {'cell_type': kind, 'id': hashlib.sha256((kind+source).encode()).hexdigest()[:12],
              'metadata': {}, 'source': source.splitlines(keepends=True)}
    if kind == 'code':
        compile(source, '<r53-notebook-cell>', 'exec')
        result.update(execution_count=None, outputs=[])
    return result


def build(source_commit, driver_sha, experiment):
    if re.fullmatch('[0-9a-f]{40}', source_commit) is None or re.fullmatch('[0-9a-f]{64}', driver_sha) is None:
        raise ValueError('Use a complete published source commit and driver SHA256')
    if experiment not in TITLES:
        raise ValueError('Unknown fixed comparison')
    treatment = ('The candidate changes only the OpenCV Lab L channel with CLAHE, '
                 '`clipLimit=2.0`, `tileGridSize=(8,8)`, then converts back to BGR. '
                 'Image geometry and the size1280 inference setting remain fixed. '
                 'This tests inference-only photometric transfer; no model is trained on the filtered images.'
                 if experiment == 'clahe' else
                 'The candidate changes only the Ultralytics image-size setting from1280 to1600. '
                 'The original1280×720 pixels and label coordinates remain unchanged. '
                 'Interpolation creates no new captured detail; actual tensor shapes, latency and memory are recorded.')
    title = TITLES[experiment]
    cells = [cell('markdown', f'''
        # R53 — {title}

        **Select a GPU runtime, then Run all.** This is a small inference comparison,
        not training. Both arms load the same hash-verified **PR137 candidate**,
        the research checkpoint whose saved predictions are reproduced by the
        PR146 reference. The selected deployment model is unchanged.

        {treatment}

        The unchanged1280 control runs first on all20 verified original development
        PNGs. It must reproduce the saved raw and postclip outputs within0.05 pixel
        coordinate and1e-5 probability tolerance, with exact match/miss/unmatched/
        duplicate IDs and per-row5/10-pixel threshold outcomes on all four references.
        A failed baseline stops the treatment and produces a diagnostic bundle.

        Confidence0.13, native NMS IoU0.70, postclip NMS, cap300, FP32, original-pixel
        corner scoring and the four existing references stay fixed. Successful
        runs measure three paired repeats with alternating arm order, after warmup.
        This notebook does not modify annotations, choose thresholds, train a model
        or promote a filter. Its results are development evidence, not a final test.
        '''), cell('markdown', '''
        ## 1. Load the immutable source

        Source, cache and generated results use separate folders. Changed or
        incompatible existing files are preserved. To run another independent
        copy, set `BAOMAI_R53_BASE` to a fresh folder before this cell. The two
        notebooks can share that base; each keeps its own work directory.
        '''), cell('code', f'''
        from pathlib import Path
        import hashlib, json, os, re, subprocess, sys

        SOURCE_COMMIT = "{source_commit}"
        DRIVER_SHA256 = "{driver_sha}"
        EXPERIMENT = "{experiment}"
        assert re.fullmatch(r"[0-9a-f]{{40}}", SOURCE_COMMIT)
        BASE = Path(os.environ.get("BAOMAI_R53_BASE", "/content/baomai-r53")).resolve()
        SOURCE, WORK, CACHE = BASE / "source", BASE / EXPERIMENT, BASE / "cache"
        BASE.mkdir(parents=True, exist_ok=True)
        ENV = {{**os.environ, "GIT_LFS_SKIP_SMUDGE": "1", "GIT_TERMINAL_PROMPT": "0"}}
        def run(command, cwd=None):
            subprocess.run(list(map(str, command)), cwd=cwd, env=ENV, check=True)
        def git(*args):
            return subprocess.check_output(["git", *args], cwd=SOURCE, env=ENV, text=True).strip()
        if not (SOURCE / ".git").exists():
            if SOURCE.exists() and any(SOURCE.iterdir()):
                raise RuntimeError("Nonempty source folder preserved. Choose a fresh BAOMAI_R53_BASE.")
            run(["git", "init", SOURCE])
            run(["git", "remote", "add", "origin", "https://github.com/Siuuuers/dl.git"], SOURCE)
            run(["git", "fetch", "--depth=1", "--filter=blob:none", "origin", SOURCE_COMMIT], SOURCE)
            run(["git", "sparse-checkout", "init", "--no-cone"], SOURCE)
            run(["git", "sparse-checkout", "set", *{SPARSE!r}], SOURCE)
            run(["git", "checkout", "--detach", SOURCE_COMMIT], SOURCE)
        if git("rev-parse", "HEAD") != SOURCE_COMMIT or git("status", "--porcelain", "--untracked-files=all"):
            raise RuntimeError("Changed or incompatible source preserved. Choose a fresh BAOMAI_R53_BASE.")
        # A clean existing sparse checkout may still lack a required path.
        run(["git", "sparse-checkout", "init", "--no-cone"], SOURCE)
        run(["git", "sparse-checkout", "set", *{SPARSE!r}], SOURCE)
        SCRIPT = SOURCE / "scripts/run_inference_ablation_r53.py"
        assert hashlib.sha256(SCRIPT.read_bytes()).hexdigest() == DRIVER_SHA256
        def command(action, *extra):
            return [sys.executable, "-B", SCRIPT, action, "--experiment", EXPERIMENT,
                "--source-commit", SOURCE_COMMIT, "--script-sha256", DRIVER_SHA256,
                "--root", SOURCE, "--work", WORK, "--cache-dir", CACHE, *extra]
        print("Source:", SOURCE_COMMIT, "Comparison:", EXPERIMENT)
        print("Work:", WORK)
        '''), cell('markdown', '''
        ## 2. Audit the checkpoint, references and original photographs

        This CPU step verifies the exact PR137 result archive, its source/data
        lineage, the selected checkpoint bytes, all20 original PNGs and four score
        references. It downloads no training photos or epoch173 initializer and
        does not load a model. Hash-verified cache inputs can be reused.
        '''), cell('code', '''
        run(command("prepare"))
        preflight = json.loads((WORK / "preflight.json").read_text())
        print("Verified images:", len(preflight["contract"]["images"]))
        print("Checkpoint SHA256:", preflight["contract"]["checkpoint_sha256"])
        print("Archive SHA256:", preflight["contract"]["archive_sha256"])
        '''), cell('markdown', '''
        ## 3. Run the baseline gate and the paired inference comparison

        The checkpoint requires native **Ultralytics8.4.182** with its `Pose26`
        architecture. Installing an arbitrary newer or older Ultralytics release
        can break deserialization or change inference. Four recorded native source
        hashes and the actual head/backend are checked before predictions.

        This cell pins Ultralytics, OpenCV, NumPy and Pillow. It retains the runtime's
        existing compatible Torch/torchvision versions via a constraints file and
        records those versions, CUDA and device details in the return. It does not
        install a random model or silently replace the CUDA stack. Numerical drift
        still has to pass the baseline gate. GPU memory failure stops the run;
        preserve the folder and select a more capable runtime for a fresh run.

        Timings separate photometry, the prediction call, record conversion/postclip,
        and the total decoded-array-to-output path. Disk reading, model loading and
        camera acquisition are excluded; the measurements do not establish camera FPS.
        '''), cell('code', '''
        from importlib.metadata import version
        import torch
        DEVICE = os.environ.get("BAOMAI_R53_DEVICE", "0")
        assert DEVICE == "cpu" or torch.cuda.is_available(), "Select a GPU runtime, or explicitly set BAOMAI_R53_DEVICE=cpu."
        constraints = BASE / "torch-constraints.txt"
        constraints.write_text("".join(f"{name}=={version(name)}\\n" for name in ("torch", "torchvision")))
        run([sys.executable, "-m", "pip", "install", "--quiet", "-c", constraints,
             "ultralytics==8.4.182", "opencv-python==5.0.0.93", "numpy==2.4.6", "Pillow==13.0.0"])
        run(command("run", "--device", DEVICE))
        '''), cell('markdown', '''
        ## 4. Review and download the evidence

        All four score references remain separate. The bundle includes raw/postclip
        predictions, paired stable row IDs, visibility and original-height strata,
        missed and unmatched/duplicate cases, baseline tolerances, all three repeat
        outputs, latency/memory records and a hash inventory. A CPU verification
        pass independently reconstructs postclip filtering, scores, paired cases
        and baseline/repeatability gates.

        `interpretation_allowed=false` means the runtime did not reproduce the
        required baseline or repeatability contract. Keep and inspect that evidence;
        it does not support a claim that the treatment helped or harmed the model.
        Even a valid improvement here needs an independent future capture session.
        '''), cell('code', '''
        run(command("verify"))
        summary = json.loads((WORK / "summary.json").read_text())
        print("Status:", summary["status"], "Interpretation allowed:", summary["interpretation_allowed"])
        print("reference | arm | misses | unmatched | duplicates | two-corner rows within5px")
        for reference in ("frozen666", "historical663", "r38", "r50"):
            for arm, reports in summary["reports"].items():
                report = reports[reference]["postclip"]
                d, c = report["detection"], report["corners_by_locatable_count"]["2"]
                print(reference, arm, d["fn"], d["fp"], d["duplicate_fp"], f"{c['within5']}/{c['eligible']}")
        timing = json.loads((WORK / "timing.json").read_text())
        print(json.dumps(timing["summary"], indent=2))
        ARCHIVE = WORK / f"baomai-{EXPERIMENT}-inference-r53-results.zip"
        from IPython.display import FileLink, display
        try:
            from google.colab import files
        except ImportError:
            display(FileLink(str(ARCHIVE)))
        else:
            files.download(str(ARCHIVE))
        ''')]
    return {'cells': cells, 'nbformat': 4, 'nbformat_minor': 5,
        'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
            'language_info': {'name': 'python'}, 'colab': {'name': f'baomai_{experiment}_inference_r53.ipynb', 'provenance': []},
            'source_commit': source_commit, 'driver_sha256': driver_sha,
            'experiment': experiment, 'training': False, 'fresh_inference': True,
            'automatic_promotion': False, 'accelerator': 'GPU'}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-commit', required=True)
    parser.add_argument('--output-dir', type=Path, default=Path(__file__).resolve().parents[1] / 'notebooks')
    args = parser.parse_args()
    driver = Path(__file__).with_name('run_inference_ablation_r53.py')
    driver_sha = hashlib.sha256(driver.read_bytes()).hexdigest()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for experiment in TITLES:
        target = args.output_dir / f'baomai_{experiment}_inference_r53.ipynb'
        target.write_text(json.dumps(build(args.source_commit, driver_sha, experiment), indent=1) + '\n')
        print(json.dumps({'path': str(target), 'source_commit': args.source_commit,
                          'driver_sha256': driver_sha, 'code_cells': 4, 'training': False}))


if __name__ == '__main__':
    main()
