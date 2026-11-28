#!/usr/bin/env python3
"""Build the pinned four-cell R56 horizontal-flip inference notebook."""
import argparse
import hashlib
import json
from pathlib import Path
import re

from build_inference_ablation_r53_notebooks import SPARSE, cell


def build(source_commit, driver_sha):
    if re.fullmatch('[0-9a-f]{40}', source_commit) is None or re.fullmatch('[0-9a-f]{64}', driver_sha) is None:
        raise ValueError('Use a complete published source commit and driver SHA256')
    cells = [cell('markdown', '''
        # R56 Explicit horizontal flip versus unchanged images

        **Select a GPU runtime, then Run all.** The two arms use the same verified
        PR137 research checkpoint at input size 1280. The control receives the
        original pixels. The treatment receives a contiguous horizontal reversal
        of those pixels. No training, label editing, output fusion or model
        promotion occurs. This is development evidence from the same 20 images.

        Predictions from the flipped view are mapped back using continuous image
        coordinates: `x_original = width - x_flipped`. Box horizontal bounds are
        reversed, and the two keypoint slots are swapped `[1, 0]`, including their
        scores. Native inverse-letterbox/clipping happens first; our inverse flip
        happens next; original-frame postclip NMS happens last. No extra clipping
        or generic `augment=True` path is used.

        The unchanged control must reproduce the saved raw/postclip baseline
        within 0.05 pixel coordinate and 1e-5 confidence tolerances, with exact
        match/miss/duplicate identities and per-row 5/10-pixel success states.
        A failed baseline blocks treatment. Three paired repeats test stability.
        '''), cell('markdown', '''
        ## 1 Load the immutable source

        Source, cache, and generated evidence have separate folders. Existing
        incompatible contents are preserved. Choose a fresh `BAOMAI_R56_BASE`
        for a separate run. The source pin includes the R56 driver and its R53
        scoring helpers; every repository Python source is verified before use.
        '''), cell('code', f'''
        from pathlib import Path
        import hashlib, json, os, re, subprocess, sys

        SOURCE_COMMIT = "{source_commit}"
        DRIVER_SHA256 = "{driver_sha}"
        assert re.fullmatch(r"[0-9a-f]{{40}}", SOURCE_COMMIT)
        BASE = Path(os.environ.get("BAOMAI_R56_BASE", "/content/baomai-r56")).resolve()
        SOURCE, WORK, CACHE = BASE / "source", BASE / "flip", BASE / "cache"
        BASE.mkdir(parents=True, exist_ok=True)
        ENV = {{**os.environ, "GIT_LFS_SKIP_SMUDGE": "1", "GIT_TERMINAL_PROMPT": "0"}}
        def run(command, cwd=None):
            subprocess.run(list(map(str, command)), cwd=cwd, env=ENV, check=True)
        def git(*args):
            return subprocess.check_output(["git", *args], cwd=SOURCE, env=ENV, text=True).strip()
        if not (SOURCE / ".git").exists():
            if SOURCE.exists() and any(SOURCE.iterdir()):
                raise RuntimeError("Nonempty source preserved. Choose a fresh BAOMAI_R56_BASE.")
            run(["git", "init", SOURCE])
            run(["git", "remote", "add", "origin", "https://github.com/Siuuuers/dl.git"], SOURCE)
            run(["git", "fetch", "--depth=1", "--filter=blob:none", "origin", SOURCE_COMMIT], SOURCE)
            run(["git", "sparse-checkout", "init", "--no-cone"], SOURCE)
            run(["git", "sparse-checkout", "set", *{SPARSE!r}], SOURCE)
            run(["git", "checkout", "--detach", SOURCE_COMMIT], SOURCE)
        if git("rev-parse", "HEAD") != SOURCE_COMMIT or git("status", "--porcelain", "--untracked-files=all"):
            raise RuntimeError("Changed or incompatible source preserved. Choose a fresh BAOMAI_R56_BASE.")
        run(["git", "sparse-checkout", "init", "--no-cone"], SOURCE)
        run(["git", "sparse-checkout", "set", *{SPARSE!r}], SOURCE)
        SCRIPT = SOURCE / "scripts/run_flip_inference_r56.py"
        assert hashlib.sha256(SCRIPT.read_bytes()).hexdigest() == DRIVER_SHA256
        def command(action, *extra):
            return [sys.executable, "-B", SCRIPT, action, "--source-commit", SOURCE_COMMIT,
                    "--script-sha256", DRIVER_SHA256, "--root", SOURCE,
                    "--work", WORK, "--cache-dir", CACHE, *extra]
        print("Source:", SOURCE_COMMIT, "Comparison: unchanged versus horizontal flip")
        print("Work:", WORK)
        '''), cell('markdown', '''
        ## 2 Audit the checkpoint references and original photographs

        The CPU preflight checks the PR137 archive and checkpoint, all 20 original
        1280×720 PNGs, all four annotation references, row identities, and source
        hashes. It loads no model. Verified existing cache inputs can be reused.
        '''), cell('code', '''
        run(command("prepare"))
        preflight = json.loads((WORK / "preflight.json").read_text())
        print("Verified images:", len(preflight["contract"]["images"]))
        print("Checkpoint SHA256:", preflight["contract"]["checkpoint_sha256"])
        print("Inverse transform:", preflight["contract"]["settings"]["flip"])
        '''), cell('markdown', '''
        ## 3 Run the baseline gate and paired comparison

        The checkpoint uses native Ultralytics 8.4.182 Pose26. Package and native
        source hashes are checked, including the native horizontal-flip convention.
        The runtime's compatible Torch/torchvision pair is retained and recorded.
        Both arms use FP32, size 1280, confidence 0.13, native NMS IoU 0.70,
        postclip NMS, and a maximum of 300 outputs. Coordinates are scored in
        original image pixels at matching IoU 0.50 and endpoint confidence 0.50.

        Timings separate input transformation, prediction, inverse mapping and
        postclip record handling. Camera acquisition and disk reading are excluded.
        Latency is recorded for completeness and is not a selection constraint.
        Memory or baseline failures stop the experiment rather than alter its settings.
        '''), cell('code', '''
        from importlib.metadata import version
        import torch
        DEVICE = os.environ.get("BAOMAI_R56_DEVICE", "0")
        assert DEVICE == "cpu" or torch.cuda.is_available(), "Select a GPU runtime, or explicitly set BAOMAI_R56_DEVICE=cpu."
        constraints = BASE / "torch-constraints.txt"
        constraints.write_text("".join(f"{name}=={version(name)}\\n" for name in ("torch", "torchvision")))
        run([sys.executable, "-m", "pip", "install", "--quiet", "-c", constraints,
             "ultralytics==8.4.182", "opencv-python==5.0.0.93", "numpy==2.4.6", "Pillow==13.0.0"])
        run(command("run", "--device", DEVICE))
        '''), cell('markdown', '''
        ## 4 Verify and download the evidence

        CPU replay checks native-view records, inverse mapping, original-frame
        postclip filtering, all score tables, paired cases, repeatability and timing.
        The bundle includes raw and postclip outputs, native flipped-coordinate
        outputs, all three repeated output sets, source files and a hash inventory.

        Complementarity reports show which reference targets are recovered or lost
        and a reference-associated either-view diagnostic bound. They do not choose
        outputs using ground truth and do not implement a fused detector.
        `interpretation_allowed=false` blocks a treatment conclusion. Even a valid
        development result needs later independent capture evaluation.
        '''), cell('code', '''
        run(command("verify"))
        summary = json.loads((WORK / "summary.json").read_text())
        print("Status:", summary["status"], "Interpretation allowed:", summary["interpretation_allowed"])
        print("reference | arm | misses | unmatched | duplicates | two-corner rows within 5 px")
        for reference in ("frozen666", "historical663", "r38", "r50"):
            for arm, reports in summary["reports"].items():
                report = reports[reference]["postclip"]
                d, c = report["detection"], report["corners_by_locatable_count"]["2"]
                print(reference, arm, d["fn"], d["fp"], d["duplicate_fp"], f"{c['within5']}/{c['eligible']}")
        complement = json.loads((WORK / "complementarity.json").read_text())
        for reference, item in complement["references"].items():
            if item["treatment_executed"]:
                all_rows = item["by_locatable_corner_count"]["all"]
                print(reference, "flip-only recoveries:", len(all_rows["flip_only_case_ids"]),
                      "control-only detections:", len(all_rows["control_only_case_ids"]),
                      "missed by both:", all_rows["missed_by_both"])
        ARCHIVE = WORK / "baomai-flip-inference-r56-results.zip"
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
            'language_info': {'name': 'python'},
            'colab': {'name': 'baomai_flip_inference_r56.ipynb', 'provenance': []},
            'source_commit': source_commit, 'driver_sha256': driver_sha, 'experiment': 'flip',
            'training': False, 'fresh_inference': True, 'automatic_promotion': False,
            'fusion_performed': False, 'accelerator': 'GPU'}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-commit', required=True)
    parser.add_argument('--output-dir', type=Path, default=Path(__file__).resolve().parents[1] / 'notebooks')
    args = parser.parse_args()
    driver_sha = hashlib.sha256(Path(__file__).with_name('run_flip_inference_r56.py').read_bytes()).hexdigest()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    target = args.output_dir / 'baomai_flip_inference_r56.ipynb'
    target.write_text(json.dumps(build(args.source_commit, driver_sha), indent=1) + '\n')
    print(json.dumps({'path': str(target), 'source_commit': args.source_commit,
                      'driver_sha256': driver_sha, 'code_cells': 4, 'training': False}))


if __name__ == '__main__':
    main()
