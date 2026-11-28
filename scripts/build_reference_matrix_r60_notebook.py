#!/usr/bin/env python3
"""Build the CPU annotation-reference comparison notebook from published source."""
import argparse
import hashlib
import json
from pathlib import Path
import re

from build_inference_ablation_r53_notebooks import cell

SPARSE = ['/scripts/', '/data/manifest.json', '/data/labels/val/',
    '/data/references/sequence-refined-reference-2026-11-18/',
    '/data/reviews/fronts-2026-11-27-r50/',
    '/docs/thesis/evidence/original-baseline/fresh/baomai-original-comparison-r59-results.zip']


def build(source_commit, driver_sha):
    if re.fullmatch('[0-9a-f]{40}', source_commit) is None or re.fullmatch('[0-9a-f]{64}', driver_sha) is None:
        raise ValueError('Full published source and driver identities are required')
    cells = [cell('markdown', '''
        # Compare the same predictions using three annotation references

        **Use a CPU runtime and Run all.** No GPU, training or new model inference
        is needed. This notebook answers: how does the original supplied model
        compare with the current two-view method when the evaluation annotations
        change? It uses the exact predictions from the already completed common-
        runtime inference comparison, and checks every input before scoring.

        The references are the original supplied annotations (664 fronts), an
        earlier fixed reference (666 fronts), and the reviewed reference (654
        fronts). They describe the same 20 photographs. Every comparison uses
        both methods on the same reference. Changing the reference changes the
        measurement, not what the model predicted or learned.
        '''), cell('markdown', '''
        ## 1 Load the fixed implementation

        Source and results are kept separately. If the existing source has been
        edited, choose a new `BAOMAI_REFERENCE_MATRIX_BASE` directory.
        '''), cell('code', f'''
        from pathlib import Path
        import hashlib, json, os, subprocess, sys

        SOURCE_COMMIT = "{source_commit}"
        DRIVER_SHA256 = "{driver_sha}"
        BASE = Path(os.environ.get("BAOMAI_REFERENCE_MATRIX_BASE", "/content/baomai-annotation-reference-r60")).resolve()
        SOURCE, WORK = BASE / "source", BASE / "work"
        BASE.mkdir(parents=True, exist_ok=True)
        ENV = {{**os.environ, "GIT_LFS_SKIP_SMUDGE": "1", "GIT_TERMINAL_PROMPT": "0", "PYTHONDONTWRITEBYTECODE": "1"}}
        def run(command, cwd=None):
            subprocess.run(list(map(str, command)), cwd=cwd, env=ENV, check=True)
        def git(*args):
            return subprocess.check_output(["git", *args], cwd=SOURCE, env=ENV, text=True).strip()
        if not (SOURCE / ".git").exists():
            if SOURCE.exists() and any(SOURCE.iterdir()):
                raise RuntimeError("Nonempty source preserved; choose a fresh BAOMAI_REFERENCE_MATRIX_BASE.")
            run(["git", "init", SOURCE])
            run(["git", "remote", "add", "origin", "https://github.com/Siuuuers/dl.git"], SOURCE)
            run(["git", "fetch", "--depth=1", "--filter=blob:none", "origin", SOURCE_COMMIT], SOURCE)
            run(["git", "sparse-checkout", "init", "--no-cone"], SOURCE)
            run(["git", "sparse-checkout", "set", *{SPARSE!r}], SOURCE)
            run(["git", "checkout", "--detach", SOURCE_COMMIT], SOURCE)
        if git("rev-parse", "HEAD") != SOURCE_COMMIT or git("status", "--porcelain", "--untracked-files=all"):
            raise RuntimeError("Changed source preserved; choose a fresh BAOMAI_REFERENCE_MATRIX_BASE.")
        run(["git", "sparse-checkout", "init", "--no-cone"], SOURCE)
        run(["git", "sparse-checkout", "set", *{SPARSE!r}], SOURCE)
        SCRIPT = SOURCE / "scripts/run_reference_matrix_r60.py"
        assert hashlib.sha256(SCRIPT.read_bytes()).hexdigest() == DRIVER_SHA256
        def command(action):
            return [sys.executable, "-B", SCRIPT, action, "--source-commit", SOURCE_COMMIT,
                    "--script-sha256", DRIVER_SHA256, "--root", SOURCE, "--work", WORK]
        print("Published implementation:", SOURCE_COMMIT)
        '''), cell('markdown', '''
        ## 2 Verify the saved predictions and annotation references

        The original annotation manifest is byte-identical to the manifest at
        the historical upstream baomai revision. All 20 original label hashes
        are verified. The two later references retain their own manifests.
        The completed inference archive, inventory, model identities, photograph
        hashes, suppression and coordinate averages are checked without loading
        any model. No annotation is edited.
        '''), cell('code', '''
        run(command("prepare"))
        preflight = json.loads((WORK / "preflight.json").read_text())
        print("Photographs:", len(preflight["contract"]["photographs"]))
        print("References:", list(preflight["contract"]["references"]))
        print(preflight["requirements"])
        '''), cell('markdown', '''
        ## 3 Recompute the complete comparison

        All methods use object confidence strictly above 0.25, corner confidence
        at least 0.50, matching intersection-over-union at least 0.50, and the same
        additional suppression at overlap above 0.70. All eligible complete
        reference objects remain in the five-/ten-pixel denominator, including
        missed objects. Duplicates are included in extra predictions.

        The previously reported fixed and reviewed scores must reproduce
        exactly. The new supplied-reference row uses the same predictions and
        scoring rule. References contain different entries and coordinates, so
        a change between references is not automatically a change on identical
        physical objects or evidence of training improvement.
        '''), cell('code', '''
        run(command("run"))
        summary = json.loads((WORK / "summary.json").read_text())
        print("Annotation reference | method | misses | extras | duplicates | pairs within 5 pixels | pairs within 10 pixels")
        for row in summary["matrix"]:
            print(row["reference_name"], row["method"], row["missed"], row["extras"], row["duplicates"],
                  f"{row['within5']}/{row['eligible_pairs']}", f"{row['within10']}/{row['eligible_pairs']}", sep=" | ")
        print(summary["interpretation"])
        '''), cell('markdown', '''
        ## 4 Verify and download the evidence

        Verification recomputes every score, match, paired change and coordinate
        average from the verified source inputs. The result ZIP preserves all
        reference labels, predictions, per-image scores and individual outcomes.
        Keep this ZIP if returning or comparing the analysis later.
        '''), cell('code', '''
        run(command("verify"))
        receipt = json.loads((WORK / "verification.json").read_text())
        print("Verification:", receipt["status"], "Matrix cells:", receipt["matrix_cells"])
        ARCHIVE = WORK / "baomai-annotation-reference-r60-results.zip"
        assert ARCHIVE.is_file()
        from IPython.display import FileLink, display
        try:
            from google.colab import files
        except ImportError:
            display(FileLink(str(ARCHIVE)))
        else:
            files.download(str(ARCHIVE))
        '''), cell('markdown', '''
        ## What would establish an effect from training annotations?

        That is a different experiment: train two copies from the same starting
        checkpoint on the same photographs, with the same settings and budget,
        changing only the training labels. Evaluate both using one fixed
        reference, then retain alternative-reference checks separately.

        A historical 137-photograph experiment already made this comparison for
        an earlier small annotation update. It does not isolate the full effect
        of the latest labels, five additional photographs and current method.
        Repeating inference alone cannot establish that causal training effect.

        These are development photographs used during annotation review and
        method selection. The matrix is not an independent capture evaluation,
        and no model is adopted automatically.
        ''')]
    return {'cells': cells, 'nbformat': 4, 'nbformat_minor': 5,
        'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
            'language_info': {'name': 'python'}, 'colab': {'name': 'baomai_annotation_reference_r60.ipynb', 'provenance': []},
            'source_commit': source_commit, 'driver_sha256': driver_sha, 'experiment': 'annotation_reference_matrix',
            'training': False, 'fresh_inference': False, 'automatic_promotion': False}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-commit', required=True)
    parser.add_argument('--output-dir', type=Path, default=Path(__file__).resolve().parents[1] / 'notebooks')
    args = parser.parse_args()
    driver_sha = hashlib.sha256(Path(__file__).with_name('run_reference_matrix_r60.py').read_bytes()).hexdigest()
    path = args.output_dir / 'baomai_annotation_reference_r60.ipynb'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(build(args.source_commit, driver_sha), indent=1) + '\n')
    print(json.dumps({'path': str(path), 'source_commit': args.source_commit, 'driver_sha256': driver_sha}))


if __name__ == '__main__':
    main()
