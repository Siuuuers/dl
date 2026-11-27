#!/usr/bin/env python3
"""Build the four-cell, CPU-only R52 annotation study notebook."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import textwrap

TRAINING_SOURCE = 'da5a67d4bec8171246c56c6b03b34ee20fc519e0'
SPARSE = [
    '/scripts/', '/data/manifest.json', '/data/labels/',
    '/data/candidates/front-review-2026-11-26-r37/',
    '/data/references/sequence-refined-reference-2026-11-18/',
    '/data/references/visible-front-completion-2026-11-14/',
    '/data/reviews/fronts-2026-11-26-r38/', '/data/reviews/fronts-2026-11-27-r50/',
    '/data/reviews/additional-fronts-2026-11-27-r47/',
    '/data/reviews/additional-fronts-2026-11-27-r48/',
    '/data/reviews/additional-fronts-2026-11-27-r52/',
    '/results/evaluation/pr146-r51-20261127/*.json', '/docs/*.md',
]


def cell(kind, source):
    source = textwrap.dedent(source).strip() + '\n'
    result = dict(cell_type=kind, id=hashlib.sha256((kind + source).encode()).hexdigest()[:12],
                  metadata={}, source=source.splitlines(keepends=True))
    if kind == 'code':
        compile(source, '<r52-notebook-cell>', 'exec')
        result.update(execution_count=None, outputs=[])
    return result


def build(source_revision):
    if source_revision != 'SOURCE_REVISION' and re.fullmatch(r'[0-9a-f]{40}', source_revision) is None:
        raise ValueError('Use a full published commit SHA')
    cells = [cell('markdown', '''
        # R52 — cassette annotation study

        **Use an ordinary CPU runtime and Run all.** Four code cells reproduce the
        PR146 saved-result audit, the R52 before/after annotation checks, both PDF
        reports and one downloadable report bundle.

        PR146 completed the paired R51 training. On the reviewed R50 reference,
        classification weighting recovered four partial fronts (15 → 11 misses)
        but increased unmatched outputs (84 → 112). This notebook investigates
        annotation quality and review workload; it does not run another training
        comparison or promote a model.

        It reads the two pinned result archives, 20 original development PNGs and
        12 supplementary JPEGs. The archives contain model files that are verified
        as bytes; this notebook never loads them. No Torch, Ultralytics, GPU or
        initial-checkpoint download is required. The only installed libraries are
        Pillow and reportlab for PDF rendering.

        Geometry flags identify rows to inspect. A neighboring corner can fall
        inside an axis-aligned box in a dense scene; the flag itself does not prove
        a labeling error. Stable source IDs, original pixels and uncertainty remain
        in the exported evidence.
        '''), cell('markdown', '''
        ## 1. Load the fixed source version

        The source checkout and generated work occupy separate folders. An existing
        changed or incompatible checkout is preserved. To start a separate run,
        set `BAOMAI_R52_BASE` to another folder before running this cell.
        '''), cell('code', f'''
        from pathlib import Path
        import json, os, re, subprocess, sys

        SOURCE_REVISION = "{source_revision}"
        TRAINING_SOURCE = "{TRAINING_SOURCE}"
        assert re.fullmatch(r"[0-9a-f]{{40}}", SOURCE_REVISION), "Use the published R52 notebook with its complete source pin."
        BASE = Path(os.environ.get("BAOMAI_R52_BASE", "/content/baomai-r52")).resolve()
        SOURCE, WORK = BASE / "source", BASE / "work"
        BASE.mkdir(parents=True, exist_ok=True)
        ENV = {{**os.environ, "GIT_LFS_SKIP_SMUDGE": "1", "GIT_TERMINAL_PROMPT": "0"}}
        def run(command, cwd=None):
            subprocess.run(list(map(str, command)), cwd=cwd, env=ENV, check=True)
        def git(*args):
            return subprocess.check_output(["git", *args], cwd=SOURCE, env=ENV, text=True).strip()
        new_checkout = not (SOURCE / ".git").exists()
        if new_checkout:
            if SOURCE.exists() and any(SOURCE.iterdir()):
                raise RuntimeError("Nonempty source folder preserved. Choose a fresh BAOMAI_R52_BASE.")
            run(["git", "init", SOURCE])
            run(["git", "remote", "add", "origin", "https://github.com/Siuuuers/dl.git"], SOURCE)
            run(["git", "fetch", "--depth=1", "--filter=blob:none", "origin", SOURCE_REVISION], SOURCE)
            run(["git", "sparse-checkout", "init", "--no-cone"], SOURCE)
            run(["git", "sparse-checkout", "set", *{SPARSE!r}], SOURCE)
            run(["git", "checkout", "--detach", SOURCE_REVISION], SOURCE)
        if git("rev-parse", "HEAD") != SOURCE_REVISION or git("status", "--porcelain", "--untracked-files=all"):
            raise RuntimeError("Changed or incompatible source preserved. Choose a fresh BAOMAI_R52_BASE.")
        # Reapply the complete sparse set on every run. An earlier clean checkout
        # may omit newly required review/observation paths without appearing dirty.
        run(["git", "sparse-checkout", "init", "--no-cone"], SOURCE)
        run(["git", "sparse-checkout", "set", *{SPARSE!r}], SOURCE)
        # The result audit checks archived Python bytes against the immutable
        # training Git tree, which may be absent from a shallow checkout.
        found = subprocess.run(["git", "cat-file", "-e", TRAINING_SOURCE + "^{{commit}}"],
                               cwd=SOURCE, env=ENV, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if found.returncode:
            run(["git", "fetch", "--depth=1", "--filter=blob:none", "origin", TRAINING_SOURCE], SOURCE)
        SCRIPT = SOURCE / "scripts/run_annotation_study_r52.py"
        def command(action):
            return [sys.executable, "-B", SCRIPT, action, "--root", SOURCE, "--work", WORK]
        print("Source:", SOURCE_REVISION)
        print("Work:", WORK)
        '''), cell('markdown', '''
        ## 2. Verify the archived results and original images

        Downloads are hash-checked and reused. This step fetches only the original
        20 development PNGs and the 12 supplementary JPEGs needed for review; it
        does not download the 137 original training PNGs or the epoch173 initializer.
        The prior PR137 archive is needed to verify the R51 data and reference arm.
        '''), cell('code', '''
        run([sys.executable, "-m", "pip", "install", "--quiet", "Pillow==13.0.0", "reportlab==4.4.9"])
        run(command("prepare"))
        inputs = json.loads((WORK / "inputs.json").read_text())
        print("Verified original photographs:", len(inputs["original_image_sha256"]))
        print("Verified supplementary photographs:", len(inputs["supplementary_image_sha256"]))
        print("PR146 archive:", inputs["pr146"]["sha256"])
        '''), cell('markdown', '''
        ## 3. Replay the annotation study and build both PDFs

        The independent result audit reconstructs every archived score table and
        postclip filter. The annotation replay checks all 507 stable supplementary
        identities, the supported side changes, protected corner slots and whole-photo
        gates. Before/after priority CSVs remain review queues, not automatic edits.

        The PDF exporter uses hash-verified original images and the saved prediction
        cache. A completed compatible review is reused. If a completed artifact has
        been changed, it is preserved and the run asks for a fresh work folder.
        '''), cell('code', '''
        run(command("review"))
        summary = json.loads((WORK / "review/audit/summary.json").read_text())
        for reference in ("frozen666", "historical663", "r38", "r50"):
            for role in ("reference", "candidate"):
                report = summary["reports"][reference]["postclip"][role]
                d, c = report["detection"], report["corners_by_locatable_count"]["2"]
                print(reference, role, "misses", d["fn"], "unmatched", d["fp"],
                      "duplicates", d["duplicate_fp"], "other", d["other_fp"],
                      "both corners within5px", c["within5"], "/", c["eligible"])
        for stage in ("before", "after"):
            geometry = json.loads((WORK / "review/geometry" / ("geometry-" + stage + ".json")).read_text())
            print("Geometry review cues", stage, geometry["counts"])
        print("Both PDFs are in:", WORK / "review/reports")
        '''), cell('markdown', '''
        ## 4. Download the evidence bundle

        The ZIP contains both PDFs, exact result and annotation replays, ranked
        review CSVs, label proposals, source hashes and a file inventory. Source
        archives and model weights are not repeated inside the report bundle.
        Keep this ZIP for the thesis record. The PDFs report observed development
        results; they do not establish performance on a new capture session.
        '''), cell('code', '''
        run(command("package"))
        ARCHIVE = WORK / "baomai-annotation-study-r52.zip"
        from IPython.display import FileLink, display
        for name in ("r52-cassette-amendments.pdf", "r52-results-and-thesis.pdf"):
            display(FileLink(str(WORK / "review/reports" / name)))
        try:
            from google.colab import files
        except ImportError:
            display(FileLink(str(ARCHIVE)))
        else:
            files.download(str(ARCHIVE))
        ''')]
    return dict(cells=cells, nbformat=4, nbformat_minor=5,
        metadata=dict(kernelspec=dict(display_name='Python 3', language='python', name='python3'),
            language_info=dict(name='python'), colab=dict(name='baomai_annotation_study_r52.ipynb', provenance=[]),
            source_revision=source_revision, training_source_revision=TRAINING_SOURCE,
            training=False, fresh_inference=False, model_loaded=False, runtime='CPU',
            purpose='Saved-result replay and original-pixel annotation quality review'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-revision', required=True)
    parser.add_argument('--output', type=Path,
                        default=Path(__file__).resolve().parents[1] / 'notebooks/baomai_annotation_study_r52.ipynb')
    args = parser.parse_args()
    notebook = build(args.source_revision)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + '\n')
    print(json.dumps(dict(output=str(args.output), code_cells=4, runtime='CPU', source_revision=args.source_revision)))


if __name__ == '__main__':
    main()
