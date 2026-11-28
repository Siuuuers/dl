#!/usr/bin/env python3
"""Build the four-cell CPU notebook for one fixed crop-context comparison."""

import argparse
import hashlib
import json
from pathlib import Path
import re

import analyze_crop_context_r71 as study
import diagnose_current_proposals_r69 as current
from build_inference_ablation_r53_notebooks import cell


def build(source_commit, root):
    if re.fullmatch("[0-9a-f]{40}", source_commit) is None:
        raise ValueError("A complete published implementation commit is required")
    study.protocol(root)
    source_names = (*study.SOURCES, "build_crop_context_r71_notebook.py", "build_inference_ablation_r53_notebooks.py")
    sources = {"scripts/" + name: hashlib.sha256((root / "scripts" / name).read_bytes()).hexdigest()
               for name in source_names}
    fixed = [study.PROTOCOL_PATH, *(str(Path(study.CONTROL_DIRECTORY) / name) for name in study.CONTROL_FILES)]
    fixed_hashes = {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in fixed}
    sparse = ["/scripts/", "/" + current.R68_PATH,
              "/" + str(Path(current.REFERENCE_PATH).parent) + "/", "/" + current.PROVENANCE_PATH,
              *("/" + name for name in fixed)]
    cells = [cell("markdown", """
        # Fixed crop context and scale comparison

        **CPU only. Run all four code cells. No training is used.**

        This comparison uses the frozen PR158 reviewed-label checkpoint and the
        same 20 development photographs. It first reproduces the current
        full-image original/reflected method. Each of two fixed candidates then
        adds the same two overlapping full-height crops, spanning columns
        0–768 and 512–1280. Input 768 keeps the crop's original pixel scale.
        This changes context and adds candidate exposure. Input 1280 also
        enlarges those same crops by 5/3.

        A crop box touching an artificial side boundary is excluded. A corner
        outside its crop before native coordinate clipping is marked unavailable.
        Every retained pose record is mapped to the original photograph. The
        established score-ranked suppression rule selects whole records from the
        baseline and both crops. It does not average new coordinates or tune scores.

        The four references use fresh matching, with r67 primary. The return
        includes misses, extras, duplicates, complete pairs, known corners on
        partial fronts, empty photographs and every decision's source record.
        Both candidate results are retained. This development comparison does
        not select a deployment default.
        """), cell("markdown", """
        ## 1 Load the immutable source and fixed protocol

        Source, verified inputs and outputs use separate directories. Existing
        edited source is preserved. Choose a fresh `BAOMAI_CROP_CONTEXT_BASE` to
        run beside it. The source modules, protocol and saved CPU control are
        verified before inference.
        """), cell("code", f'''
        from pathlib import Path
        import hashlib, json, os, subprocess, sys

        SOURCE_COMMIT = "{source_commit}"
        SOURCE_HASHES = {sources!r}
        FIXED_HASHES = {fixed_hashes!r}
        BASE = Path(os.environ.get("BAOMAI_CROP_CONTEXT_BASE", "/content/baomai-crop-context-r71")).resolve()
        SOURCE, WORK = BASE / "source", BASE / "work"
        INPUT_DIR, OUTPUT = WORK / "inputs", WORK / "crop-context"
        BASE.mkdir(parents=True, exist_ok=True)
        ENV = {{**os.environ, "GIT_LFS_SKIP_SMUDGE": "1", "GIT_TERMINAL_PROMPT": "0", "PYTHONDONTWRITEBYTECODE": "1"}}
        def run(command, cwd=None):
            subprocess.run(list(map(str, command)), cwd=cwd, env=ENV, check=True)
        def git(*args):
            return subprocess.check_output(["git", *args], cwd=SOURCE, env=ENV, text=True).strip()
        if not (SOURCE / ".git").exists():
            if SOURCE.exists() and any(SOURCE.iterdir()):
                raise RuntimeError("Existing files preserved. Choose a fresh BAOMAI_CROP_CONTEXT_BASE.")
            run(["git", "init", SOURCE])
            run(["git", "remote", "add", "origin", "https://github.com/Siuuuers/dl.git"], SOURCE)
            run(["git", "fetch", "--depth=1", "--filter=blob:none", "origin", SOURCE_COMMIT], SOURCE)
            run(["git", "sparse-checkout", "init", "--no-cone"], SOURCE)
            run(["git", "sparse-checkout", "set", *{sparse!r}], SOURCE)
            run(["git", "checkout", "--detach", SOURCE_COMMIT], SOURCE)
        def verify_source():
            if git("rev-parse", "HEAD") != SOURCE_COMMIT or git("status", "--porcelain", "--untracked-files=all"):
                raise RuntimeError("Source changed. Choose a fresh BAOMAI_CROP_CONTEXT_BASE.")
            for name, expected in {{**SOURCE_HASHES, **FIXED_HASHES}}.items():
                if hashlib.sha256((SOURCE / name).read_bytes()).hexdigest() != expected:
                    raise RuntimeError("Pinned file differs: " + name)
        verify_source()
        WORK.mkdir(parents=True, exist_ok=True)
        os.environ["YOLO_CONFIG_DIR"] = str(WORK / "runtime-config")
        ENV["YOLO_CONFIG_DIR"] = os.environ["YOLO_CONFIG_DIR"]
        print("Immutable source:", SOURCE_COMMIT)
        '''), cell("markdown", """
        ## 2 Verify the runtime, checkpoint, photographs and reference inputs

        Existing Torch and torchvision installations are retained. CPU packages
        are installed if absent. Ultralytics, NumPy, OpenCV and Pillow use the
        recorded versions. Critical native source files are checked, and the
        actual runtime is recorded. Reuse verified cached inputs or download
        them from the immutable project paths.
        """), cell("code", '''
        from importlib.util import find_spec
        from importlib.metadata import version, PackageNotFoundError
        if find_spec("torch") is None or find_spec("torchvision") is None:
            run([sys.executable, "-m", "pip", "install", "--quiet", "--index-url",
                 "https://download.pytorch.org/whl/cpu", "torch==2.10.0+cpu", "torchvision==0.25.0+cpu"])
        required = {"ultralytics": "8.4.182", "numpy": "2.4.6", "opencv-python": "5.0.0.93", "Pillow": "13.0.0"}
        def installed(name):
            try:
                return version(name)
            except PackageNotFoundError:
                return None
        missing = [name + "==" + expected for name, expected in required.items() if installed(name) != expected]
        if missing:
            constraints = WORK / "torch-constraints.txt"
            constraints.write_text("".join(name + "==" + version(name) + "\\n" for name in ("torch", "torchvision")))
            run([sys.executable, "-m", "pip", "install", "--quiet", "--constraint", constraints, *missing])
        sys.dont_write_bytecode = True
        sys.path.insert(0, str(SOURCE / "scripts"))
        import analyze_crop_context_r71 as study
        plan = study.prepare(SOURCE, INPUT_DIR)
        runtime = study.current.native_runtime()
        print("Frozen checkpoint:", plan["checkpoint"])
        print("Verified photographs:", len(plan["images"]))
        print("Runtime packages:", runtime["packages"])
        '''), cell("markdown", """
        ## 3 Verify the control and run both fixed candidates

        The fresh full-image control must reproduce saved R69 CPU per-image
        matches, misses, extra and duplicate identities, and every known-corner
        five- and ten-pixel outcome. Numeric record differences are retained.
        A failed control stops the run before either crop candidate.

        The run makes 40 full-image calls and 80 crop calls. It records exact
        letterbox geometry, pre-clamp crop corners, artificial-edge decisions,
        source IDs, merged selections and fresh evaluations. The checkpoint and
        frozen model state must remain unchanged. All-negative photographs are
        included in each arm.
        """), cell("code", '''
        summary, records = study.analyze(SOURCE, INPUT_DIR, OUTPUT, plan)
        study.write_outputs(SOURCE, OUTPUT, summary, records)
        from IPython.display import Markdown, display
        display(Markdown((OUTPUT / "results.md").read_text()))
        print("Control gate:", summary["control_gate_passed"])
        print("Recorded comparisons:", summary["record_counts"])
        print("Crop decisions:", json.dumps(summary["crop_counts"], indent=2))
        '''), cell("markdown", """
        ## 4 Verify and download the evidence

        Recheck the immutable source and every saved member, then package the
        complete predictions and decision records. The archive contains both
        fixed candidate outcomes, the protocol, runtime and input identities.
        It contains no new model or training labels.
        """), cell("code", '''
        verify_source()
        RESULT = study.package_results(OUTPUT, SOURCE_COMMIT)
        manifest = {"source_commit": SOURCE_COMMIT, "notebook_code_cells": 4,
                    "fresh_inference": True, "training": False, "optimizer_updates": 0,
                    "labels_changed": False, "threshold_search": False, "automatic_promotion": False,
                    "control_gate_passed": summary["control_gate_passed"],
                    "matrix_rows": len(summary["matrix"]),
                    "result": {"filename": RESULT.name, "bytes": RESULT.stat().st_size,
                               "sha256": hashlib.sha256(RESULT.read_bytes()).hexdigest()}}
        (WORK / "package-manifest.json").write_text(json.dumps(manifest, indent=2) + "\\n")
        print(json.dumps(manifest, indent=2))
        from IPython.display import FileLink
        try:
            from google.colab import files
        except ImportError:
            display(FileLink(str(RESULT)))
        else:
            files.download(str(RESULT))
        ''')]
    return {"cells": cells, "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"}, "source_commit": source_commit,
        "source_hashes": sources, "fixed_input_hashes": fixed_hashes,
        "fresh_inference": True, "training_required": False}, "nbformat": 4, "nbformat_minor": 5}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    notebook = build(args.source_commit, root)
    for i, item in enumerate(notebook["cells"]):
        if item["cell_type"] == "code":
            compile("".join(item["source"]), f"cell-{i}", "exec")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(notebook, indent=1) + "\n")
    print(args.output, args.output.stat().st_size, "bytes, four code cells syntax checked")


if __name__ == "__main__":
    main()
