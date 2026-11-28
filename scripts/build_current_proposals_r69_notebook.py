#!/usr/bin/env python3
"""Build one four-cell CPU notebook for current proposals and fixed corner suppression."""

import argparse
import hashlib
import json
from pathlib import Path
import re

from build_inference_ablation_r53_notebooks import cell
from diagnose_current_proposals_r69 import (
    COMPANION_PROTOCOL, COMPANION_SCRIPT, PROTOCOL_PATH, PROVENANCE_PATH,
    R68_PATH, REFERENCE_PATH, SOURCES,
)


def build(source_commit, root):
    if not re.fullmatch("[0-9a-f]{40}", source_commit):
        raise ValueError("A complete published implementation commit is required")
    sources = {name: hashlib.sha256((root / "scripts" / name).read_bytes()).hexdigest() for name in SOURCES}
    protocols = {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in (PROTOCOL_PATH, COMPANION_PROTOCOL)}
    cells = [cell("markdown", """
        # Current proposals and one fixed corner-suppression experiment

        **CPU only. Run all four code cells. No optimizer or training is used.**
        The first analysis runs fresh inference with the frozen PR158 reviewed-label
        checkpoint on the same 20 development photographs, in both original and
        reflected views. It captures every decoded FP32 box, score and corner before
        confidence filtering and native suppression, then verifies the exact native
        reconstruction and records candidate lineage.

        The second analysis uses the previously saved R68 outputs for five methods.
        It compares unchanged outputs, additional box suppression at IoU 0.60, and
        the same fixed rule qualified by two accepted corresponding corners within
        five pixels. It has a separate protocol and produces separate evidence.
        The qualified rule is also checked on the fresh current-checkpoint outputs.

        Both analyses retain r50, r67, supplied664 and frozen666 references, with r67
        primary. Thresholds, labels and model weights are unchanged. Selection uses
        predictions alone; reference-based candidate inspection is diagnostic.
        Results describe development evidence and do not automatically promote a
        model or postprocessing rule.

        Inputs include one approximately 21.7 MB checkpoint, 14.9 MB of original
        photographs and about 4.3 MB of archived evidence. The raw return preserves
        about 34 MB of FP32 tensor values using lossless compression.
        """), cell("markdown", """
        ## 1 Load the published source and both fixed protocols

        Source, cached inputs and outputs remain separate. An edited source checkout
        is preserved and rejected; choose another `BAOMAI_CURRENT_PROPOSALS_BASE`
        to run beside it. Every project module and both protocols are hash-checked.
        """), cell("code", f'''
        from pathlib import Path
        import hashlib, json, os, subprocess, sys

        SOURCE_COMMIT = "{source_commit}"
        SOURCE_HASHES = {sources!r}
        PROTOCOL_HASHES = {protocols!r}
        BASE = Path(os.environ.get("BAOMAI_CURRENT_PROPOSALS_BASE", "/content/baomai-current-proposals-r69")).resolve()
        SOURCE, WORK = BASE / "source", BASE / "work"
        INPUT_DIR, OUTPUT, COMPANION = WORK / "inputs", WORK / "current-proposals", WORK / "corner-suppression"
        BASE.mkdir(parents=True, exist_ok=True)
        ENV = {{**os.environ, "GIT_LFS_SKIP_SMUDGE": "1", "GIT_TERMINAL_PROMPT": "0", "PYTHONDONTWRITEBYTECODE": "1"}}
        def run(command, cwd=None):
            subprocess.run(list(map(str, command)), cwd=cwd, env=ENV, check=True)
        def git(*args):
            return subprocess.check_output(["git", *args], cwd=SOURCE, env=ENV, text=True).strip()
        if not (SOURCE / ".git").exists():
            if SOURCE.exists() and any(SOURCE.iterdir()):
                raise RuntimeError("Existing files preserved; choose a fresh BAOMAI_CURRENT_PROPOSALS_BASE.")
            run(["git", "init", SOURCE])
            run(["git", "remote", "add", "origin", "https://github.com/Siuuuers/dl.git"], SOURCE)
            run(["git", "fetch", "--depth=1", "--filter=blob:none", "origin", SOURCE_COMMIT], SOURCE)
            run(["git", "sparse-checkout", "init", "--no-cone"], SOURCE)
            run(["git", "sparse-checkout", "set", "/scripts/", "/{R68_PATH}",
                 "/{str(Path(REFERENCE_PATH).parent)}/", "/{PROVENANCE_PATH}",
                 "/{PROTOCOL_PATH}", "/{COMPANION_PROTOCOL}"], SOURCE)
            run(["git", "checkout", "--detach", SOURCE_COMMIT], SOURCE)
        def verify_source():
            if git("rev-parse", "HEAD") != SOURCE_COMMIT or git("status", "--porcelain", "--untracked-files=all"):
                raise RuntimeError("Source changed; choose a fresh BAOMAI_CURRENT_PROPOSALS_BASE.")
            for name, expected in SOURCE_HASHES.items():
                if hashlib.sha256((SOURCE / "scripts" / name).read_bytes()).hexdigest() != expected:
                    raise RuntimeError("Implementation hash differs: " + name)
            for name, expected in PROTOCOL_HASHES.items():
                if hashlib.sha256((SOURCE / name).read_bytes()).hexdigest() != expected:
                    raise RuntimeError("Fixed protocol changed: " + name)
        verify_source()
        WORK.mkdir(parents=True, exist_ok=True)
        os.environ["YOLO_CONFIG_DIR"] = str(WORK / "runtime-config")
        ENV["YOLO_CONFIG_DIR"] = os.environ["YOLO_CONFIG_DIR"]
        print("Immutable source:", SOURCE_COMMIT)
        '''), cell("markdown", """
        ## 2 Verify the runtime, checkpoint, photographs and references

        Existing Torch and torchvision installations are retained. CPU packages are
        installed only if they are absent. The pinned Ultralytics, NumPy, OpenCV and
        Pillow versions are installed only when missing or different. Critical
        native source files are checked before inference; all actual versions are
        recorded, including CPU versus the archived L4 environment.

        Reuse verified input files when present. Otherwise obtain the checkpoint
        parts and photographs from immutable GitHub paths, and verify complete
        hashes. No training label or reference file is written.
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
        import diagnose_current_proposals_r69 as audit
        plan = audit.prepare(SOURCE, INPUT_DIR)
        runtime = audit.native_runtime()
        print("Frozen checkpoint:", plan["checkpoint"])
        print("Verified photographs:", len(plan["images"]))
        print("Runtime packages:", runtime["packages"])
        '''), cell("markdown", """
        ## 3 Collect all proposals and run the two bounded analyses

        Capture both views at the unchanged confidence/NMS operating point. Preserve
        raw FP32 tensors and exact coordinate conversion, native returned indices,
        rank/tie witnesses and postclip lineage. The frozen model state must remain
        unchanged after normal inference warmup. Compare fresh CPU outputs with the
        archived L4 results while retaining numerical and scored-outcome differences.

        Diagnose first-lost stages, view support, fixed-pool coverage and duplicate
        competitions. Candidate box quality and confidence-qualified known-corner
        errors remain separate. Then run the companion's fixed saved-output rule and
        verify its reviewed-checkpoint result against the fresh-runtime rule check.
        """), cell("code", f'''
        summary, records = audit.analyze(SOURCE, INPUT_DIR, OUTPUT, plan)
        audit.write_outputs(SOURCE, OUTPUT, summary, records)
        run([sys.executable, "-B", SOURCE / "scripts/{COMPANION_SCRIPT}",
             "--r68-result-zip", INPUT_DIR / audit.INPUTS["r68"]["filename"], "--output", COMPANION])
        audit.verify_companion_continuity(records["qualified-rule-runtime-continuity"], COMPANION)
        from IPython.display import Markdown, display
        display(Markdown((OUTPUT / "results.md").read_text()))
        display(Markdown((COMPANION / "results.md").read_text()))
        print("Raw diagnostic records:", summary["record_counts"])
        print("Captured raw archive:", summary["decoded_archive"])
        '''), cell("markdown", """
        ## 4 Verify and download the complete evidence

        Recheck source and every result member. Package the lossless raw tensors,
        stage diagnostics, native/final comparisons and reference records, plus the
        companion results under a separate `corner-suppression/` prefix. Input and
        runtime identities remain explicit. No model or threshold is promoted.
        """), cell("code", '''
        verify_source()
        RESULT = audit.package_results(OUTPUT, SOURCE_COMMIT, COMPANION)
        companion_summary = json.loads((COMPANION / "summary.json").read_bytes())
        manifest = {"source_commit": SOURCE_COMMIT, "notebook_code_cells": 4,
                    "fresh_inference": True, "training": False, "optimizer_updates": 0,
                    "labels_changed": False, "threshold_search": False, "automatic_promotion": False,
                    "raw_diagnostic_matrix_rows": len(summary["matrix"]),
                    "companion_matrix_rows": len(companion_summary["matrix"]),
                    "companion_archived_continuity_verified": True,
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
        "source_hashes": sources, "protocol_hashes": protocols,
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
    print(args.output, args.output.stat().st_size, "bytes; four code cells syntax checked")


if __name__ == "__main__":
    main()
