#!/usr/bin/env python3
"""Build the four-cell CPU partial-front diagnostic and fixed union notebook."""

import argparse
import hashlib
import json
from pathlib import Path
import re

from build_inference_ablation_r53_notebooks import cell
from analyze_partial_detection_r68 import (
    HISTORICAL_PATH, PROTOCOL_PATH, PROTOCOL_SHA, PROVENANCE_PATH, R67_PATH,
    REFERENCE_PATH, SOURCES,
)


def build(source_commit, root):
    if not re.fullmatch("[0-9a-f]{40}", source_commit):
        raise ValueError("A complete published implementation commit is required")
    hashes = {name: hashlib.sha256((root / "scripts" / name).read_bytes()).hexdigest() for name in SOURCES}
    cells = [cell("markdown", """
        # Partial-front stages and one reflected-output union

        **CPU only. Run all four code cells.** This notebook uses about 27.3 MB of
        immutable archived predictions and labels. No photographs, model weights,
        GPU, training or new neural inference are required. NumPy is the only
        numerical dependency and is installed only if it is missing.

        First, trace three restored partial-front misses through the saved stages
        actually available for each checkpoint. Complete pre-confidence/pre-NMS
        tensors exist only for the historical adapted original view. Newer caches
        are already filtered; their limits remain explicit.

        Then replay one fixed hypothesis: keep the completed original detections,
        add all reflected outputs unused by the existing mutual associations,
        and run the existing final suppression rule. The rule is fixed before
        full scoring, uses no reference to choose additions, and is evaluated
        across five methods and four references. The three focused restorations
        cannot be recovered by this union; other missed fronts justify the probe.

        This is development evidence. No thresholds, labels, scores, coordinates
        or model weights are tuned, and no method is automatically promoted.
        """), cell("markdown", """
        ## 1 Load the immutable implementation

        Source stays separate from cached inputs and generated outputs. A changed
        source checkout is preserved and rejected; choose another
        `BAOMAI_PARTIAL_DETECTION_BASE` to run beside an edited copy.
        """), cell("code", f'''
        from pathlib import Path
        import hashlib, json, os, subprocess, sys

        SOURCE_COMMIT = "{source_commit}"
        SOURCE_HASHES = {hashes!r}
        BASE = Path(os.environ.get("BAOMAI_PARTIAL_DETECTION_BASE", "/content/baomai-partial-detection-r68")).resolve()
        SOURCE, WORK = BASE / "source", BASE / "work"
        INPUT_DIR, OUTPUT = WORK / "inputs", WORK / "results"
        BASE.mkdir(parents=True, exist_ok=True)
        ENV = {{**os.environ, "GIT_LFS_SKIP_SMUDGE": "1", "GIT_TERMINAL_PROMPT": "0", "PYTHONDONTWRITEBYTECODE": "1"}}
        def run(command, cwd=None):
            subprocess.run(list(map(str, command)), cwd=cwd, env=ENV, check=True)
        def git(*args):
            return subprocess.check_output(["git", *args], cwd=SOURCE, env=ENV, text=True).strip()
        if not (SOURCE / ".git").exists():
            if SOURCE.exists() and any(SOURCE.iterdir()):
                raise RuntimeError("Existing files preserved; choose a fresh BAOMAI_PARTIAL_DETECTION_BASE.")
            run(["git", "init", SOURCE])
            run(["git", "remote", "add", "origin", "https://github.com/Siuuuers/dl.git"], SOURCE)
            run(["git", "fetch", "--depth=1", "--filter=blob:none", "origin", SOURCE_COMMIT], SOURCE)
            run(["git", "sparse-checkout", "init", "--no-cone"], SOURCE)
            run(["git", "sparse-checkout", "set", "/scripts/", "/{HISTORICAL_PATH}", "/{R67_PATH}",
                 "/{str(Path(REFERENCE_PATH).parent)}/", "/{PROVENANCE_PATH}", "/{PROTOCOL_PATH}"], SOURCE)
            run(["git", "checkout", "--detach", SOURCE_COMMIT], SOURCE)
        def verify_source():
            if git("rev-parse", "HEAD") != SOURCE_COMMIT or git("status", "--porcelain", "--untracked-files=all"):
                raise RuntimeError("Source changed; choose a fresh BAOMAI_PARTIAL_DETECTION_BASE.")
            for name, expected in SOURCE_HASHES.items():
                if hashlib.sha256((SOURCE / "scripts" / name).read_bytes()).hexdigest() != expected:
                    raise RuntimeError("Implementation hash differs: " + name)
            if hashlib.sha256((SOURCE / "{PROTOCOL_PATH}").read_bytes()).hexdigest() != "{PROTOCOL_SHA}":
                raise RuntimeError("Fixed protocol changed.")
        verify_source()
        WORK.mkdir(parents=True, exist_ok=True)
        print("Immutable source:", SOURCE_COMMIT)
        '''), cell("markdown", """
        ## 2 Verify the saved inputs and fixed protocol

        Archive hashes, sizes, CRC, member inventories, reference lineage and
        source hashes must agree. Cached files are reused. Historical labels
        and training labels remain untouched. Existing compatible NumPy is used.
        """), cell("code", '''
        import importlib.util
        if importlib.util.find_spec("numpy") is None:
            run([sys.executable, "-m", "pip", "install", "--quiet", "numpy==2.3.5"])
        import numpy as np
        sys.dont_write_bytecode = True
        sys.path.insert(0, str(SOURCE / "scripts"))
        import analyze_partial_detection_r68 as audit
        audit.stage_inputs(SOURCE, INPUT_DIR)
        protocol = json.loads((SOURCE / audit.PROTOCOL_PATH).read_bytes())
        print("Verified immutable archives:", len(audit.INPUTS))
        print("NumPy:", np.__version__)
        print("Fixed comparison:", protocol["variant"], "across", len(protocol["methods"]), "methods and", len(protocol["references"]), "references")
        print("Full raw pool: historical adapted original view only; current caches are post-native-NMS.")
        '''), cell("markdown", """
        ## 3 Trace the focused misses and evaluate the fixed union

        Reproduce archived baselines and the existing historical CPU/GPU identity
        bridge. Report best-overlap box geometry, actual stage support and a
        finite-pool confidence-only bound. Compare the focused sizes with the
        unchanged training-label distribution, without equating size and visual
        similarity.

        The union adds only unused reflected outputs. An existing mutual box link
        remains used when its corner pair is unavailable. Keep all original boxes,
        scores and completed averaged corners, then suppress same-class overlap
        strictly above 0.70 in score order. Fresh matching records additions,
        removed originals, gains, losses, rematches and known-corner outcomes.
        """), cell("code", '''
        summary, records = audit.analyze(SOURCE, INPUT_DIR)
        audit.write_outputs(OUTPUT, summary, records, SOURCE)
        from IPython.display import Markdown, display
        display(Markdown((OUTPUT / "results.md").read_text()))
        print("Fixed comparison rows:", len(summary["matrix"]))
        print("Focused stage counts:", summary["focused_stage_counts"])
        print("Evidence records:", summary["record_counts"])
        '''), cell("markdown", """
        ## 4 Verify and package the evidence

        Recheck the immutable source and every saved result member. The package
        includes unchanged/union outputs, source-view identities, all four label
        references, the complete comparison matrix, stage evidence and paired
        change ledgers. This records one development experiment; it selects no
        threshold or production method.
        """), cell("code", '''
        verify_source()
        RESULT = audit.package_results(OUTPUT, SOURCE_COMMIT)
        manifest = {"source_commit": SOURCE_COMMIT, "notebook_code_cells": 4,
                    "archived_baseline_replay": "exact", "fresh_inference": False,
                    "training": False, "labels_changed": False, "threshold_search": False,
                    "comparison_rows": len(summary["matrix"]), "automatic_promotion": False,
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
        "source_hashes": hashes, "fresh_inference": False, "training_required": False},
        "nbformat": 4, "nbformat_minor": 5}


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
