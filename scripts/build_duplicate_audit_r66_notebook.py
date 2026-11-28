#!/usr/bin/env python3
"""Build a four-cell CPU replay of duplicate and unmatched output probes."""

import argparse
import hashlib
import json
from pathlib import Path
import re

from analyze_duplicate_audit_r66 import HISTORICAL_PATH, SOURCES
from build_inference_ablation_r53_notebooks import cell


def build(source_commit, root):
    if not re.fullmatch("[0-9a-f]{40}", source_commit):
        raise ValueError("Use a complete published implementation commit")
    hashes = {name: hashlib.sha256((root / "scripts" / name).read_bytes()).hexdigest() for name in SOURCES}
    cells = [cell("markdown", """
        # Understand duplicate and unmatched cassette outputs

        **CPU only: run all four code cells.** This notebook replays completed
        predictions from the historical method and both arms of PR157 and
        PR158. No photographs, model weights, GPU, training or fresh neural
        inference are needed. Inputs total approximately 10.5 MB.

        Three separate fixed probes remove completed outputs: require an
        existing mutual reflected-view box association, or add suppression
        above box overlap 0.60 or 0.50. Retained boxes, corner coordinates and
        scores stay unchanged. Suppression is applied after the existing
        inference pipeline; it does not replace native NMS. The settings are
        illustrations of the duplicate-versus-missed-front tradeoff. None is
        selected or promoted.

        All three annotation references are reported. Unmatched outputs are
        evaluation outcomes, not automatically physical negatives. Complete-pair
        success is accompanied by detection of zero-, one- and two-known-corner
        fronts. The same 20 photographs remain development data.
        """), cell("markdown", """
        ## 1 Load the fixed implementation

        Source is pinned to an immutable commit. A changed source checkout is
        preserved and rejected; choose a new `BAOMAI_DUPLICATE_AUDIT_BASE` if
        rerunning beside edited files.
        """), cell("code", f'''
        from pathlib import Path
        import hashlib, json, os, subprocess, sys

        SOURCE_COMMIT = "{source_commit}"
        SOURCE_HASHES = {hashes!r}
        BASE = Path(os.environ.get("BAOMAI_DUPLICATE_AUDIT_BASE", "/content/baomai-duplicate-audit-r66")).resolve()
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
                raise RuntimeError("Existing files preserved; choose a fresh BAOMAI_DUPLICATE_AUDIT_BASE.")
            run(["git", "init", SOURCE])
            run(["git", "remote", "add", "origin", "https://github.com/Siuuuers/dl.git"], SOURCE)
            run(["git", "fetch", "--depth=1", "--filter=blob:none", "origin", SOURCE_COMMIT], SOURCE)
            run(["git", "sparse-checkout", "init", "--no-cone"], SOURCE)
            run(["git", "sparse-checkout", "set", "/scripts/", "/{HISTORICAL_PATH}"], SOURCE)
            run(["git", "checkout", "--detach", SOURCE_COMMIT], SOURCE)
        def verify_source():
            if git("rev-parse", "HEAD") != SOURCE_COMMIT or git("status", "--porcelain", "--untracked-files=all"):
                raise RuntimeError("Source changed; choose a fresh BAOMAI_DUPLICATE_AUDIT_BASE.")
            for name, expected in SOURCE_HASHES.items():
                if hashlib.sha256((SOURCE / "scripts" / name).read_bytes()).hexdigest() != expected:
                    raise RuntimeError("Implementation hash differs: " + name)
        verify_source()
        WORK.mkdir(parents=True, exist_ok=True)
        sys.dont_write_bytecode = True
        sys.path.insert(0, str(SOURCE / "scripts"))
        import analyze_duplicate_audit_r66 as audit
        from IPython.display import Markdown, display
        print("Immutable source:", SOURCE_COMMIT)
        '''), cell("markdown", """
        ## 2 Verify the three saved-input archives

        The historical reference archive is read from the pinned checkout.
        PR157 and PR158 archives are fetched at their exact return commits if
        not already present. Archive size, SHA-256, membership, CRC and every
        inventoried member are checked. A corrupt existing input is rejected.
        No checkpoint is downloaded or loaded.
        """), cell("code", '''
        audit.stage_inputs(SOURCE, INPUT_DIR)
        for name, spec in audit.INPUTS.items():
            print(name, spec["bytes"], "bytes; SHA-256", spec["sha256"])
        print("Three immutable prediction archives verified.")
        '''), cell("markdown", """
        ## 3 Replay the baselines and three fixed probes

        Archived predictions, view associations, scores and cases must replay
        exactly before the probes are scored. Each probe starts independently
        from its unchanged method; filters are not combined. Evaluation matches
        are recomputed after removal. Cross-view associations and averaged
        coordinates are not recomputed.

        The maximum-cardinality bipartite matching calculation is a diagnostic
        upper bound on how many reference matches the same output boxes permit.
        It is not an alternate published score and cannot be used at inference.
        """), cell("code", '''
        summary, records = audit.analyze(SOURCE, INPUT_DIR)
        audit.write_outputs(OUTPUT, summary, records)
        display(Markdown((OUTPUT / "results.md").read_text()))
        print("All-reference matrix rows:", len(summary["matrix"]))
        print("Archived per-image baseline scores replayed:", summary["provenance"]["replayed_image_scores"])
        print("Exact paired changes and removals:", summary["record_counts"])
        '''), cell("markdown", """
        ## 4 Verify the saved evidence and download

        Recompute the analysis and compare all saved result bytes. The ZIP
        contains the all-reference matrix, availability strata, changed target
        identities, removed-output records, baseline errors, retained original
        indices, matching bounds and source/input provenance.

        No labels or application defaults are changed. Fewer duplicates with
        more missed partial fronts is reported as a tradeoff. This completed
        CPU analysis does not provide independent capture-session performance.
        """), cell("code", '''
        verify_source()
        repeated_summary, repeated_records = audit.analyze(SOURCE, INPUT_DIR)
        if repeated_summary != summary or repeated_records != records:
            raise RuntimeError("Independent saved-input replay differs")
        CHECK = WORK / "verification"
        audit.write_outputs(CHECK, repeated_summary, repeated_records)
        inventory = json.loads((OUTPUT / "inventory.json").read_bytes())
        for name in [*inventory, "inventory.json"]:
            if (OUTPUT / name).read_bytes() != (CHECK / name).read_bytes():
                raise RuntimeError("Saved result bytes differ: " + name)
        RESULT = audit.package_results(OUTPUT, SOURCE_COMMIT)
        manifest = {"source_commit": SOURCE_COMMIT, "notebook_code_cells": 4,
                    "saved_output_replay": "byte-identical", "fresh_inference": False,
                    "training": False, "labels_modified": False, "automatic_promotion": False,
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
        "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}


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
