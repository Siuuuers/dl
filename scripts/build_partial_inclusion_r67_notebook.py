#!/usr/bin/env python3
"""Build a four-cell CPU replay of the accepted partial-front reference additions."""

import argparse
import hashlib
import json
from pathlib import Path
import re

from build_inference_ablation_r53_notebooks import cell
from replay_partial_inclusion_r67 import HISTORICAL_PATH, PROVENANCE_PATH, REFERENCE_PATH, SOURCES


def build(source_commit, root):
    if not re.fullmatch("[0-9a-f]{40}", source_commit):
        raise ValueError("Use a complete published implementation commit")
    hashes = {name: hashlib.sha256((root / "scripts" / name).read_bytes()).hexdigest() for name in SOURCES}
    candidate_directory = str(Path(REFERENCE_PATH).parent)
    cells = [cell("markdown", """
        # Restore four reviewed partial fronts and replay the same predictions

        **CPU only: run all four code cells.** This notebook compares the frozen
        r50 evaluation reference with the separately versioned r67 reference.
        Four justified partial-front rows are restored using exact historical
        coordinates and corner-availability flags. Existing rows retain their
        bytes and positions. Three uncertain candidates remain held.

        The supplied original model, historical retained method and both arms
        of PR157 and PR158 use their unchanged saved predictions. No photographs,
        model weights, GPU, training or fresh neural inference are needed.
        Standard-library analysis uses approximately 10.5 MB of archived inputs.

        This measures a reference-population change, not learned improvement.
        The same 20 photographs remain development evidence. The new reference
        has 658 fronts and 583 complete corner pairs; partial-front detection is
        reported separately. No ignore-region mechanism is introduced.
        """), cell("markdown", """
        ## 1 Load the fixed implementation and reference

        Source and candidate labels come from an immutable commit. Existing
        modified files are preserved and rejected; choose another
        `BAOMAI_PARTIAL_INCLUSION_BASE` to run beside an edited checkout.
        """), cell("code", f'''
        from pathlib import Path
        import hashlib, json, os, subprocess, sys

        SOURCE_COMMIT = "{source_commit}"
        SOURCE_HASHES = {hashes!r}
        BASE = Path(os.environ.get("BAOMAI_PARTIAL_INCLUSION_BASE", "/content/baomai-partial-inclusion-r67")).resolve()
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
                raise RuntimeError("Existing files preserved; choose a fresh BAOMAI_PARTIAL_INCLUSION_BASE.")
            run(["git", "init", SOURCE])
            run(["git", "remote", "add", "origin", "https://github.com/Siuuuers/dl.git"], SOURCE)
            run(["git", "fetch", "--depth=1", "--filter=blob:none", "origin", SOURCE_COMMIT], SOURCE)
            run(["git", "sparse-checkout", "init", "--no-cone"], SOURCE)
            run(["git", "sparse-checkout", "set", "/scripts/", "/{HISTORICAL_PATH}",
                 "/{candidate_directory}/", "/{PROVENANCE_PATH}"], SOURCE)
            run(["git", "checkout", "--detach", SOURCE_COMMIT], SOURCE)
        def verify_source():
            if git("rev-parse", "HEAD") != SOURCE_COMMIT or git("status", "--porcelain", "--untracked-files=all"):
                raise RuntimeError("Source changed; choose a fresh BAOMAI_PARTIAL_INCLUSION_BASE.")
            for name, expected in SOURCE_HASHES.items():
                if hashlib.sha256((SOURCE / "scripts" / name).read_bytes()).hexdigest() != expected:
                    raise RuntimeError("Implementation hash differs: " + name)
        verify_source()
        WORK.mkdir(parents=True, exist_ok=True)
        sys.dont_write_bytecode = True
        sys.path.insert(0, str(SOURCE / "scripts"))
        import replay_partial_inclusion_r67 as audit
        from IPython.display import Markdown, display
        print("Immutable source:", SOURCE_COMMIT)
        '''), cell("markdown", """
        ## 2 Verify saved inputs and append-only reference lineage

        Verify archive sizes, hashes, membership, CRC and all member inventories.
        Check each of the 20 candidate label files against archived r50 bytes,
        the four accepted historical lines and the three explicit holds.
        No historical reference or training-label file is written.
        """), cell("code", '''
        audit.stage_inputs(SOURCE, INPUT_DIR)
        spec = audit.INPUTS["historical"]
        old = audit.verified_zip(INPUT_DIR / spec["filename"], spec["sha256"], spec["bytes"])
        _, _, lineage, reference_manifest, _ = audit.candidate_reference(
            SOURCE, audit.read(old, "reference-labels.json.gz")["r50"])
        print("Three immutable prediction archives verified.")
        print("Candidate population:", reference_manifest["expected_counts"])
        for row in reference_manifest["restored_rows"]:
            print(row["stable_id"], "->", row["candidate_reference_id"])
        print("Held candidates:", [row["stable_id"] for row in reference_manifest["held_cases"]])
        '''), cell("markdown", """
        ## 3 Replay baselines and measure reference-only changes

        Archived predictions, view associations, scores and cases must reproduce
        exactly. Each of the six r50 primary-method baselines is then checked
        against its saved per-image scores before fresh matching against r67.

        Keep all output records and coordinates unchanged. Report restored
        targets, any retained-target reassignment, output-category transitions,
        and detection with zero, one or two known corners alongside pair scores.

        A separate eight-row supplement repeats the three previously fixed r66
        output restrictions for the PR158 reviewed-label arm only. It checks
        whether a filter removes an output now supported by a restored front.
        It does not replace native NMS or choose a threshold.
        """), cell("code", '''
        summary, records = audit.analyze(SOURCE, INPUT_DIR)
        audit.write_outputs(OUTPUT, summary, records)
        display(Markdown((OUTPUT / "results.md").read_text()))
        print("Result rows:", len(summary["matrix"]))
        print("Archived score objects replayed:", summary["provenance"]["replayed_image_scores"])
        print("Reference-change ledgers:", summary["record_counts"])
        '''), cell("markdown", """
        ## 4 Verify saved evidence and download

        Verify every result member and package the matrix, full measurements,
        per-image scores, exact before/after labels, target lineage, restored
        outcomes and unchanged predictions. Source and input identities remain
        explicit. No model or threshold is selected or promoted.
        """), cell("code", '''
        verify_source()
        RESULT = audit.package_results(OUTPUT, SOURCE_COMMIT)
        manifest = {"source_commit": SOURCE_COMMIT, "notebook_code_cells": 4,
                    "r50_baseline_replay": "exact", "fresh_inference": False,
                    "training": False, "predictions_modified": False,
                    "historical_references_modified": False, "training_labels_modified": False,
                    "new_reference": reference_manifest["expected_counts"],
                    "automatic_promotion": False,
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
    for index, item in enumerate(notebook["cells"]):
        if item["cell_type"] == "code":
            compile("".join(item["source"]), f"cell-{index}", "exec")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(notebook, indent=1) + "\n")
    print(args.output, args.output.stat().st_size, "bytes; four code cells syntax checked")


if __name__ == "__main__":
    main()
