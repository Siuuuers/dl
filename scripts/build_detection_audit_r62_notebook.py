#!/usr/bin/env python3
"""Build the small CPU detection audit and reflected-box probe notebook."""

import argparse
import hashlib
import json
from pathlib import Path
import re

from build_inference_ablation_r53_notebooks import cell


ARCHIVE = "docs/thesis/evidence/annotation-reference/baomai-annotation-reference-r60-results.zip"
DRIVERS = ("analyze_detection_errors_r62.py", "probe_reflected_anchor_r62.py")


def build(source_commit, root):
    if re.fullmatch("[0-9a-f]{40}", source_commit) is None:
        raise ValueError("A complete published source commit is required")
    hashes = {name: hashlib.sha256((root / "scripts" / name).read_bytes()).hexdigest() for name in DRIVERS}
    cells = [cell("markdown", """
        # Understand the remaining cassette detections

        **CPU only. Run all four code cells.** This notebook reproduces a completed
        analysis using saved predictions from the same 20 development photographs.
        It needs no GPU, image download, model download or training.

        First it explains which outputs are extra, which are duplicate, and which
        missed fronts have incomplete corner annotations. Then it tests one fixed
        alternative: retain the reflected view's boxes and use the same equal
        corner averaging. It does not search thresholds or change annotations.

        Overlap groups are review cues, not proof that an output is background.
        The different annotation references describe the same photographs. This
        notebook does not replace the pending classification-loss or training-label
        experiments, and it does not select an application model.
        """), cell("markdown", """
        ## 1 Load the fixed implementation

        The source is pinned to a published commit and kept separate from output.
        If this directory already contains edited source, choose another
        `BAOMAI_DETECTION_AUDIT_BASE` location.
        """), cell("code", f'''
        from pathlib import Path
        import hashlib, json, os, subprocess, sys

        SOURCE_COMMIT = "{source_commit}"
        DRIVER_HASHES = {hashes!r}
        BASE = Path(os.environ.get("BAOMAI_DETECTION_AUDIT_BASE", "/content/baomai-detection-audit-r62")).resolve()
        SOURCE, WORK = BASE / "source", BASE / "work"
        BASE.mkdir(parents=True, exist_ok=True)
        ENV = {{**os.environ, "GIT_LFS_SKIP_SMUDGE": "1", "GIT_TERMINAL_PROMPT": "0", "PYTHONDONTWRITEBYTECODE": "1"}}
        def run(command, cwd=None):
            subprocess.run(list(map(str, command)), cwd=cwd, env=ENV, check=True)
        def git(*args):
            return subprocess.check_output(["git", *args], cwd=SOURCE, env=ENV, text=True).strip()
        if not (SOURCE / ".git").exists():
            if SOURCE.exists() and any(SOURCE.iterdir()):
                raise RuntimeError("Existing source preserved; choose a fresh BAOMAI_DETECTION_AUDIT_BASE.")
            run(["git", "init", SOURCE])
            run(["git", "remote", "add", "origin", "https://github.com/Siuuuers/dl.git"], SOURCE)
            run(["git", "fetch", "--depth=1", "--filter=blob:none", "origin", SOURCE_COMMIT], SOURCE)
            run(["git", "sparse-checkout", "init", "--no-cone"], SOURCE)
            run(["git", "sparse-checkout", "set", "/scripts/", "/{ARCHIVE}"], SOURCE)
            run(["git", "checkout", "--detach", SOURCE_COMMIT], SOURCE)
        if git("rev-parse", "HEAD") != SOURCE_COMMIT or git("status", "--porcelain", "--untracked-files=all"):
            raise RuntimeError("Changed source preserved; choose a fresh BAOMAI_DETECTION_AUDIT_BASE.")
        for name, expected in DRIVER_HASHES.items():
            if hashlib.sha256((SOURCE / "scripts" / name).read_bytes()).hexdigest() != expected:
                raise RuntimeError("Source identity differs: " + name)
        WORK.mkdir(parents=True, exist_ok=True)
        print("Published source:", SOURCE_COMMIT)
        '''), cell("markdown", """
        ## 2 Reproduce and partition the current errors

        The fixed archive and all its members are checked. The original scores,
        associations and predictions must reproduce exactly before new counts
        are reported. Duplicate outputs remain a subset of extras. The two other
        groups use maximum reference IoU below 0.10 or from 0.10 to below 0.50.

        Missing corners do not remove their fronts from detection evaluation.
        Saved native outputs already passed confidence 0.25 and ordinary NMS,
        so this cache cannot identify every reason a lower-scored or suppressed
        proposal is absent.
        """), cell("code", '''
        AUDIT = WORK / "detection-errors"
        run([sys.executable, "-B", SOURCE / "scripts/analyze_detection_errors_r62.py",
             "--root", SOURCE, "--output", AUDIT])
        audit = json.loads((AUDIT / "summary.json").read_text())
        from IPython.display import Markdown, display
        display(Markdown((AUDIT / "results.md").read_text()))
        print("Review records:", audit["review_case_count"])
        '''), cell("markdown", """
        ## 3 Reproduce the one fixed reflected-box experiment

        Keep reflected-view boxes, scores and identities. Associate each with
        the original view using mutual best predicted-box IoU at least 0.50;
        average the two corresponding corners equally when both pairs are
        available. Otherwise retain the reflected record. No reference label
        chooses the associations or weights.

        The detection confidence stays strictly above 0.25, NMS overlap at 0.70,
        reference-matching IoU at least 0.50 and corner confidence at least 0.50.
        All three reference versions are reported, including adverse results.
        """), cell("code", '''
        PROBE = WORK / "reflected-anchor"
        run([sys.executable, "-B", SOURCE / "scripts/probe_reflected_anchor_r62.py",
             "--root", SOURCE, "--work", PROBE])
        probe = json.loads((PROBE / "summary.json").read_text())
        rows = ["| Reference | Method | Misses | Extras | Duplicates | Both within 5 pixels | Both within 10 pixels |",
                "| --- | --- | ---: | ---: | ---: | ---: | ---: |"]
        for r in probe["matrix"]:
            rows.append(f"| {r['reference']} | {r['method']} | {r['missed']} | {r['extras']} | {r['duplicates']} | {r['within5']}/{r['eligible_pairs']} | {r['within10']}/{r['eligible_pairs']} |")
        display(Markdown("\\n".join(rows)))
        print("This exploratory choice has tradeoffs; no model is promoted.")
        '''), cell("markdown", """
        ## 4 Verify the saved result and download

        Recompute both analyses from the fixed input and compare every saved
        result. The compact ZIP preserves summary tables, exact review cases,
        paired changes, view associations and source identities. The analysis
        is already completed; rerunning is optional.
        """), cell("code", '''
        import gzip
        from zipfile import ZipFile, ZipInfo, ZIP_DEFLATED
        sys.dont_write_bytecode = True
        sys.path.insert(0, str(SOURCE / "scripts"))
        from analyze_detection_errors_r62 import analyze as analyze_errors, markdown as audit_markdown
        from probe_reflected_anchor_r62 import analyze as analyze_probe, ARCHIVE_PATH

        if git("rev-parse", "HEAD") != SOURCE_COMMIT or git("status", "--porcelain", "--untracked-files=all"):
            raise RuntimeError("Source changed after setup")
        for name, expected in DRIVER_HASHES.items():
            if hashlib.sha256((SOURCE / "scripts" / name).read_bytes()).hexdigest() != expected:
                raise RuntimeError("Source identity differs: " + name)
        replay_audit, replay_cases = analyze_errors(SOURCE)
        if replay_audit != json.loads((AUDIT / "summary.json").read_text()) or replay_cases != json.loads((AUDIT / "cases.json").read_text()):
            raise RuntimeError("Saved error audit differs from replay")
        if (AUDIT / "results.md").read_text() != audit_markdown(replay_audit):
            raise RuntimeError("Saved audit table differs")
        replay_probe, replay_records = analyze_probe(SOURCE / ARCHIVE_PATH, SOURCE)
        if replay_probe != json.loads((PROBE / "summary.json").read_text()):
            raise RuntimeError("Saved reflected-box summary differs from replay")
        for name, value in replay_records.items():
            if json.loads(gzip.decompress((PROBE / (name + ".json.gz")).read_bytes())) != value:
                raise RuntimeError("Saved probe record differs: " + name)
        probe_members = ["summary.json", *[name + ".json.gz" for name in replay_records]]
        saved_inventory = json.loads((PROBE / "inventory.json").read_text())
        actual_inventory = {name: {"bytes": (PROBE / name).stat().st_size,
                                  "sha256": hashlib.sha256((PROBE / name).read_bytes()).hexdigest()}
                            for name in probe_members}
        if saved_inventory != actual_inventory:
            raise RuntimeError("Saved probe inventory differs")
        payload = {}
        for folder, names in ((AUDIT, ["summary.json", "cases.json", "results.md"]),
                              (PROBE, [*probe_members, "inventory.json"])):
            for name in names:
                payload[folder.name + "/" + name] = (folder / name).read_bytes()
        manifest = {"source_commit": SOURCE_COMMIT, "drivers_sha256": DRIVER_HASHES,
                    "verification": "all saved analyses exactly replayed", "fresh_inference": False,
                    "training": False, "development_only": True,
                    "files": {name: {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
                              for name, data in payload.items()}}
        payload["manifest.json"] = (json.dumps(manifest, indent=2) + "\\n").encode()
        RESULT = WORK / "baomai-detection-audit-r62-results.zip"
        with ZipFile(RESULT, "w", ZIP_DEFLATED) as archive:
            for name, data in sorted(payload.items()):
                entry = ZipInfo(name, date_time=(2026, 1, 1, 0, 0, 0))
                entry.compress_type = ZIP_DEFLATED
                entry.external_attr = 0o100644 << 16
                archive.writestr(entry, data)
        with ZipFile(RESULT) as archive:
            if archive.testzip() is not None or any(archive.read(name) != data for name, data in payload.items()):
                raise RuntimeError("Saved result ZIP differs")
        print("All saved results verified.")
        print("Result:", RESULT.name, "bytes:", RESULT.stat().st_size)
        from IPython.display import FileLink
        try:
            from google.colab import files
        except ImportError:
            display(FileLink(str(RESULT)))
        else:
            files.download(str(RESULT))
        '''), cell("markdown", """
        ## Continue the model work

        Complete the already prepared Quality Focal Loss versus native-training
        comparison first, then the controlled training-label comparison. Both
        require GPU training and have their own pinned notebooks. This diagnostic
        does not supply those still-pending model results.

        For any future genuinely unseen-group evaluation, trace initialization
        history: holding a photograph out of fine-tuning is insufficient if the
        starting checkpoint was already trained on it.
        """)]
    return {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
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
