#!/usr/bin/env python3
"""Build the fixed-source CPU proposal and box-geometry audit notebook."""

import argparse
import hashlib
import json
from pathlib import Path
import re

from build_inference_ablation_r53_notebooks import cell


REFERENCE_ARCHIVE = "docs/thesis/evidence/annotation-reference/baomai-annotation-reference-r60-results.zip"
DRIVERS = ("analyze_proposal_stages_r63.py", "analyze_box_geometry_r63.py")
OUTPUTS = ("summary.json", "cases.json", "results.md", "manifest.json")


def build(source_commit, root):
    if re.fullmatch("[0-9a-f]{40}", source_commit) is None:
        raise ValueError("A complete published source commit is required")
    hashes = {name: hashlib.sha256((root / "scripts" / name).read_bytes()).hexdigest() for name in DRIVERS}
    cells = [cell("markdown", """
        # Explain the remaining cassette-front misses

        **CPU only. Run all four code cells. No training, model download or image
        download is needed.** This notebook reuses the earlier full decoded
        prediction archive, about 15.6 MB, and the completed three-reference
        comparison. It checks that they describe the same checkpoint and images.

        The first analysis traces the current missed fronts through confidence
        filtering, ordinary suppression, clipping, final suppression and reference
        matching. The second describes box size and extent disagreement. It does
        not lower the evaluation threshold, repair labels or choose a new model.

        Pipeline stages describe where eligible candidates disappear. They do not
        establish why training produced a particular score or box. A nearest box
        selected using reference overlap is a diagnostic association, not proof
        that both boxes depict the same physical front.
        """), cell("markdown", """
        ## 1 Load the published implementation

        Source, downloaded inputs and generated outputs stay separate. A changed
        source directory is preserved; choose a fresh `BAOMAI_PROPOSAL_AUDIT_BASE`
        to start another run.
        """), cell("code", f'''
        from pathlib import Path
        import hashlib, json, os, subprocess, sys

        SOURCE_COMMIT = "{source_commit}"
        DRIVER_HASHES = {hashes!r}
        BASE = Path(os.environ.get("BAOMAI_PROPOSAL_AUDIT_BASE", "/content/baomai-proposal-audit-r63")).resolve()
        SOURCE, WORK = BASE / "source", BASE / "work"
        CACHE = Path(os.environ.get("BAOMAI_PROPOSAL_AUDIT_CACHE", str(BASE / "cache"))).resolve()
        BASE.mkdir(parents=True, exist_ok=True)
        ENV = {{**os.environ, "GIT_LFS_SKIP_SMUDGE": "1", "GIT_TERMINAL_PROMPT": "0", "PYTHONDONTWRITEBYTECODE": "1"}}
        def run(command, cwd=None):
            subprocess.run(list(map(str, command)), cwd=cwd, env=ENV, check=True)
        def git(*args):
            return subprocess.check_output(["git", *args], cwd=SOURCE, env=ENV, text=True).strip()
        if not (SOURCE / ".git").exists():
            if SOURCE.exists() and any(SOURCE.iterdir()):
                raise RuntimeError("Existing source preserved; choose a fresh BAOMAI_PROPOSAL_AUDIT_BASE.")
            run(["git", "init", SOURCE])
            run(["git", "remote", "add", "origin", "https://github.com/Siuuuers/dl.git"], SOURCE)
            run(["git", "fetch", "--depth=1", "--filter=blob:none", "origin", SOURCE_COMMIT], SOURCE)
            run(["git", "sparse-checkout", "init", "--no-cone"], SOURCE)
            run(["git", "sparse-checkout", "set", "/scripts/", "/{REFERENCE_ARCHIVE}"], SOURCE)
            run(["git", "checkout", "--detach", SOURCE_COMMIT], SOURCE)
        def verify_source():
            if git("rev-parse", "HEAD") != SOURCE_COMMIT or git("status", "--porcelain", "--untracked-files=all"):
                raise RuntimeError("Changed source preserved; choose a fresh BAOMAI_PROPOSAL_AUDIT_BASE.")
            for name, expected in DRIVER_HASHES.items():
                if hashlib.sha256((SOURCE / "scripts" / name).read_bytes()).hexdigest() != expected:
                    raise RuntimeError("Source identity differs: " + name)
        verify_source()
        WORK.mkdir(parents=True, exist_ok=True)
        print("Published source:", SOURCE_COMMIT)
        '''), cell("markdown", """
        ## 2 Obtain the verified decoded predictions

        NumPy is the only numerical dependency. The earlier archive contains all
        decoded proposals for the same 20 photographs, before score filtering
        and suppression. Its byte count and SHA-256 must match the frozen return.
        Existing compatible downloads are reused.
        """), cell("code", f'''
        import importlib.util
        if importlib.util.find_spec("numpy") is None:
            run([sys.executable, "-m", "pip", "install", "--quiet", "numpy==2.4.6"])
        import numpy as np
        sys.dont_write_bytecode = True
        sys.path.insert(0, str(SOURCE / "scripts"))
        from audit_front_diagnosis_r49_results import obtain_archive, ARCHIVE_BYTES, ARCHIVE_SHA
        R49_ARCHIVE = obtain_archive(CACHE)
        assert R49_ARCHIVE.stat().st_size == ARCHIVE_BYTES
        assert hashlib.sha256(R49_ARCHIVE.read_bytes()).hexdigest() == ARCHIVE_SHA
        REFERENCE_ARCHIVE = SOURCE / "{REFERENCE_ARCHIVE}"
        print("Decoded archive verified:", R49_ARCHIVE.name, ARCHIVE_BYTES, "bytes")
        print("NumPy:", np.__version__)
        '''), cell("markdown", """
        ## 3 Trace the current errors and inspect box geometry

        Historical native outputs must replay exactly. At the current operating
        point, matching and scored outcomes must agree with the completed
        comparison, while any CPU/GPU numerical differences are recorded.

        The main detection confidence remains strictly above 0.25, native and
        final NMS overlap 0.70, and reference matching IoU at least 0.50. The
        existing two-view corner results remain separate: the older full tensors
        describe the original-view boxes used by that method.
        """), cell("code", '''
        STAGES, GEOMETRY = WORK / "proposal-stages", WORK / "box-geometry"
        def stage_command(output):
            return [sys.executable, "-B", SOURCE / "scripts/analyze_proposal_stages_r63.py",
                    "--root", SOURCE, "--r49-archive", R49_ARCHIVE,
                    "--reference-archive", REFERENCE_ARCHIVE, "--output", output]
        def geometry_command(output):
            return [sys.executable, "-B", SOURCE / "scripts/analyze_box_geometry_r63.py",
                    "--root", SOURCE, "--output", output]
        run(stage_command(STAGES))
        run(geometry_command(GEOMETRY))
        from IPython.display import Markdown, display
        display(Markdown((STAGES / "results.md").read_text()))
        display(Markdown((GEOMETRY / "results.md").read_text()))
        '''), cell("markdown", """
        ## 4 Recompute, verify and download

        Both analyses run again into separate verification directories. Every
        expected file must match the first run byte for byte. The result ZIP
        contains only the verified files and a manifest. These results explain
        development errors; they are not a new trained-model result or a robot
        positioning test.
        """), cell("code", f'''
        from zipfile import ZipFile, ZipInfo, ZIP_DEFLATED
        verify_source()
        REPLAY = WORK / "verification"
        run(stage_command(REPLAY / "proposal-stages"))
        run(geometry_command(REPLAY / "box-geometry"))
        EXPECTED = {OUTPUTS!r}
        entries = {{}}
        for name, directory in (("proposal-stages", STAGES), ("box-geometry", GEOMETRY)):
            for filename in EXPECTED:
                original = (directory / filename).read_bytes()
                replayed = (REPLAY / name / filename).read_bytes()
                if original != replayed:
                    raise RuntimeError("Saved result differs on replay: " + name + "/" + filename)
                entries[name + "/" + filename] = original
        manifest = {{"schema": "cassette-proposal-notebook-package-r63-v1",
                    "source_commit": SOURCE_COMMIT, "driver_hashes": DRIVER_HASHES,
                    "fresh_inference": False, "training": False,
                    "files": {{name: {{"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}}
                              for name, data in sorted(entries.items())}}}}
        entries["manifest.json"] = (json.dumps(manifest, indent=2, sort_keys=True) + "\\n").encode()
        RETURN = WORK / "baomai-proposal-audit-r63-results.zip"
        temporary = WORK / "baomai-proposal-audit-r63-results.tmp.zip"
        with ZipFile(temporary, "w", compression=ZIP_DEFLATED, compresslevel=9) as bundle:
            for name, data in sorted(entries.items()):
                info = ZipInfo(name, (2026, 1, 1, 0, 0, 0))
                info.compress_type = ZIP_DEFLATED
                info.external_attr = 0o100644 << 16
                bundle.writestr(info, data, compress_type=ZIP_DEFLATED, compresslevel=9)
        with ZipFile(temporary) as bundle:
            assert bundle.testzip() is None and set(bundle.namelist()) == set(entries)
            assert len(bundle.namelist()) == len(entries)
            for name, data in entries.items():
                assert bundle.read(name) == data
        if RETURN.exists() and RETURN.read_bytes() != temporary.read_bytes():
            raise RuntimeError("Different existing package preserved; choose a fresh base directory.")
        temporary.replace(RETURN)
        verify_source()
        print("Both analyses replayed exactly; verified", len(entries), "ZIP members.")
        from IPython.display import FileLink
        try:
            from google.colab import files
        except ImportError:
            display(FileLink(str(RETURN)))
        else:
            files.download(str(RETURN))
        ''')]
    return {"cells": cells, "nbformat": 4, "nbformat_minor": 5,
            "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                         "language_info": {"name": "python"}, "source_commit": source_commit,
                         "driver_hashes": hashes, "training_required": False,
                         "fresh_inference": False,
                         "colab": {"name": "baomai_proposal_audit_r63.ipynb", "provenance": []}}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    target = args.output or root / "notebooks/baomai_proposal_audit_r63.ipynb"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(build(args.source_commit, root), indent=1) + "\n")
    print(json.dumps({"path": str(target), "source_commit": args.source_commit}))


if __name__ == "__main__":
    main()
