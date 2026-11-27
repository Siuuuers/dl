#!/usr/bin/env python3
"""Build the small, source-pinned R47 expanded-data pilot notebook."""
import argparse
import json
from pathlib import Path
import re
import textwrap

import run_additional_front_pilot_r47 as pilot


def cell(kind, value):
    result = dict(cell_type=kind, metadata={}, source=textwrap.dedent(value).strip().splitlines(keepends=True))
    if result["source"] and not result["source"][-1].endswith("\n"):
        result["source"][-1] += "\n"
    if kind == "code":
        result.update(execution_count=None, outputs=[])
        compile("".join(result["source"]), "<notebook-cell>", "exec")
    return result


def build(root, source_revision):
    if not re.fullmatch(r"[0-9a-f]{40}", source_revision):
        raise ValueError("Use a published full source commit SHA")
    release_path = root / pilot.REVIEW / "release-status.json"
    release = pilot.read(pilot.checked(release_path, pilot.RELEASE_SHA256))
    complete = sorted(release["annotation_complete_ids"])
    counts = [pilot.counts([pilot.labels((root / pilot.REVIEW / "complete-candidates/labels" / (identity + ".txt")).read_bytes(), identity)]) for identity in complete]
    candidate = [base + sum(item[index] for item in counts) for index, base in enumerate((137, 3784, 6837, 731, 13))]
    reference = [138, 3827, 6923, 731, 13]
    budgets = {"reference": pilot.expected_budget(reference[0]), "candidate": pilot.expected_budget(candidate[0])}
    ids = ", ".join("`" + identity + "`" for identity in complete)
    shared_changed = pilot.sha(root / pilot.REVIEW / "complete-candidates/labels/frame_0383.txt") != pilot.OLD_EXTRA_LABEL_SHA
    treatment_note = "The candidate also contains the frozen R47 corrections to shared frame_0383." if shared_changed else "Shared frame_0383 labels are byte-identical in both arms."
    cells = [cell("markdown", f"""
        # R47 - expanded supplementary-data pilot

        **Choose a GPU runtime, then Run all.** The notebook uses four code cells.
        The first two perform CPU preparation. A T4 completed the prior R45 recipe;
        this notebook checks the actual GPU/runtime and runs a discarded resource
        trial before the two full arms.

        Reference: the original 137 photos with effective R37 labels, plus the
        exact frame_0383 labels used by PR133. Candidate: the same original 137
        photos, plus all complete R47 photos: {ids}.
        {treatment_note}

        | Planned training exposure | Reference | Candidate |
        | --- | ---: | ---: |
        | Photographs | {reference[0]} | {candidate[0]} |
        | Fronts | {reference[1]} | {candidate[1]} |
        | Locatable corner slots | {reference[2]} | {candidate[2]} |
        | Unavailable corner slots | {reference[3]} | {candidate[3]} |
        | Epochs | 30 | 30 |
        | Batches | {budgets['reference']['batches']} | {budgets['candidate']['batches']} |
        | Expected measured optimizer updates | {budgets['reference']['steps']} | {budgets['candidate']['steps']} |
        | Image presentations | {budgets['reference']['images']} | {budgets['candidate']['images']} |

        This is an **expanded-data, equal-epoch pilot**. Additional exposure is part
        of the treatment; the experiment does not isolate new data from compute.
        Both arms start independently from epoch173 with the same seed46 recipe.
        Held photographs are excluded as whole images. The original 20 development
        photos and all three evaluation references remain unchanged.

        Keep the downloaded return ZIP. Optional Drive backups preserve verified
        completed arms. Interrupted runs remain intact; no partial optimizer resume
        or automatic model promotion occurs. PR35 remains selected pending review.
        """), cell("markdown", """
        ## 1. Load the published source

        The small notebook points to one immutable source commit. It preserves an
        incompatible existing checkout; choose a fresh base folder when necessary.
        """), cell("code", f"""
        from pathlib import Path
        import hashlib, json, os, shutil, subprocess, sys

        SOURCE_REVISION = "{source_revision}"
        RELEASE_SHA256 = "{pilot.RELEASE_SHA256}"
        BASE = Path(os.environ.get("BAOMAI_R47_BASE", "/content/baomai-r47")).resolve()
        SOURCE, WORK, CACHE = BASE / "source", BASE / "work", BASE / "cache"
        REVIEW = WORK / "review"
        BASE.mkdir(parents=True, exist_ok=True)
        ENV = {{**os.environ, "GIT_LFS_SKIP_SMUDGE": "1", "GIT_TERMINAL_PROMPT": "0"}}
        def run(command, cwd=None):
            subprocess.run(list(map(str, command)), cwd=cwd, env=ENV, check=True)
        def sha(path):
            with Path(path).open("rb") as stream:
                return hashlib.file_digest(stream, "sha256").hexdigest()
        if not (SOURCE / ".git").exists():
            if SOURCE.exists() and any(SOURCE.iterdir()):
                raise RuntimeError("Preserved nonempty source folder; choose a fresh BAOMAI_R47_BASE")
            run(["git", "init", SOURCE])
            run(["git", "remote", "add", "origin", "https://github.com/Siuuuers/dl.git"], SOURCE)
            run(["git", "fetch", "--depth=1", "--filter=blob:none", "origin", SOURCE_REVISION], SOURCE)
            run(["git", "sparse-checkout", "init", "--no-cone"], SOURCE)
            run(["git", "sparse-checkout", "set", "/scripts/", "/data/manifest.json", "/data/labels/",
                 "/data/candidates/front-review-2026-11-26-r37/", "/data/reviews/fronts-2026-11-26-r38/",
                 "/data/reviews/additional-fronts-2026-11-27-r46/", "/data/reviews/additional-fronts-2026-11-27-r47/",
                 "/data/references/sequence-refined-reference-2026-11-18/",
                 "/data/references/visible-front-completion-2026-11-14/"], SOURCE)
            run(["git", "checkout", "--detach", SOURCE_REVISION], SOURCE)
        head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=SOURCE, text=True).strip()
        dirty = subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], cwd=SOURCE, text=True)
        if head != SOURCE_REVISION or dirty:
            raise RuntimeError("Source differs; preserve it and choose a fresh BAOMAI_R47_BASE")
        assert sha(SOURCE / "{pilot.REVIEW}/release-status.json") == RELEASE_SHA256
        SCRIPT = SOURCE / "scripts/run_additional_front_pilot_r47.py"
        def command(action, *extra):
            return [sys.executable, "-B", SCRIPT, action, "--root", SOURCE,
                    "--work", WORK, "--review", REVIEW, *extra]
        print("Pinned source:", head)
        """), cell("markdown", """
        ## 2. Verify labels and restore inputs

        This CPU step replays the R47 review, checks the complete-photo export and
        freezes both treatments. The cache downloads only the 157 original PNGs,
        epoch173 and the pinned review archive. Every payload is hash checked.
        It does not install an ML framework or train a model.
        """), cell("code", """
        run([sys.executable, "-m", "pip", "install", "--quiet", "Pillow==13.0.0", "reportlab==4.4.9"])
        run([sys.executable, "-B", SOURCE / "scripts/build_front_review_r47.py",
             "--r44-archive", CACHE / "r44-review.zip", "--download-r44", "--output", REVIEW])
        run(command("prepare", "--inputs-dir", CACHE / "inputs", "--download-inputs"))
        PLAN = json.loads((WORK / "staging.json").read_text())
        print("Complete candidate photos:", PLAN["treatment"]["candidate_extra_ids"])
        for role, budget in PLAN["budgets"].items():
            print(role, PLAN["counts"][role]["train"], "batches", budget["batches"],
                  "updates", budget["steps"], "image presentations", budget["images"])
        print("CPU preflight complete. No training has occurred.")
        """), cell("markdown", """
        ## 3. Resource trial and both full arms

        The actual native training source, dataset loader, coordinates, image
        exposure and optimizer-update schedule are checked. Both full arms use
        fresh epoch173; the four-batch trial is discarded. Verified completed arms
        can be recovered from Drive. Set `USE_DRIVE = False` to skip that backup.
        """), cell("code", """
        from importlib.metadata import version
        import torch
        assert torch.cuda.is_available(), "Select a GPU runtime before this cell."
        print("GPU:", torch.cuda.get_device_name(0))
        constraints = BASE / "torch-constraints.txt"
        constraints.write_text("".join(f"{name}=={version(name)}\\n" for name in ("torch", "torchvision")))
        run([sys.executable, "-m", "pip", "install", "-c", constraints, "ultralytics==8.4.182", "psutil"])
        USE_DRIVE = True
        BACKUP = None
        if USE_DRIVE:
            try:
                from google.colab import drive
            except ImportError:
                print("Drive unavailable; keep the final downloaded ZIP.")
            else:
                drive.mount("/content/drive")
                BACKUP = Path("/content/drive/MyDrive/baomai-r47") / SOURCE_REVISION
                BACKUP.mkdir(parents=True, exist_ok=True)
        extra = ["--backup-dir", BACKUP] if BACKUP else []
        run(command("run-pair", *extra))
        """), cell("markdown", """
        ## 4. Score and download the complete return

        Inference uses unchanged operating points. The return contains both final
        checkpoints, receipts, prediction caches, exact labels, source closure and
        review provenance. Existing cached evaluation and return files are verified
        before reuse. Counts are reported separately for current R38, frozen666 and
        historical663; they describe the same development photographs.
        """), cell("code", """
        ARCHIVE = BASE / "baomai-additional-front-pilot-r47-results.zip"
        available_return = ARCHIVE if ARCHIVE.exists() else (
            BACKUP / ARCHIVE.name if BACKUP and (BACKUP / ARCHIVE.name).exists() else None)
        if available_return:
            run(command("restore-return", "--output", available_return))
            if available_return != ARCHIVE:
                shutil.copyfile(available_return, ARCHIVE)
                assert sha(available_return) == sha(ARCHIVE)
        else:
            complete_evaluation = all((WORK / "evaluation" / name).is_file()
                                      for name in ("summary.json", "predictions.json"))
            run(command("verify-evaluation" if complete_evaluation else "evaluate"))
        run(command("package", "--output", ARCHIVE))
        summary = json.loads((WORK / "evaluation/summary.json").read_text())
        for reference in ("r38", "frozen666", "historical663"):
            for role in ("reference", "candidate"):
                report = summary["references"][reference]["postclip"][role]
                d, c = report["detection"], report["corners_by_locatable_count"]["2"]
                print(reference, role, "misses", d["fn"], "unmatched", d["fp"],
                      "duplicate subset", d["duplicate_fp"], "other", d["other_fp"],
                      "both within5px", c["within5"], "/", c["eligible"],
                      "within10px", c["within10"])
        if BACKUP:
            destination = BACKUP / ARCHIVE.name
            if destination.exists():
                assert sha(destination) == sha(ARCHIVE), "Different completed return preserved"
            else:
                shutil.copyfile(ARCHIVE, destination)
            assert sha(destination) == sha(ARCHIVE)
        print("Return:", ARCHIVE, "SHA256:", sha(ARCHIVE))
        try:
            from google.colab import files
        except ImportError:
            print("Saved locally:", ARCHIVE)
        else:
            files.download(str(ARCHIVE))
        """)]
    return dict(cells=cells, metadata=dict(kernelspec=dict(display_name="Python 3", language="python", name="python3"),
                language_info=dict(name="python"), colab=dict(name="baomai_additional_front_pilot_r47.ipynb", provenance=[]),
                source_revision=source_revision, release_sha256=pilot.RELEASE_SHA256, training=True,
                experiment="expanded-data, equal-epoch pilot"), nbformat=4, nbformat_minor=5)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or args.root / "notebooks/baomai_additional_front_pilot_r47.ipynb"
    notebook = build(args.root.resolve(), args.source_revision)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n")
    print(output, "bytes", output.stat().st_size, "code cells", sum(cell["cell_type"] == "code" for cell in notebook["cells"]))


if __name__ == "__main__":
    main()
