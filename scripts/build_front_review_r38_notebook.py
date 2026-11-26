#!/usr/bin/env python3
"""Generate the small R38 CPU replay notebook after immutable source publication."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import textwrap

sys.dont_write_bytecode = True
from front_review_r38_replay import CONTRACT_PATH, SOURCE_FILES, read_json, require, sha, verify_inputs

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = "notebooks/baomai_front_review_r38.ipynb"


def cell(kind, source):
    result = {"cell_type": kind, "metadata": {}, "source": textwrap.dedent(source).lstrip().splitlines(keepends=True)}
    if kind == "code":
        result.update(execution_count=None, outputs=[])
    return result


def verify_published_files(root, revision, files):
    require(re.fullmatch(r"[0-9a-f]{40}", revision) is not None, "A full immutable Git commit is required")
    entries = subprocess.check_output(["git", "-C", str(root), "ls-tree", "-r", "-z", revision]).split(b"\0")
    tree = {}
    for entry in entries:
        if entry:
            info, name = entry.split(b"\t", 1)
            tree[name.decode()] = info.split()[2].decode()
    for relative in files:
        data = (root / relative).read_bytes()
        blob = hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()
        require(tree.get(relative) == blob, f"Published source differs from local replay input: {relative}")


def build_notebook(root, revision):
    root = Path(root).resolve(); contract = read_json(root / CONTRACT_PATH)
    verify_inputs(root)
    all_files = set(contract["input_files_sha256"]) | set(SOURCE_FILES) | {CONTRACT_PATH}
    verify_published_files(root, revision, all_files)
    source_hashes = {**contract["source_files_sha256"], CONTRACT_PATH: sha(root / CONTRACT_PATH)}
    setup = '''
import hashlib, json, os, subprocess, sys, tempfile
from pathlib import Path

sys.dont_write_bytecode = True
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
REPLAY_ENV = {key: value for key, value in os.environ.items() if key not in ("PYTHONPATH", "PYTHONHOME")}
SOURCE_REVISION = __REVISION__
CONTRACT_SHA256 = __CONTRACT_HASH__
SOURCE_FILES_SHA256 = __SOURCE_HASHES__
SOURCE_URL = os.environ.get("BAOMAI_R38_SOURCE_URL", "https://github.com/Siuuuers/dl.git")
WORK = Path(os.environ.get("BAOMAI_R38_WORK", str(Path("/content") / "baomai-r38-replay" if Path("/content").is_dir() else Path.cwd() / "baomai-r38-replay"))).expanduser().resolve()
REPO = WORK / "source"
OUTPUT = WORK / "replay"
ARCHIVE = WORK / "baomai-front-review-r38-replay-results.zip"

def git(*args, cwd=None):
    env = {**os.environ, "GIT_LFS_SKIP_SMUDGE": "1", "GIT_TERMINAL_PROMPT": "0"}
    process = subprocess.run(["git", "-c", "filter.lfs.process=", "-c", "filter.lfs.smudge=", "-c", "filter.lfs.clean=", "-c", "filter.lfs.required=false", *args], cwd=cwd, env=env, text=True, capture_output=True)
    if process.returncode:
        raise RuntimeError(process.stderr.strip() or "Git source retrieval failed")
    return process.stdout.strip()

def verify_checkout(path):
    if git("rev-parse", "HEAD", cwd=path) != SOURCE_REVISION:
        raise RuntimeError("Source revision differs. Preserve this work folder and choose a new BAOMAI_R38_WORK path.")
    if git("status", "--porcelain", "--untracked-files=all", cwd=path):
        raise RuntimeError("Source checkout is dirty. Preserve its changes and choose a new BAOMAI_R38_WORK path.")
    for relative, expected in SOURCE_FILES_SHA256.items():
        file = path / relative
        if not file.is_file() or hashlib.sha256(file.read_bytes()).hexdigest() != expected:
            raise RuntimeError(f"Pinned replay source hash differs: {relative}")

def assemble_source():
    WORK.mkdir(parents=True, exist_ok=True)
    if REPO.exists():
        verify_checkout(REPO)
    else:
        with tempfile.TemporaryDirectory(prefix=".r38-source-", dir=WORK) as temporary:
            staged = Path(temporary) / "source"
            staged.mkdir()
            git("init", "--quiet", cwd=staged)
            git("fetch", "--quiet", "--depth", "1", SOURCE_URL, SOURCE_REVISION, cwd=staged)
            git("checkout", "--quiet", "--detach", "FETCH_HEAD", cwd=staged)
            verify_checkout(staged)
            staged.replace(REPO)
    return REPO

assemble_source()
print(f"Verified R38 replay source {SOURCE_REVISION[:12]}. CPU only; no image/weight downloads or package installation.")
'''.replace("__REVISION__", repr(revision)).replace("__CONTRACT_HASH__", repr(source_hashes[CONTRACT_PATH])).replace("__SOURCE_HASHES__", repr(source_hashes))
    notebook = {
        "nbformat": 4, "nbformat_minor": 5,
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                     "language_info": {"name": "python"}, "colab": {"name": "baomai_front_review_r38.ipynb"},
                     "baomai": {"kind": "cached-review-replay", "source_revision": revision,
                                "contract_path": CONTRACT_PATH, "contract_sha256": source_hashes[CONTRACT_PATH],
                                "training": False, "inference": False, "code_cells": 4}},
        "cells": [
            cell("markdown", """
                # R38 · Review existing results on CPU

                Choose a **CPU / standard Python runtime**, then **Run all**. This notebook fetches one pinned source commit, verifies the saved inputs, replays the existing PR123/PR121/PR35 predictions, and exports a small results ZIP. No GPU, Torch installation, camera-image download or model weights are needed.

                All eight R38 requests concern development labels: **three replacements and five explicit exclusions**. All 137 reviewed training files and eligible R37 training supervision remain unchanged. A second GPU training pair would have no new annotation treatment. The R37/R38, frozen 666-front and historical 663-front scores remain separate; **PR35 stays selected**.

                Changed scores under R38 describe a changed reference. A deleted missed row is not a recovered detection. The ZIP preserves original R37 row identities, including deleted rows. The inherited pending `new_000010:24` is retained exactly.
            """),
            cell("code", setup),
            cell("markdown", """
                ## 1 · Verify the review and cached inputs

                Check source and file hashes, all 157 original split identities, the 137 unchanged training files, eligible R37 labels, the unchanged original 20-image/664-front trainer validation, and deletion-aware row maps. The replay verifies image **identity hashes recorded in the caches**, without loading the original image bytes. Original-pixel annotation review is documented separately.
            """),
            cell("code", """
                verify_checkout(REPO)
                checked = subprocess.run(
                    [sys.executable, "-S", str(REPO / "scripts/front_review_r38_replay.py"),
                     "--root", str(REPO), "--verify-only", "--contract-sha256", CONTRACT_SHA256],
                    text=True, capture_output=True, env=REPLAY_ENV,
                )
                if checked.returncode:
                    raise RuntimeError(checked.stderr.strip())
                INPUT_AUDIT = json.loads(checked.stdout)
                print(f"Verified {INPUT_AUDIT['input_files_verified']} input files; all 137 reviewed training labels are unchanged.")
                print("Eligible R37 training:", INPUT_AUDIT["eligible_r37_training_counts"])
                print("Original trainer validation:", INPUT_AUDIT["original_trainer_validation"])
                print("R37 development fronts:", INPUT_AUDIT["review_counts"]["r37"]["val"]["objects"])
                print("R38 development fronts:", INPUT_AUDIT["review_counts"]["r38"]["val"]["objects"])
            """),
            cell("markdown", """
                ## 2 · Replay and export

                Use the original fixed scoring rules: detector confidence 0.13, box matching IoU 0.50, endpoint confidence 0.50, and 5/10-pixel corner reports. The cached inference used ordinary NMS at 0.70. No threshold or model-head sweep runs. Replay must exactly reproduce the preserved audited reports before the ZIP is accepted.

                Re-running clean, completed cells verifies and reuses the same results. Changed source or output files are preserved and rejected, so a review run cannot silently replace your edits.
            """),
            cell("code", """
                verify_checkout(REPO)
                completed = subprocess.run(
                    [sys.executable, "-S", str(REPO / "scripts/front_review_r38_replay.py"),
                     "--root", str(REPO), "--source-revision", SOURCE_REVISION,
                     "--contract-sha256", CONTRACT_SHA256, "--output", str(OUTPUT), "--archive", str(ARCHIVE)],
                    text=True, capture_output=True, env=REPLAY_ENV,
                )
                if completed.returncode:
                    raise RuntimeError(completed.stderr.strip())
                RESULT = json.loads(completed.stdout)
                COMPARISON = json.loads((OUTPUT / "comparison.json").read_text())
                for name in ("frozen666", "historical663", "r37", "r38"):
                    data = COMPARISON["references"][name]
                    print(f"\\n{name}")
                    for role in ("reference", "candidate", "pr35"):
                        model = data["models"][role]
                        detection, quality = model["detection"], model["corner_quality"]
                        pairs = [item["both_visible_objects_within_radius"] for item in quality["radii"]]
                        print(f"  {role}: missed {detection['fn']}, unmatched {detection['fp']}, duplicates {detection['duplicate_fp']}; corner pairs 5/10 px {pairs}/{quality['both_visible_objects']}")
                print(f"\\nVerified ZIP: {ARCHIVE.name} ({RESULT['archive_size_bytes']:,} bytes)")
                print("SHA256:", RESULT["archive_sha256"])
            """),
            cell("markdown", """
                ## 3 · Keep the small result ZIP

                The ZIP contains the separate score reports, R37-to-R38 reference-change analysis, row maps, operations, input/context audit, replay contract and file hashes. It contains no model checkpoints or original photographs. Keep it with the two cropped-row review PDFs. This notebook does not train an R38 model or change the selected checkpoint.

                Colab downloads the ZIP below. In another notebook environment, use the printed absolute path. Initial source retrieval needs Git/network access; a verified existing source checkout can be reused offline.
            """),
            cell("code", """
                print("Result file:", ARCHIVE)
                if os.environ.get("BAOMAI_R38_DOWNLOAD", "1") == "1":
                    try:
                        from google.colab import files
                    except ImportError:
                        print("Open the result path above in your notebook file browser.")
                    else:
                        files.download(str(ARCHIVE))
            """),
        ],
    }
    for i, item in enumerate(notebook["cells"]):
        item["id"] = f"r38-{i:02d}"
        if item["cell_type"] == "code":
            compile("".join(item["source"]), f"r38-cell-{i}", "exec")
    path = root / NOTEBOOK_PATH; path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n")
    return {"path": NOTEBOOK_PATH, "sha256": sha(path), "size_bytes": path.stat().st_size,
            "source_revision": revision, "contract_sha256": source_hashes[CONTRACT_PATH],
            "published_files_verified": len(all_files), "code_cells": 4}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--source-revision", required=True)
    args = parser.parse_args()
    print(json.dumps(build_notebook(args.root, args.source_revision), indent=2))
