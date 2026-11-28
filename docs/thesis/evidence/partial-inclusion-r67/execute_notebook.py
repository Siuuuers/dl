"""Execute the actual four notebook cells unchanged in one IPython namespace."""

import argparse
import copy
import datetime
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
import traceback
from zoneinfo import ZoneInfo

import IPython
from IPython.core.interactiveshell import InteractiveShell
from IPython.utils.capture import capture_output


def sha(data):
    return hashlib.sha256(data).hexdigest()


def need(ok, message):
    if not ok:
        raise RuntimeError(message)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    args = parser.parse_args()
    root, base = args.root.resolve(), args.base.resolve()
    evidence = root / "docs/thesis/evidence/partial-inclusion-r67"
    notebook_path = root / "notebooks/baomai_partial_inclusion_r67.ipynb"
    need(not base.exists(), "Use a fresh execution base; existing work is preserved")
    need(not (evidence / "notebook-execution.json").exists(), "Existing execution receipt is preserved")
    notebook_bytes = notebook_path.read_bytes()
    notebook = json.loads(notebook_bytes)
    need(sum(c["cell_type"] == "code" for c in notebook["cells"]) == 4, "Expected four code cells")
    source, work = base / "source", base / "work"
    source.parent.mkdir(parents=True)
    env = {**os.environ, "GIT_LFS_SKIP_SMUDGE": "1", "GIT_TERMINAL_PROMPT": "0"}
    subprocess.run(["git", "clone", "--shared", "--no-checkout", str(root), str(source)], env=env, check=True)
    for command in (["sparse-checkout", "init", "--no-cone"],
                    ["sparse-checkout", "set", "/scripts/", "/docs/thesis/evidence/annotation-reference/baomai-annotation-reference-r60-results.zip",
                     "/data/reviews/fronts-2026-11-28-r67/", "/docs/thesis/evidence/partial-inclusion-r67/excluded-front-provenance.json"],
                    ["checkout", "--detach", args.source_commit]):
        subprocess.run(["git", "-C", str(source), *command], env=env, check=True)
    input_dir = work / "inputs"
    input_dir.mkdir(parents=True)
    for filename in ("historical-r60.zip", "quality-focal-pr157.zip", "training-labels-pr158.zip"):
        shutil.copyfile(args.inputs / filename, input_dir / filename)
    os.environ["BAOMAI_PARTIAL_INCLUSION_BASE"] = str(base)
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    sys.dont_write_bytecode = True
    shell = InteractiveShell.instance()
    executed = copy.deepcopy(notebook)
    receipt = {
        "schema": "cassette-partial-inclusion-notebook-execution-r67-v1", "status": "running",
        "source_commit": args.source_commit, "timezone": "Asia/Hong_Kong",
        "started_at": datetime.datetime.now(ZoneInfo("Asia/Hong_Kong")).isoformat(),
        "notebook": {"path": str(notebook_path.relative_to(root)), "bytes": len(notebook_bytes), "sha256": sha(notebook_bytes)},
        "execution": "All four actual code cells, unchanged, sequentially in one IPython InteractiveShell namespace.",
        "source_acquisition": "Clean isolated local clone checked out at the published immutable commit before cell 1.",
        "input_acquisition": "Three archived inputs preseeded from local copies; the actual cells verify all hashes and inventories.",
        "fresh_inference": False, "training": False, "historical_references_modified": False,
        "training_labels_modified": False, "predictions_modified": False, "new_evaluation_reference": True, "automatic_promotion": False,
        "environment": {"python": platform.python_version(), "IPython": IPython.__version__, "platform": platform.platform()},
        "runner_sha256": sha(Path(__file__).read_bytes()), "cells": [],
        "limitations": ["Local IPython execution, not hosted Google Colab.",
                        "GitHub download paths were not exercised because immutable source and inputs were preseeded.",
                        "The non-Colab FileLink was captured; the Colab browser download interface was not tested."]}
    try:
        number = 0
        for index, cell in enumerate(notebook["cells"]):
            if cell["cell_type"] != "code":
                continue
            number += 1
            code = "".join(cell["source"])
            started = time.monotonic()
            with capture_output(stdout=True, stderr=True, display=True) as captured:
                result = shell.run_cell(code, store_history=False)
            error = result.error_before_exec or result.error_in_exec
            outputs = []
            for name, text in (("stdout", captured.stdout), ("stderr", captured.stderr)):
                if text:
                    outputs.append({"output_type": "stream", "name": name, "text": text.splitlines(keepends=True)})
            for rich in captured.outputs:
                outputs.append({"output_type": "display_data", "data": rich.data, "metadata": rich.metadata or {}})
            if error:
                outputs.append({"output_type": "error", "ename": type(error).__name__, "evalue": str(error),
                                "traceback": traceback.format_exception(type(error), error, error.__traceback__)})
            executed["cells"][index]["execution_count"] = number
            executed["cells"][index]["outputs"] = outputs
            log = evidence / f"notebook-cell-{number}.txt"
            log.write_text(captured.stdout + captured.stderr)
            receipt["cells"].append({"code_cell": number, "notebook_cell_index": index,
                "source_sha256": sha(code.encode()), "source_unchanged": executed["cells"][index]["source"] == cell["source"],
                "status": "failed" if error else "passed", "seconds": round(time.monotonic() - started, 3),
                "log": log.name, "log_sha256": sha(log.read_bytes()), "rich_output_count": len(captured.outputs)})
            print(f"Cell {number}: {receipt['cells'][-1]['status']}", flush=True)
            if error:
                raise RuntimeError(f"Notebook code cell {number} failed") from error
        need(notebook_path.read_bytes() == notebook_bytes, "Original notebook changed")
        need(shell.user_ns["SOURCE_COMMIT"] == args.source_commit, "Executed source pin differs")
        shell.user_ns["verify_source"]()
        output = work / "results"
        inventory = json.loads((output / "inventory.json").read_bytes())
        for filename in [*inventory, "inventory.json"]:
            target, data = evidence / filename, (output / filename).read_bytes()
            need(not target.exists() or target.read_bytes() == data, "Existing analysis differs: " + filename)
            target.write_bytes(data)
        package = json.loads((work / "package-manifest.json").read_bytes())
        archive = output / package["result"]["filename"]
        need(sha(archive.read_bytes()) == package["result"]["sha256"], "Result package identity differs")
        shutil.copyfile(archive, evidence / archive.name)
        shutil.copyfile(work / "package-manifest.json", evidence / "package-manifest.json")
        receipt.update(status="passed", input_notebook_unchanged=True,
                       source_checkout_clean=True, package=package["result"],
                       matrix_rows=len(shell.user_ns["summary"]["matrix"]),
                       record_counts=shell.user_ns["summary"]["record_counts"],
                       archived_image_scores_replayed=shell.user_ns["summary"]["provenance"]["replayed_image_scores"])
    except Exception as error:
        receipt["status"] = "failed"
        receipt["error"] = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        executed_path = evidence / "executed-notebook.ipynb"
        executed_path.write_text(json.dumps(executed, indent=1) + "\n")
        receipt["executed_notebook"] = {"path": executed_path.name, "bytes": executed_path.stat().st_size,
                                        "sha256": sha(executed_path.read_bytes())}
        receipt["finished_at"] = datetime.datetime.now(ZoneInfo("Asia/Hong_Kong")).isoformat()
        (evidence / "notebook-execution.json").write_text(json.dumps(receipt, indent=2) + "\n")
        print(json.dumps({"status": receipt["status"], "cells": [c["status"] for c in receipt["cells"]],
                          "receipt": str(evidence / "notebook-execution.json")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
