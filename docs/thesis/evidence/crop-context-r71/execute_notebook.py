"""Execute the four unchanged R71 notebook code cells in one IPython namespace."""

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

import IPython
from IPython.core.interactiveshell import InteractiveShell
from IPython.utils.capture import capture_output


def sha(data):
    return hashlib.sha256(data).hexdigest()


def need(value, message):
    if not value:
        raise RuntimeError(message)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    args = parser.parse_args()
    root, base = args.root.resolve(), args.base.resolve()
    need(not base.exists(), "Existing execution work is preserved. Choose a new base.")
    source, work, evidence = base / "source", base / "work", base / "execution-evidence"
    evidence.mkdir(parents=True)
    notebook_path = root / "notebooks/baomai_crop_context_r71.ipynb"
    notebook_bytes = notebook_path.read_bytes()
    notebook = json.loads(notebook_bytes)
    need(notebook["metadata"]["source_commit"] == args.source_commit, "Notebook source pin differs")
    need(sum(c["cell_type"] == "code" for c in notebook["cells"]) == 4, "Expected four code cells")
    env = {**os.environ, "GIT_LFS_SKIP_SMUDGE": "1", "GIT_TERMINAL_PROMPT": "0", "PYTHONDONTWRITEBYTECODE": "1"}
    subprocess.run(["git", "clone", "--shared", "--no-checkout", str(root), str(source)], env=env, check=True)
    fixed = list(notebook["metadata"]["fixed_input_hashes"])
    sparse = ["/scripts/", "/data/reviews/fronts-2026-11-28-r67/",
        "/docs/thesis/evidence/partial-inclusion-r67/excluded-front-provenance.json",
        "/docs/thesis/evidence/partial-detection-r68/baomai-partial-detection-r68-results.zip",
        *("/" + name for name in fixed)]
    for command in (["sparse-checkout", "init", "--no-cone"], ["sparse-checkout", "set", *sparse],
                    ["checkout", "--detach", args.source_commit]):
        subprocess.run(["git", "-C", str(source), *command], env=env, check=True)
    shutil.copytree(args.inputs, work / "inputs")
    os.environ["BAOMAI_CROP_CONTEXT_BASE"] = str(base)
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    sys.dont_write_bytecode = True
    executed = copy.deepcopy(notebook)
    receipt = {"schema": "cassette-crop-context-notebook-execution-r71-v1", "status": "running",
        "source_commit": args.source_commit, "timezone": "UTC",
        "started_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "notebook": {"path": str(notebook_path.relative_to(root)), "bytes": len(notebook_bytes), "sha256": sha(notebook_bytes)},
        "execution": "All four actual code cells, unchanged, sequentially in one IPython InteractiveShell namespace.",
        "source_acquisition": "Clean isolated local clone checked out at the published immutable commit before cell 1.",
        "input_acquisition": "Checkpoint, return manifest, two archives and all twenty original photographs preseeded from verified local copies. Actual notebook cells recheck all input hashes.",
        "environment": {"python": platform.python_version(), "IPython": IPython.__version__, "platform": platform.platform()},
        "runner_sha256": sha(Path(__file__).read_bytes()), "cells": [],
        "fresh_inference": True, "training": False, "optimizer_updates": 0, "labels_changed": False,
        "threshold_search": False, "automatic_promotion": False,
        "limitations": ["Local IPython execution, not hosted Google Colab.",
            "GitHub download paths were not exercised because immutable source and inputs were preseeded.",
            "The FileLink output was captured. The Colab browser download interface was not tested."]}
    shell = InteractiveShell.instance()
    try:
        number = 0
        for index, cell in enumerate(notebook["cells"]):
            if cell["cell_type"] != "code":
                continue
            number += 1
            code = "".join(cell["source"])
            print(f"Starting code cell {number}", flush=True)
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
            executed["cells"][index].update(execution_count=number, outputs=outputs)
            log = evidence / f"notebook-cell-{number}.txt"
            log.write_text(captured.stdout + captured.stderr)
            receipt["cells"].append({"code_cell": number, "notebook_cell_index": index,
                "source_sha256": sha(code.encode()), "source_unchanged": executed["cells"][index]["source"] == cell["source"],
                "status": "failed" if error else "passed", "seconds": round(time.monotonic() - started, 3),
                "log": log.name, "log_sha256": sha(log.read_bytes()), "rich_output_count": len(captured.outputs)})
            print(f"Code cell {number}: {receipt['cells'][-1]['status']} ({receipt['cells'][-1]['seconds']} s)", flush=True)
            if error:
                raise RuntimeError(f"Notebook code cell {number} failed") from error
        need(notebook_path.read_bytes() == notebook_bytes, "The source notebook changed during execution")
        need(shell.user_ns["SOURCE_COMMIT"] == args.source_commit, "Executed source commit differs")
        shell.user_ns["verify_source"]()
        summary = shell.user_ns["summary"]
        package = json.loads((work / "package-manifest.json").read_bytes())
        output = work / "crop-context"
        archive = output / package["result"]["filename"]
        need(sha(archive.read_bytes()) == package["result"]["sha256"], "Result archive identity differs")
        receipt.update(status="passed", input_notebook_unchanged=True, source_checkout_clean=True,
            package=package["result"], matrix_rows=len(summary["matrix"]), record_counts=summary["record_counts"],
            runtime=summary["runtime"], control_gate_passed=summary["control_gate_passed"],
            control_records_exact=summary["control_records_exact"],
            protocol_sha256=summary["protocol_sha256"], source_sha256=summary["analysis_source_sha256"],
            checkpoint=summary["checkpoint"], image_inputs=summary["image_inputs"],
            reference_input_sha256=summary["reference_input_sha256"],
            input_files={str(p.relative_to(work / "inputs")): {"bytes": p.stat().st_size, "sha256": sha(p.read_bytes())}
                for p in sorted((work / "inputs").rglob("*")) if p.is_file()})
        shutil.copyfile(work / "package-manifest.json", evidence / "package-manifest.json")
    except Exception as error:
        receipt["status"] = "failed"
        receipt["error"] = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        executed_path = evidence / "executed-notebook.ipynb"
        executed_path.write_text(json.dumps(executed, indent=1) + "\n")
        receipt["executed_notebook"] = {"path": executed_path.name, "bytes": executed_path.stat().st_size,
                                       "sha256": sha(executed_path.read_bytes())}
        receipt["finished_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        (evidence / "notebook-execution.json").write_text(json.dumps(receipt, indent=2) + "\n")
        print(json.dumps({"status": receipt["status"], "cells": [c["status"] for c in receipt["cells"]],
                          "receipt": str(evidence / "notebook-execution.json")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
