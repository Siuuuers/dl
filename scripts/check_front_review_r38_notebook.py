#!/usr/bin/env python3
"""Execute every R38 notebook cell against actual inputs and an offline Git remote."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import zipfile

sys.dont_write_bytecode = True
from front_review_r38_replay import CONTRACT_PATH, RESULT_NAME, json_bytes, read_json, require, sha

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = "notebooks/baomai_front_review_r38.ipynb"


def command(args, cwd=None, env=None, expect_success=True):
    process = subprocess.run(args, cwd=cwd, env=env, text=True, capture_output=True)
    if expect_success and process.returncode:
        raise RuntimeError(f"Validation command failed: {args[0]}\n{process.stdout}\n{process.stderr}")
    return process


def check(root, notebook_path):
    root = Path(root).resolve(); path = root / notebook_path; notebook = read_json(path)
    metadata = notebook["metadata"]["baomai"]
    revision = metadata["source_revision"]; contract_hash = metadata["contract_sha256"]
    require(metadata["training"] is False and metadata["inference"] is False
            and metadata["contract_path"] == CONTRACT_PATH, "Notebook scope differs")
    code = [(i, "".join(cell["source"])) for i, cell in enumerate(notebook["cells"]) if cell["cell_type"] == "code"]
    require(len(code) == metadata["code_cells"] == 4, "Expected four complete notebook code cells")
    joined = "\n".join(text for _, text in code)
    require(joined.count("def assemble_source(") == 1, "Source loader must have exactly one definition")
    require(all(token not in joined for token in ("pip install", "import torch", ".train(", "BASE_REVISION", "SOURCE_TREE", "b64decode")),
            "Notebook contains training, package installation or an obsolete embedded-source path")
    for i, text in code:
        compile(text, f"r38-cell-{i}", "exec")
    contract = read_json(root / CONTRACT_PATH)
    require(sha(root / CONTRACT_PATH) == contract_hash, "Notebook pins another contract")
    runner = """
import json, os, sys
sys.dont_write_bytecode = True
document = json.load(open(os.environ['BAOMAI_R38_NOTEBOOK_FILE']))
namespace = {'__name__': '__main__'}
for i, cell in enumerate(document['cells']):
    if cell['cell_type'] == 'code':
        exec(compile(''.join(cell['source']), f'actual-notebook-cell-{i}', 'exec'), namespace)
"""
    with tempfile.TemporaryDirectory(prefix="r38-notebook-integration-") as temporary:
        base = Path(temporary); remote = base / "remote.git"; work = base / "work"
        env = {key: value for key, value in os.environ.items() if key not in ("PYTHONPATH", "PYTHONHOME")}
        env.update(PYTHONDONTWRITEBYTECODE="1", PYTHONNOUSERSITE="1", GIT_LFS_SKIP_SMUDGE="1",
                   GIT_TERMINAL_PROMPT="0", BAOMAI_R38_SOURCE_URL=str(remote), BAOMAI_R38_WORK=str(work),
                   BAOMAI_R38_DOWNLOAD="0", BAOMAI_R38_NOTEBOOK_FILE=str(path))
        command(["git", "init", "--bare", "--quiet", str(remote)], env=env)
        command(["git", "--git-dir", str(remote), "fetch", "--quiet", "--depth", "1", str(root),
                 f"{revision}:refs/heads/pinned"], env=env)
        first = command([sys.executable, "-S", "-c", runner], cwd=base, env=env)
        archive = work / RESULT_NAME; first_hash = sha(archive)
        input_audit = read_json(work / "replay" / "input-audit.json")
        source = work / "source"
        require(input_audit["source"]["revision"] == revision and input_audit["preserved_replay_exactly_reproduced"] is True,
                "Actual replay/source verification was not recorded")
        require(input_audit["input_files_verified"] == len(contract["input_files_sha256"])
                and len(input_audit["unchanged_review_training_files"]) == 137, "Actual input inventory was not verified")
        command([sys.executable, "-S", "-c", runner], cwd=base, env=env)
        require(sha(archive) == first_hash, "Clean notebook reuse changed the result ZIP")
        with zipfile.ZipFile(archive) as bundle:
            require(bundle.testzip() is None, "Result ZIP failed CRC verification")
            index = json.loads(bundle.read("manifest.json"))
            require(set(bundle.namelist()) == {"manifest.json", *(item["path"] for item in index["files"])},
                    "Result ZIP member inventory differs")
            for item in index["files"]:
                data = bundle.read(item["path"])
                require(len(data) == item["size_bytes"] and hashlib.sha256(data).hexdigest() == item["sha256"],
                        "Result ZIP member hash differs")
            zip_members = len(bundle.namelist())

        # Verify individual input hash checks, not only Git's dirty-checkout guard.
        negative_inputs = {}
        targets = {"prediction_cache": contract["prediction_caches"]["candidate"]["path"],
                   "eligible_training_label": contract["unchanged_effective_training_labels"][0]["effective_label"]["path"]}
        for label, relative in targets.items():
            target = source / relative; original = target.read_bytes()
            try:
                target.write_bytes(original + b"\n")
                process = command([sys.executable, "-S", str(source / "scripts/front_review_r38_replay.py"),
                                   "--root", str(source), "--verify-only", "--contract-sha256", contract_hash],
                                  cwd=base, env=env, expect_success=False)
                require(process.returncode != 0 and "Replay input hash differs" in process.stderr,
                        f"Altered {label} was not rejected by the input verifier")
                negative_inputs[label] = {"path": relative, "rejected": True, "guard": "actual SHA256 mismatch"}
            finally:
                target.write_bytes(original)

        dirty_file = source / "scripts/score_pr123_review.py"; original = dirty_file.read_bytes()
        try:
            dirty_file.write_bytes(original + b"\n# preserved integration-test edit\n")
            process = command([sys.executable, "-S", "-c", runner], cwd=base, env=env, expect_success=False)
            require(process.returncode != 0 and "Source checkout is dirty" in process.stderr
                    and dirty_file.read_bytes() != original, "Dirty source was not preserved and rejected")
        finally:
            dirty_file.write_bytes(original)
        source_tree = command(["git", "-C", str(source), "rev-parse", "HEAD^{tree}"], env=env).stdout.strip()
        git_env = {**env, "GIT_AUTHOR_NAME": "Notebook integration test", "GIT_AUTHOR_EMAIL": "test@example.invalid",
                   "GIT_COMMITTER_NAME": "Notebook integration test", "GIT_COMMITTER_EMAIL": "test@example.invalid",
                   "GIT_AUTHOR_DATE": "2000-01-01T00:00:00+00:00", "GIT_COMMITTER_DATE": "2000-01-01T00:00:00+00:00"}
        wrong = command(["git", "-C", str(source), "commit-tree", source_tree, "-m", "Wrong revision fixture"], env=git_env).stdout.strip()
        command(["git", "-C", str(source), "checkout", "--quiet", "--detach", wrong], env=env)
        process = command([sys.executable, "-S", "-c", runner], cwd=base, env=env, expect_success=False)
        require(process.returncode != 0 and "Source revision differs" in process.stderr, "Wrong source revision was accepted")
        require(command(["git", "-C", str(source), "rev-parse", "HEAD"], env=env).stdout.strip() == wrong,
                "Wrong-revision checkout was silently replaced")
        report = {
            "schema_version": 1, "kind": "R38 actual-notebook CPU replay validation", "status": "passed",
            "notebook": {"path": notebook_path, "sha256": sha(path), "size_bytes": path.stat().st_size,
                         "code_cells_executed_in_order": [i for i, _ in code]},
            "source": {"revision": revision, "tree": source_tree}, "contract_sha256": contract_hash,
            "execution": {"python_version": sys.version.split()[0], "site_packages_enabled": False,
                          "git_transport": "temporary local bare remote containing the exact published commit",
                          "network_used_by_notebook_execution": False, "input_data": "actual committed caches, labels, manifests and row maps",
                          "synthetic_predictions_or_labels": False, "training_executed": False, "inference_executed": False,
                          "model_weights_loaded": False, "original_images_loaded": False},
            "checks": {"all_code_cells_compiled": True, "complete_notebook_executed": True,
                       "actual_cached_scores_exactly_reproduced": True, "all_137_training_labels_byte_identical": True,
                       "all_137_effective_training_labels_verified": True, "original_trainer_validation_20_664_preserved": True,
                       "all_original_row_maps_verified": True, "clean_reuse_exact_zip": True,
                       "dirty_source_preserved_and_rejected": True, "wrong_revision_preserved_and_rejected": True,
                       "zip_members_crc_size_sha256_verified": True},
            "altered_input_rejections": negative_inputs,
            "verified_input_files": input_audit["input_files_verified"],
            "prediction_caches_verified": input_audit["prediction_caches"],
            "review_operation_counts": input_audit["review_operation_counts"],
            "original_request_lineage": input_audit["original_request_lineage"],
            "result_archive": {"filename": RESULT_NAME, "sha256": first_hash,
                               "size_bytes": archive.stat().st_size, "members": zip_members},
            "stdout_summary": [line for line in first.stdout.splitlines()
                               if line.startswith(("Verified ", "Original trainer", "R37 development", "R38 development"))],
        }
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--notebook", default=NOTEBOOK_PATH)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(); report = check(args.root, args.notebook)
    args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_bytes(json_bytes(report))
    print(json.dumps({"status": "passed", "validation_path": str(args.output),
                      "validation_sha256": sha(args.output), "notebook": report["notebook"],
                      "source": report["source"], "result_archive": report["result_archive"]}, indent=2))
