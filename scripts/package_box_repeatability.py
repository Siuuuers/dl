#!/usr/bin/env python3
"""Package the fixed seed-43/44 box-supervision study without loading models."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import tempfile
import zipfile

from evaluate_pose import file_sha256, preflight_dataset
from package_gpu_results import (INDEX_NAME, check_pair, collect_arm, copy_member,
                                 shared_settings, verify_archive)
from pose_checkpoint import (BOX_REPEATABILITY_RECIPES, _artifact, _check_staging,
                             _contained_path, _read_json, _require, _same, recipe_identity)


STUDY_ID = "box-supervision-repeatability-v1"
SEEDS = (43, 44)
ARMS = ("reference", "candidate")
CANDIDATE_ID = "box-refinement-2026-11-07"
CANDIDATE_MANIFEST = f"data/candidates/{CANDIDATE_ID}/manifest.json"
CANDIDATE_SHA256 = "b61d012052001377089ad048ba13da14b8cd19d665ac8f0ef7828e45c530ec3e"


def study_settings(receipt):
    """Across seeds, allow only the declared seed and run-location differences."""
    return {key: value for key, value in shared_settings(receipt).items() if key != "seed"}


def check_study_arm(bundle, seed):
    identity, receipt, proposal = bundle["identity"], bundle["receipt"], bundle["staging"]["source"]
    expected = recipe_identity(BOX_REPEATABILITY_RECIPES[seed])
    _require(all(identity.get(key) == value for key, value in expected.items())
             and type(receipt["training_settings"]["seed"]) is int
             and receipt["training_settings"]["seed"] == seed,
             f"seed {seed}: expected its exact repeatability recipe and seed")
    _require(proposal["candidate_id"] == CANDIDATE_ID
             and proposal["candidate_manifest"] == CANDIDATE_MANIFEST
             and proposal["candidate_manifest_sha256"] == CANDIDATE_SHA256,
             f"seed {seed}: expected the unchanged reviewed box candidate")


def collect_trial(preflight, run, candidate):
    """A successful four-batch trial is supplemental evidence, never a checkpoint."""
    receipt_path = _contained_path(run, "training-receipt.json")
    receipt = _read_json(receipt_path)
    expected = recipe_identity(BOX_REPEATABILITY_RECIPES[43])
    _require(receipt["schema_version"] == 2 and receipt["purpose"] == "resource-trial"
             and receipt["status"] == "trial-complete" and receipt["arm"] == "candidate"
             and all(receipt.get(key) == value for key, value in expected.items()),
             "supplemental trial must be the completed seed-43 candidate resource trial")
    _require(type(receipt["batches_completed"]) is int and receipt["batches_completed"] == 4
             and type(receipt["optimizer_steps"]) is int and receipt["optimizer_steps"] > 0
             and receipt["optimizer_parameter_change_observed"] is True
             and receipt["completed_epochs"] == [] and receipt["checkpoint_selection"] == "none"
             and receipt["training_returned"] is False and receipt["finalization_completed"] is False
             and receipt["validation_executed"] is False and "checkpoint" not in receipt
             and not (run / "checkpoint-manifest.json").exists()
             and not (run / "checkpoint-manifest.json").is_symlink(),
             "resource trial must record four batches and updates without a completed model")
    _require(type(receipt["execution"]["exit_code"]) is int and receipt["execution"]["exit_code"] == 0
             and receipt["execution"]["stop_reason"] is None
             and receipt["inputs_reverified_after_worker"] is True,
             "resource trial did not finish with reverified inputs")
    full = candidate["receipt"]
    for name in ("source", "runtime", "initial_checkpoint_sha256", "reference_manifest_sha256", "model_card_sha256"):
        _require(_same(receipt[name], full[name]), f"resource trial/full-run {name} differ")
    _require(_same(shared_settings(receipt), shared_settings(full)), "resource trial/full-run settings differ")
    stage_path = _contained_path(run, "staging-manifest.json")
    _require(file_sha256(stage_path) == receipt["staging_manifest_sha256"], "resource trial staging hash differs")
    staging = _read_json(stage_path)
    variants = _check_staging(preflight, staging)
    _require(_same(staging["source"], candidate["staging"]["source"])
             and receipt["data_yaml_sha256"] == variants["candidate"]["data_yaml_sha256"]
             and receipt["training_settings"]["data"] == str(Path(staging["output_root"]) / variants["candidate"]["data_yaml"]),
             "resource trial uses different candidate inputs")
    _require(receipt["log"]["path"] == "train.log", "resource trial must identify its original train.log")
    log_path = _artifact(run, receipt["log"])
    _require(log_path.stat().st_size > 0, "resource trial log is empty")
    files = [{"path": f"resource-trial/{name}", "sha256": file_sha256(path),
              "size_bytes": path.stat().st_size, "source_path": path}
             for name, path in (("training-receipt.json", receipt_path),
                                ("staging-manifest.json", stage_path), ("train.log", log_path))]
    return {"purpose": "resource-trial", "status": "trial-complete", "seed": 43, "arm": "candidate",
            "batches_completed": 4, "optimizer_steps": receipt["optimizer_steps"],
            "eligible_checkpoint": False, **expected,
            "files": [row["path"] for row in files]}, files


def package_repeatability(root, runs, output, *, trial=None):
    """Require all four exact completed runs before publishing one flat ZIP."""
    root = Path(root).resolve()
    _require(set(runs) == set(SEEDS) and all(set(runs[seed]) == set(ARMS) for seed in SEEDS),
             "the study requires exactly reference/candidate runs for seeds 43 and 44")
    locations = {}
    for seed in SEEDS:
        for arm in ARMS:
            path = Path(runs[seed][arm])
            _require(not path.is_symlink() and path.is_dir(), f"seed {seed} {arm}: missing or linked run directory")
            locations[seed, arm] = path.resolve()
    trial_path = None
    if trial is not None:
        trial_path = Path(trial)
        _require(not trial_path.is_symlink() and trial_path.is_dir(), "resource trial directory is missing or linked")
        trial_path = trial_path.resolve()
    protected = [*locations.values(), *([trial_path] if trial_path else [])]
    _require(not any(first.is_relative_to(second) for i, first in enumerate(protected)
                     for j, second in enumerate(protected) if i != j), "study run directories must not overlap")
    output = Path(output)
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"output already exists; preserve it and choose a new ZIP: {output}")
    output = output.resolve()
    _require(output.suffix.lower() == ".zip", "output must have a .zip extension")
    _require(not any(output.is_relative_to(path) for path in (root, *protected)),
             "output must be outside the source and every run directory")
    _require(file_sha256(_contained_path(root, CANDIDATE_MANIFEST)) == CANDIDATE_SHA256,
             "the repeatability study requires the exact reviewed box-candidate manifest")
    preflight = preflight_dataset(root, require_weights=False)
    pairs, rows = {}, []
    for seed in SEEDS:
        pair = {arm: collect_arm(preflight, locations[seed, arm], arm) for arm in ARMS}
        for bundle in pair.values():
            check_study_arm(bundle, seed)
        check_pair(pair["reference"], pair["candidate"])
        pairs[seed] = pair
        rows.extend({**row, "path": f"seed-{seed}/{row['path']}"}
                    for bundle in pair.values() for row in bundle["files"])
    first, second = pairs[43]["reference"], pairs[44]["reference"]
    comparisons = (("training source", first["receipt"]["source"], second["receipt"]["source"]),
                   ("runtime", first["receipt"]["runtime"], second["receipt"]["runtime"]),
                   ("dataset proposal", first["staging"]["source"], second["staging"]["source"]),
                   ("settings other than seed/run paths", study_settings(first["receipt"]), study_settings(second["receipt"])))
    for name, before, after in comparisons:
        _require(_same(before, after), f"cross-seed {name} differ; return one unchanged study environment")
    trial_index = None
    if trial_path is not None:
        trial_index, trial_rows = collect_trial(preflight, trial_path, pairs[43]["candidate"])
        rows.extend(trial_rows)
    rows.sort(key=lambda row: row["path"])
    index = {"schema_version": 1, "kind": "baomai-box-repeatability-results", "study_id": STUDY_ID,
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "seeds": list(SEEDS),
        "source": first["staging"]["source"], "training_source": first["receipt"]["source"],
        "runtime": first["receipt"]["runtime"], "shared_training_settings": study_settings(first["receipt"]),
        "pairs": {f"seed-{seed}": {"seed": seed, **recipe_identity(BOX_REPEATABILITY_RECIPES[seed]),
            "arms": {arm: {"checkpoint_manifest": f"seed-{seed}/{arm}/checkpoint-manifest.json",
                           "checkpoint_identity": pairs[seed][arm]["identity"]} for arm in ARMS}} for seed in SEEDS},
        "resource_trial": trial_index,
        "files": [{key: value for key, value in row.items() if key != "source_path"} for row in rows],
        "verification_scope": "Four completed paired runs; exact declared seeds/recipes and candidate; matched source/runtime and settings except seed/run paths. No model loading, inference, or adoption approval."}
    index_bytes = (json.dumps(index, indent=2, allow_nan=False) + "\n").encode("utf-8")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".box-repeatability-", dir=output.parent) as temporary:
        archive_path = Path(temporary) / "results.zip"
        with zipfile.ZipFile(archive_path, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=1) as archive:
            archive.writestr(INDEX_NAME, index_bytes)
            for row in rows:
                copy_member(archive, row)
        verify_archive(archive_path, index_bytes, rows)
        os.link(archive_path, output)
    return {"archive": str(output), "sha256": file_sha256(output), "size_bytes": output.stat().st_size,
            "files": len(rows) + 1, "study_id": STUDY_ID, "seeds": list(SEEDS), "pairs": index["pairs"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--seed-43", type=Path, nargs=2, metavar=("REFERENCE", "CANDIDATE"), required=True)
    parser.add_argument("--seed-44", type=Path, nargs=2, metavar=("REFERENCE", "CANDIDATE"), required=True)
    parser.add_argument("--trial", type=Path, help="optional completed four-batch seed-43 candidate resource trial")
    parser.add_argument("--output", type=Path, required=True, help="new ZIP outside all source/run directories")
    args = parser.parse_args()
    runs = {seed: dict(zip(ARMS, getattr(args, f"seed_{seed}"))) for seed in SEEDS}
    try:
        result = package_repeatability(args.root, runs, args.output, trial=args.trial)
    except (ValueError, OSError, KeyError, TypeError, zipfile.BadZipFile) as error:
        print(f"Repeatability return package refused: {error}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
