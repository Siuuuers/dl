#!/usr/bin/env python3
"""Run one R31-label ordinary-sampling partial-target box-loss comparison role."""

import sys
sys.dont_write_bytecode = True

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import shutil

from evaluate_pose import ULTRALYTICS_VERSION, file_sha256, preflight_dataset
from partial_box_loss_r31 import (DEFAULT_PROTOCOL, ROLES, PartialBoxLoss, experiment_identity,
                                    load_protocol, source_identity, validate_state, verify_run, verify_native_sources)
from pose_checkpoint import recipe_identity, training_label_identity, write_checkpoint_manifest
from stage_pose_training import MANIFEST, stage_datasets
from train_pose_comparison import descriptor, monitor_worker, run_worker, verify_input_files, write_json
from verify_pose_candidate import require


ROOT = Path(__file__).resolve().parents[1]


def worker(config):
    root = Path(config["root"]).resolve()
    require(root == ROOT, "worker root differs from executing repository")
    context = load_protocol(root, config["protocol"])
    document = context["document"]
    require(config["experiment"] == experiment_identity(context, config["role"]), "worker partial-target loss identity differs")
    require(config["arm"] == "candidate" and config["candidate_manifest"] == document["candidate_manifest"],
            "both loss roles require the exact eligible R31 candidate labels")
    require(all(config[key] == value for key, value in recipe_identity(document["recipe_id"]).items()),
            "worker numerical recipe differs")
    require(file_sha256(config["initial_weights"]) == document["initial_checkpoint_sha256"], "initializer differs")
    return run_worker(config, loss_controller=PartialBoxLoss(root, context, config["role"]))


def run(args):
    root, output = args.root.resolve(), args.output.absolute()
    require(root == ROOT, "--root must identify the executing repository")
    context = load_protocol(root, args.protocol)
    document = context["document"]
    require(not output.exists() and not output.is_symlink(), "run output must be a fresh directory")
    output = output.resolve()
    require(not output.is_relative_to(root) and not root.is_relative_to(output), "run output must be outside source")
    require(args.threads == 4 and args.max_seconds > 0 and args.max_rss_mib > 0,
            "use four threads and positive resource limits")
    require(args.trial_batches == document["execution"]["resource_trial_batches"], "resource-trial budget differs")
    require(args.device == "cpu" or args.device.isdecimal(), "select cpu or one CUDA index explicitly")
    require(args.trial or args.device.isdecimal(), "full comparison requires a CUDA device")
    require(importlib.metadata.version("ultralytics") == ULTRALYTICS_VERSION, "pinned Ultralytics required")
    verify_native_sources()
    preflight = preflight_dataset(root, require_weights=True)
    require(preflight["model_sha256"] == document["initial_checkpoint_sha256"], "initial checkpoint differs")
    source = source_identity(root)
    require(args.trial or not source["working_tree_status"], "full comparison requires clean source")
    numerical, identity = recipe_identity(document["recipe_id"]), experiment_identity(context, args.role)
    config = {"root": str(root), "output": str(output), "trial": args.trial,
              "trial_batches": args.trial_batches, "arm": "candidate", "role": args.role,
              "device": args.device, "threads": args.threads, "max_seconds": args.max_seconds,
              "max_rss_mib": args.max_rss_mib, "initial_weights": preflight["weights"],
              "candidate_manifest": document["candidate_manifest"],
              "protocol": str(context["path"].relative_to(root)), "experiment": identity, **numerical}
    output.mkdir(parents=True)
    stage_datasets(root, root / document["candidate_manifest"], output / "staged-data")
    shutil.copyfile(output / "staged-data" / MANIFEST, output / MANIFEST)
    staging = json.loads((output / MANIFEST).read_text())
    labels = training_label_identity(staging, "candidate")
    require(labels is not None, "accepted R31 supervision identity is missing")
    config["training_labels"] = labels
    write_json(output / "run-config.json", config)
    execution = monitor_worker([sys.executable, str(Path(__file__).resolve()),
                                "--worker", str(output / "run-config.json")], output, config)
    progress_path = output / "training-progress.json"
    progress = json.loads(progress_path.read_text()) if progress_path.exists() else {"status": "failed"}
    receipt = {**progress, "schema_version": 2,
               "purpose": "resource-trial" if args.trial else "paired-finetune",
               "created_at_utc": datetime.now(timezone.utc).isoformat(), "arm": "candidate",
               "experiment": identity, "initial_checkpoint_sha256": preflight["model_sha256"],
               "reference_manifest_sha256": preflight["manifest_sha256"],
               "model_card_sha256": preflight["model_card_sha256"], "training_labels": labels,
               "staging_manifest_sha256": file_sha256(output / MANIFEST),
               "data_yaml_sha256": staging["variants"]["candidate"]["data_yaml_sha256"],
               "source": source, "execution": execution, **numerical,
               "checkpoint_selection": "none" if args.trial else "final_epoch",
               "inputs_reverified_after_worker": False, "log": descriptor(output / "train.log", output)}
    valid = execution["exit_code"] == 0 and execution["stop_reason"] is None
    try:
        require(all(progress.get(key) == value for key, value in numerical.items()), "worker recipe identity differs")
        require(progress.get("training_labels") == labels, "worker R31 supervision identity differs")
        state = progress["partial_target_loss"]
        validate_state(state, root, context, args.role, complete=not args.trial)
        require(state["images_seen"] == progress["images_processed"]
                and state["batches_seen"] == progress["batches_completed"], "loss instrumentation did not observe every batch")
        if args.trial:
            require(progress["batches_completed"] == args.trial_batches
                    and progress["status"] == "trial-complete", "resource trial did not complete")
        else:
            require(progress["optimizer_steps"] == document["execution"]["expected_optimizer_steps"],
                    "optimizer-update budget differs")
        verify_input_files(output, staging)
        require(source == source_identity(root), "source or protocol changed during execution")
        require(file_sha256(preflight["weights"]) == preflight["model_sha256"], "initializer changed")
        require(file_sha256(root / "data/manifest.json") == preflight["manifest_sha256"], "source manifest changed")
        require(file_sha256(root / document["candidate_manifest"]) == staging["source"]["candidate_manifest_sha256"],
                "R31 candidate manifest changed")
        receipt["inputs_reverified_after_worker"] = True
    except (ValueError, OSError, KeyError, TypeError) as error:
        valid = False
        receipt["input_verification_error"] = str(error)
    expected_status = "trial-complete" if args.trial else "complete"
    valid = valid and receipt["status"] == expected_status
    if not valid:
        receipt["status"] = "failed"
    if valid and not args.trial:
        receipt["checkpoint"] = descriptor(output / "fit/weights/last.pt", output, checkpoint=True)
        receipt["results"] = descriptor(output / "fit/results.csv", output)
    write_json(output / "training-receipt.json", receipt)
    if valid and not args.trial:
        try:
            write_checkpoint_manifest(output, preflight)
            verify_run(root, output, args.role, context)
        except (ValueError, OSError, KeyError, TypeError) as error:
            receipt["status"] = "failed"
            receipt["checkpoint_manifest_error"] = str(error)
            write_json(output / "training-receipt.json", receipt)
            raise
    print(json.dumps({"output": str(output), "role": args.role, "purpose": receipt["purpose"],
                      "status": receipt["status"], "execution": execution,
                      "batches_completed": receipt.get("batches_completed", 0),
                      "optimizer_steps": receipt.get("optimizer_steps", 0),
                      "checkpoint_manifest_created": valid and not args.trial}, allow_nan=False))
    return 0 if valid else 1


def main(argv=None):
    arguments = list(sys.argv[1:] if argv is None else argv)
    if len(arguments) == 2 and arguments[0] == "--worker":
        return worker(json.loads(Path(arguments[1]).read_text()))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--protocol", default=DEFAULT_PROTOCOL)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--role", choices=ROLES, required=True)
    parser.add_argument("--device", required=True, help="CUDA index; CPU is permitted for a resource trial")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--trial", action="store_true")
    mode.add_argument("--train", action="store_true")
    parser.add_argument("--trial-batches", type=int, default=4)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--max-seconds", type=int, required=True)
    parser.add_argument("--max-rss-mib", type=int, default=6000)
    args = parser.parse_args(arguments)
    try:
        return run(args)
    except (ValueError, OSError, KeyError, TypeError, importlib.metadata.PackageNotFoundError) as error:
        print(f"{type(error).__name__}: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
