#!/usr/bin/env python3
"""R32: prespecified seed47 replication of the unchanged R31 box-loss comparison."""

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import zipfile

sys.dont_write_bytecode = True

from evaluate_pose import _relative_file, file_sha256
from package_gpu_results import INDEX_NAME, check_pair, collect_arm, copy_member, shared_settings, verify_archive
from pose_checkpoint import _same, recipe_identity, get_recipe, INCREMENTAL_CONFIRMATION_RECIPE_ID
from partial_box_loss_r31 import (PartialBoxLoss, make_criterion, verify_native_sources, training_inventory,
                                  validate_state, metadata_preflight, NATIVE_SOURCE_HASHES,
                                  SPEC, CRITERION_CONTRACT, COUNT_FIELDS, BRANCHES)
from predict_pose_folder import BASELINE_SHA256, _reference_model
from restore_gpu_inputs import read_json, require


DEFAULT_PROTOCOL = "experiments/partial-box-replication-2026-11-25-r32.json"
PROTOCOL_SHA256 = "68874ae3d4f6ce9d41131a98e887bcac0bd41aeed96f2fc409294530596880e0"
ROLES = ("control", "candidate")


def digest_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def load_protocol(root, path=DEFAULT_PROTOCOL):
    root, path = Path(root).resolve(), Path(path)
    path = path.resolve() if path.is_absolute() else _relative_file(root, str(path))
    require(path == root / DEFAULT_PROTOCOL, "partial-target protocol path differs")
    sha = file_sha256(path)
    require(PROTOCOL_SHA256 is None or sha == PROTOCOL_SHA256, "partial-target protocol differs")
    document = read_json(path)
    require(document["schema_version"] == 1 and document["kind"] == "baomai-partial-box-loss-protocol"
            and document["experiment_id"] == "partial-box-replication-r32", "wrong seed47 replication experiment")
    require(document["recipe_id"] == INCREMENTAL_CONFIRMATION_RECIPE_ID
            and get_recipe(document["recipe_id"])["seed"] == 47, "prespecified replication seed differs")
    require(document["candidate_manifest"] == "data/candidates/front-review-2026-11-24-r31/manifest.json"
            and document["label_arm_for_both_experimental_roles"] == "candidate", "both roles require R31 labels")
    require(all(document[key] == value for key, value in recipe_identity(document["recipe_id"]).items()),
            "base numerical recipe differs")
    require(document["initial_checkpoint_sha256"] == BASELINE_SHA256, "epoch173 initializer differs")
    for relative, expected in (("data/manifest.json", document["source_manifest_sha256"]),
                               (document["candidate_manifest"], document["candidate_manifest_sha256"])):
        require(file_sha256(_relative_file(root, relative)) == expected, "source identity differs: " + relative)
    accepted = document["accepted_review"]
    require(file_sha256(_relative_file(root, accepted["path"])) == accepted["sha256"], "accepted R31 review differs")
    require(all(document["loss_weighting"].get(key) == value for key, value in SPEC.items()), "fixed loss treatment differs")
    execution = document["execution"]
    require(execution["epochs"] == 30 and execution["resource_trial_batches"] == 4
            and execution["expected_batches"] == 1050 and execution["expected_images"] == 4110
            and execution["expected_optimizer_steps"] == 78 and execution["same_runtime_required"] is True
            and execution["automatic_promotion"] is False, "fixed experiment budget differs")
    evaluation = document["evaluation"]
    for prefix in ("primary", "diagnostic"):
        require(file_sha256(_relative_file(root, evaluation[prefix + "_reference"]))
                == evaluation[prefix + "_reference_sha256"], "frozen evaluation reference differs")
    current = evaluation["current_review_reference"]
    require(file_sha256(_relative_file(root, current["path"])) == current["sha256"], "R32 diagnostic review differs")
    from front_review_r32 import verify_review
    verify_review(root)
    require(document["replication"]["report_both_seeds"] is True
            and document["replication"]["select_better_seed"] is False
            and document["replication"]["previous_seed"] == 46, "replication reporting contract differs")
    return {"document": document, "sha256": sha, "path": path}


def source_identity(root):
    from train_pose_comparison import source_identity as base_source
    root = Path(root).resolve()
    result = base_source(root)
    for name in ("scripts/partial_box_loss_r31.py", "scripts/train_pose_partial_box_loss_r31.py",
                 "scripts/partial_box_replication_r32.py", "scripts/train_pose_partial_box_replication_r32.py",
                 "scripts/front_review_r32.py", "experiments/partial-box-loss-2026-11-24-r31.json", DEFAULT_PROTOCOL):
        result["files_sha256"][name] = file_sha256(_relative_file(root, name))
    return result


def experiment_identity(context, role):
    require(role in ROLES, "unknown partial-target role")
    document = context["document"]
    return {"id": document["experiment_id"], "role": role, "protocol_sha256": context["sha256"],
            "base_recipe_id": document["recipe_id"], "base_recipe_sha256": document["recipe_sha256"],
            "label_arm": "candidate", "label_manifest": document["candidate_manifest"],
            "label_manifest_sha256": document["candidate_manifest_sha256"],
            "loss_factor": document["loss_weighting"][role + "_factor"],
            "loss_weighting_spec_sha256": digest_json(document["loss_weighting"])}


def verify_trial(root, run, role, context=None):
    from pose_checkpoint import (_check_runtime_v2, _check_staging, _contained_path,
                                 check_training_settings, training_label_identity)
    from stage_pose_training import MANIFEST
    root, run = Path(root).resolve(), Path(run).resolve()
    context = context or load_protocol(root)
    receipt = read_json(run / "training-receipt.json")
    require(receipt["purpose"] == "resource-trial" and receipt["status"] == "trial-complete"
            and receipt["arm"] == "candidate" and receipt["checkpoint_selection"] == "none", "not a completed loss resource trial")
    require(_same(receipt["experiment"], experiment_identity(context, role)), "trial experiment differs")
    validate_state(receipt["partial_target_loss"], root, context, role, complete=False)
    state, document = receipt["partial_target_loss"], context["document"]
    expected = document["execution"]["resource_trial_batches"]
    require(receipt["batches_completed"] == state["batches_seen"] == expected
            and receipt["images_processed"] == state["images_seen"] == expected * 4
            and receipt["optimizer_steps"] > 0 and receipt["optimizer_parameter_change_observed"] is True,
            "trial did not establish real training exposure and updates")
    require(receipt["execution"]["exit_code"] == 0 and receipt["execution"]["stop_reason"] is None
            and receipt["inputs_reverified_after_worker"] is True, "trial stopped or lost verified inputs")
    require(receipt["source"]["files_sha256"] == source_identity(root)["files_sha256"], "trial source differs")
    require(all(receipt[key] == value for key, value in recipe_identity(document["recipe_id"]).items())
            and receipt["initial_checkpoint_sha256"] == document["initial_checkpoint_sha256"], "trial base recipe differs")
    staging_path = _contained_path(run, MANIFEST)
    require(file_sha256(staging_path) == receipt["staging_manifest_sha256"], "trial staging hash differs")
    staging, preflight = read_json(staging_path), metadata_preflight(root)
    variants = _check_staging(preflight, staging)
    require(receipt["reference_manifest_sha256"] == preflight["manifest_sha256"]
            and receipt["model_card_sha256"] == preflight["model_card_sha256"], "trial source metadata differs")
    require(staging["source"]["candidate_manifest"] == document["candidate_manifest"]
            and staging["source"]["candidate_manifest_sha256"] == document["candidate_manifest_sha256"]
            and _same(receipt["training_labels"], training_label_identity(staging, "candidate")), "trial label arm differs")
    variant = variants["candidate"]
    require(receipt["data_yaml_sha256"] == variant["data_yaml_sha256"], "trial selected another dataset")
    stage_root = Path(staging["output_root"])
    require(stage_root.is_absolute() and ".." not in stage_root.parts and stage_root.name == "staged-data",
            "trial staging location differs")
    settings, model = receipt["training_settings"], Path(receipt["training_settings"]["model"])
    relative = Path(preflight["weights"]).relative_to(root)
    require(model.is_absolute() and ".." not in model.parts and model.parts[-len(relative.parts):] == relative.parts,
            "trial initializer location differs")
    check_training_settings(settings, document["recipe_id"], model=model,
                            data=stage_root / variant["data_yaml"], device=settings["device"], project=stage_root.parent)
    _check_runtime_v2(receipt["runtime"], settings)
    log = _contained_path(run, receipt["log"]["path"])
    require(log.stat().st_size > 0 and file_sha256(log) == receipt["log"]["sha256"], "trial log hash differs")
    require(len(receipt["batch_samples"]) == expected and len(state["epochs"]) == 1, "trial batch observations differ")
    for index, sample in enumerate(receipt["batch_samples"]):
        shape = sample["shape"]
        require(len(shape) == 4 and shape[:2] == [4, 3] and min(shape[2:]) > 0 and max(shape[2:]) == 1280
                and sample["epoch"] == 1 and sample["batch"] == index + 1
                and sample["image_ids"] == state["epochs"][0]["batches"][index]["image_ids"], "trial batch geometry differs")
    return receipt



def verify_run(root, run, role, context=None, require_weights=True):
    root, run = Path(root).resolve(), Path(run).resolve()
    require(require_weights is True, "returned loss runs require actual checkpoint bytes")
    context = context or load_protocol(root)
    receipt = read_json(run / "training-receipt.json")
    require(_same(receipt.get("experiment"), experiment_identity(context, role)), "completed loss role differs")
    state = receipt["partial_target_loss"]
    validate_state(state, root, context, role, complete=True)
    require(state["images_seen"] == receipt["images_processed"] and state["batches_seen"] == receipt["batches_completed"]
            and receipt["optimizer_steps"] == context["document"]["execution"]["expected_optimizer_steps"], "loss run budget differs")
    require(receipt["source"]["files_sha256"] == source_identity(root)["files_sha256"], "completed loss implementation differs")
    bundle = collect_arm(metadata_preflight(root), run, "candidate")
    staging, document = bundle["staging"], context["document"]
    require(receipt["training_settings"]["device"].isdecimal() and receipt["runtime"]["device"].startswith("cuda:")
            and receipt["runtime"]["threads"] == 4, "full loss comparison requires fixed CUDA/four-thread runtime")
    require(staging["source"]["candidate_manifest"] == document["candidate_manifest"]
            and staging["source"]["candidate_manifest_sha256"] == document["candidate_manifest_sha256"], "completed R31 labels differ")
    # The pinned native saver serializes EMA and removes its criterion. Verify the
    # returned pickle has no dependency on either experiment module.
    with zipfile.ZipFile(run / "fit/weights/last.pt") as checkpoint:
        pickles = [checkpoint.read(name) for name in checkpoint.namelist() if name.endswith("data.pkl")]
    require(len(pickles) == 1 and all(token not in pickles[0] for token in
            (b"partial_box_loss_r31", b"train_pose_partial_box_loss_r31",
             b"partial_box_replication_r32", b"train_pose_partial_box_replication_r32", b"PartialBoxLoss")), "custom loss leaked into checkpoint")
    for entry in bundle["files"]:
        entry["path"] = role + "/" + entry["path"].split("/", 1)[1]
    return bundle


def package_pair(root, control, candidate, output, verify_existing=False, protocol=DEFAULT_PROTOCOL):
    root = Path(root).resolve()
    context = load_protocol(root, protocol)
    runs = {"control": Path(control).resolve(), "candidate": Path(candidate).resolve()}
    require(all(path.is_dir() and not Path(raw).is_symlink() for raw, path in
                ((control, runs["control"]), (candidate, runs["candidate"]))), "run directories must be regular")
    require(not runs["control"].is_relative_to(runs["candidate"]) and not runs["candidate"].is_relative_to(runs["control"]),
            "run directories overlap")
    bundles = {role: verify_run(root, run, role, context) for role, run in runs.items()}
    check_pair(bundles["control"], bundles["candidate"])
    receipts = {role: bundle["receipt"] for role, bundle in bundles.items()}
    require(receipts["control"]["training_labels"] == receipts["candidate"]["training_labels"], "paired loss labels differ")
    for a, b in zip(receipts["control"]["partial_target_loss"]["epochs"], receipts["candidate"]["partial_target_loss"]["epochs"]):
        require(a["observed_image_ids"] == b["observed_image_ids"], "ordinary paired image order differs")
        require(all(x["targets"] == y["targets"] and x["partial_targets"] == y["partial_targets"]
                    for x, y in zip(a["batches"], b["batches"])), "paired post-augmentation target cohorts differ")
    rows = [entry for bundle in bundles.values() for entry in bundle["files"]]
    rows.append({"path": "experiment-protocol.json", "sha256": context["sha256"],
                 "size_bytes": context["path"].stat().st_size, "source_path": context["path"]})
    rows.sort(key=lambda row: row["path"])
    receipt = receipts["control"]
    index = {"schema_version": 1, "kind": "baomai-partial-box-loss-results", "created_at_utc": datetime.now(timezone.utc).isoformat(),
             "protocol_sha256": context["sha256"], "source": receipt["source"], "runtime": receipt["runtime"],
             "shared_base_settings": shared_settings(receipt),
             "roles": {role: {"checkpoint_manifest": f"{role}/checkpoint-manifest.json", "experiment": bundle["receipt"]["experiment"],
                              "partial_target_loss": bundle["receipt"]["partial_target_loss"], "checkpoint_identity": bundle["identity"]}
                       for role, bundle in bundles.items()},
             "files": [{key: value for key, value in row.items() if key != "source_path"} for row in rows],
             "scope": "Completed fixed-budget partial-target box-regression comparison; classification objective unchanged. No accuracy approval or model promotion."}
    output = Path(output).absolute()
    require(not output.is_symlink() and output.suffix.lower() == ".zip", "output must be a regular ZIP path")
    output = output.resolve()
    require(all(not output.is_relative_to(path) for path in (root, *runs.values())), "ZIP overlaps source/run directories")
    if verify_existing:
        with zipfile.ZipFile(output) as archive:
            previous = json.loads(archive.read(INDEX_NAME))
        index["created_at_utc"] = previous["created_at_utc"]
        require(_same(previous, index), "existing loss return index differs")
    else:
        require(not output.exists(), "return ZIP already exists")
    index_bytes = (json.dumps(index, indent=2, allow_nan=False) + "\n").encode()
    if verify_existing:
        verify_archive(output, index_bytes, rows)
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".partial-loss-return-", dir=output.parent) as temporary:
            staged = Path(temporary) / "results.zip"
            with zipfile.ZipFile(staged, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=1) as archive:
                archive.writestr(INDEX_NAME, index_bytes)
                for entry in rows:
                    copy_member(archive, entry)
            verify_archive(staged, index_bytes, rows)
            os.link(staged, output)
    return {"archive": str(output), "sha256": file_sha256(output), "size_bytes": output.stat().st_size,
            "files": len(rows) + 1, "status": "verified", "accuracy_evaluated": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("verify", "package"):
        item = sub.add_parser(name)
        item.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
        item.add_argument("--protocol", default=DEFAULT_PROTOCOL)
        if name == "package":
            for option in ("control", "candidate", "output"):
                item.add_argument("--" + option, type=Path, required=True)
            item.add_argument("--verify-existing", action="store_true")
    args = parser.parse_args()
    if args.command == "verify":
        context = load_protocol(args.root, args.protocol)
        result = {"status": "verified", "protocol_sha256": context["sha256"],
                  "inventory": training_inventory(args.root, context), "training_executed": False}
    else:
        result = package_pair(args.root, args.control, args.candidate, args.output, args.verify_existing, args.protocol)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
