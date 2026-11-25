#!/usr/bin/env python3
"""Verify and package the native-loss R31-versus-R33 supervision comparison."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import tempfile
import zipfile

sys.dont_write_bytecode = True
from evaluate_pose import _relative_file, file_sha256
from front_review_r33_training import verify_candidate, PROTOCOL_PATH
from package_gpu_results import INDEX_NAME, check_pair, collect_arm, copy_member, shared_settings, verify_archive
from pose_checkpoint import _same, recipe_identity, training_label_identity
from predict_pose_folder import BASELINE_SHA256, _reference_model
from restore_gpu_inputs import read_json, require
from train_pose_comparison import source_identity

DEFAULT_PROTOCOL = PROTOCOL_PATH
PROTOCOL_SHA256 = "9782658c99d56740af0302c4edd80a071e169ca404c31b01948c7a2092d078ed"
ROLES = ("reference", "candidate")


def load_protocol(root, path=DEFAULT_PROTOCOL):
    root = Path(root).resolve()
    path = Path(path)
    path = path.resolve() if path.is_absolute() else _relative_file(root, str(path))
    require(path == root/DEFAULT_PROTOCOL and file_sha256(path) == PROTOCOL_SHA256,
            "R33 protocol identity differs")
    document = read_json(path)
    require(document["experiment_id"] == "front-review-r33"
            and document["recipe_id"] == "reviewed-incremental-fronts-seed46-v1"
            and all(document[k] == v for k, v in recipe_identity(document["recipe_id"]).items()),
            "Native comparison numerical recipe differs")
    require(document["initial_checkpoint_sha256"] == BASELINE_SHA256
            and document["execution"]["loss"] == "unmodified_native_Pose26"
            and document["execution"]["partial_box_factor"] == 1
            and document["execution"]["automatic_promotion"] is False, "Native comparison contract differs")
    verified = verify_candidate(root, root/document["candidate"]["path"])
    require(verified["training_input_delta"]["report"]["summary"]["changed_rows"] > 0,
            "Identical-label comparison is not useful")
    require(set(document["roles"]) == set(ROLES), "Role inventory differs")
    for prefix in ("primary", "diagnostic"):
        e = document["evaluation"]
        require(file_sha256(_relative_file(root, e[prefix+"_reference"])) == e[prefix+"_reference_sha256"],
                "Frozen evaluation reference differs")
    return {"document": document, "path": path, "sha256": file_sha256(path), "candidate_verification": verified}


def metadata_preflight(root):
    root = Path(root).resolve()
    weights, card_hash = _reference_model(root)
    return {"root": str(root), "weights": str(weights), "model_sha256": BASELINE_SHA256,
            "model_card_sha256": card_hash, "manifest_sha256": file_sha256(root/"data/manifest.json")}


def check_common(root, receipt, staging, role, context):
    document = context["document"]
    require(role in ROLES and receipt["arm"] == role, "Label role differs")
    require("partial_target_loss" not in receipt and "experiment" not in receipt,
            "Native run unexpectedly used a custom loss experiment")
    require(receipt["source"]["files_sha256"] == source_identity(root)["files_sha256"], "Training source differs")
    require(all(receipt[k] == v for k, v in recipe_identity(document["recipe_id"]).items())
            and receipt["initial_checkpoint_sha256"] == BASELINE_SHA256, "Recipe or initializer differs")
    require(receipt["execution"]["exit_code"] == 0 and receipt["execution"]["stop_reason"] is None
            and receipt["inputs_reverified_after_worker"] is True, "Run stopped or lost input verification")
    require(staging["source"]["candidate_manifest"] == document["candidate"]["path"]
            and staging["source"]["candidate_manifest_sha256"] == document["candidate"]["sha256"],
            "Staged R33 comparison differs")
    require(_same(receipt["training_labels"], training_label_identity(staging, role)), "Selected supervision differs")
    require(staging["variants"][role]["counts"] == document["roles"][role]["counts"], "Role training counts differ")
    require(receipt["training_labels"]["training_protocol"] ==
            {"path": context["path"].relative_to(root).as_posix(), "sha256": context["sha256"]},
            "Staged training protocol differs")


def verify_trial(root, run, role="candidate", context=None):
    from pose_checkpoint import (_check_runtime_v2, _check_staging, _contained_path, check_training_settings)
    from stage_pose_training import MANIFEST
    root, run = Path(root).resolve(), Path(run).resolve()
    context = context or load_protocol(root)
    receipt = read_json(run/"training-receipt.json")
    require(receipt["purpose"] == "resource-trial" and receipt["status"] == "trial-complete"
            and receipt["checkpoint_selection"] == "none", "Not a completed native resource trial")
    path = _contained_path(run, MANIFEST)
    require(file_sha256(path) == receipt["staging_manifest_sha256"], "Trial staging hash differs")
    staging, preflight = read_json(path), metadata_preflight(root)
    variants = _check_staging(preflight, staging)
    check_common(root, receipt, staging, role, context)
    expected = context["document"]["execution"]["resource_trial"]["batches"]
    require(receipt["batches_completed"] == expected == 4 and receipt["images_processed"] == 16
            and receipt["optimizer_steps"] > 0 and receipt["optimizer_parameter_change_observed"] is True,
            "Trial did not establish real image exposure and parameter updates")
    require(receipt["reference_manifest_sha256"] == preflight["manifest_sha256"]
            and receipt["model_card_sha256"] == preflight["model_card_sha256"], "Trial source metadata differs")
    variant = variants[role]
    require(receipt["data_yaml_sha256"] == variant["data_yaml_sha256"], "Trial dataset differs")
    stage_root = Path(staging["output_root"])
    settings, model = receipt["training_settings"], Path(receipt["training_settings"]["model"])
    relative = Path(preflight["weights"]).relative_to(root)
    require(stage_root.is_absolute() and ".." not in stage_root.parts and stage_root.name == "staged-data"
            and model.is_absolute() and ".." not in model.parts
            and model.parts[-len(relative.parts):] == relative.parts, "Trial input location differs")
    check_training_settings(settings, context["document"]["recipe_id"], model=model,
                            data=stage_root/variant["data_yaml"], device=settings["device"], project=stage_root.parent)
    _check_runtime_v2(receipt["runtime"], settings)
    log = _contained_path(run, receipt["log"]["path"])
    require(log.stat().st_size > 0 and file_sha256(log) == receipt["log"]["sha256"], "Trial log differs")
    require(len(receipt["batch_samples"]) == expected, "Trial batch observations differ")
    for index, sample in enumerate(receipt["batch_samples"]):
        shape = sample["shape"]
        require(len(shape) == 4 and shape[:2] == [4, 3] and min(shape[2:]) > 0 and max(shape[2:]) == 1280
                and sample["epoch"] == 1 and sample["batch"] == index+1
                and len(sample["image_ids"]) == 4, "Trial batch geometry or image IDs differ")
    return receipt


def verify_run(root, run, role, context=None):
    root, run = Path(root).resolve(), Path(run).resolve()
    context = context or load_protocol(root)
    bundle = collect_arm(metadata_preflight(root), run, role)
    receipt, staging = bundle["receipt"], bundle["staging"]
    check_common(root, receipt, staging, role, context)
    require(receipt["purpose"] == "paired-finetune" and receipt["status"] == "complete"
            and receipt["checkpoint_selection"] == "final_epoch", "Native run is incomplete")
    require(receipt["batches_completed"] == 1050 and receipt["images_processed"] == 4110
            and receipt["optimizer_steps"] == 78, "Native fixed budget differs")
    require(receipt["training_settings"]["device"].isdecimal()
            and receipt["runtime"]["device"].startswith("cuda:")
            and receipt["runtime"]["threads"] == 4, "Full comparison requires one CUDA device and four threads")
    return bundle


def package_pair(root, reference, candidate, output, verify_existing=False, protocol=DEFAULT_PROTOCOL, context=None):
    root = Path(root).resolve()
    context = context or load_protocol(root, protocol)
    runs = {"reference": Path(reference).resolve(), "candidate": Path(candidate).resolve()}
    require(all(p.is_dir() and not Path(raw).is_symlink() for raw, p in
                ((reference, runs["reference"]), (candidate, runs["candidate"]))), "Run directories must be regular")
    require(not runs["reference"].is_relative_to(runs["candidate"])
            and not runs["candidate"].is_relative_to(runs["reference"]), "Run directories overlap")
    bundles = {role: verify_run(root, run, role, context) for role, run in runs.items()}
    check_pair(bundles["reference"], bundles["candidate"])
    require([s["image_ids"] for s in bundles["reference"]["receipt"]["batch_samples"]]
            == [s["image_ids"] for s in bundles["candidate"]["receipt"]["batch_samples"]],
            "Observed initial batch image order differs between arms")
    rows = [item for bundle in bundles.values() for item in bundle["files"]]
    for name, path in (("experiment-protocol.json", context["path"]),
                       ("training-input-delta.json", root/context["document"]["training_input_delta"]["path"])):
        rows.append({"path": name, "sha256": file_sha256(path), "size_bytes": path.stat().st_size, "source_path": path})
    rows.sort(key=lambda r: r["path"])
    receipt = bundles["reference"]["receipt"]
    index = {"schema_version": 1, "kind": "baomai-native-front-review-results",
             "created_at_utc": datetime.now(timezone.utc).isoformat(), "protocol_sha256": context["sha256"],
             "source": receipt["source"], "runtime": receipt["runtime"], "shared_settings": shared_settings(receipt),
             "roles": {role: {"checkpoint_manifest": f"{role}/checkpoint-manifest.json",
                              "checkpoint_identity": bundle["identity"],
                              "training_labels": bundle["receipt"]["training_labels"]}
                       for role, bundle in bundles.items()},
             "files": [{k: v for k, v in row.items() if k != "source_path"} for row in rows],
             "scope": ("Native loss in both arms; only the accepted training "
                       + ("boundary" if context["document"]["experiment_id"] == "front-review-r33" else "annotation")
                       + " delta differs. No accuracy approval or model promotion.")}
    output = Path(output).absolute()
    require(not output.is_symlink() and output.suffix.lower() == ".zip", "Output must be a regular ZIP path")
    output = output.resolve()
    require(all(not output.is_relative_to(p) for p in (root, *runs.values())), "ZIP overlaps source or runs")
    if verify_existing:
        with zipfile.ZipFile(output) as archive:
            previous = json.loads(archive.read(INDEX_NAME))
        index["created_at_utc"] = previous["created_at_utc"]
        require(_same(previous, index), "Existing result index differs")
    else:
        require(not output.exists(), "Return ZIP already exists")
    payload = (json.dumps(index, indent=2, allow_nan=False)+"\n").encode()
    if verify_existing:
        verify_archive(output, payload, rows)
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".r33-native-return-", dir=output.parent) as temporary:
            staged = Path(temporary)/"results.zip"
            with zipfile.ZipFile(staged, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=1) as archive:
                archive.writestr(INDEX_NAME, payload)
                for row in rows:
                    copy_member(archive, row)
            verify_archive(staged, payload, rows)
            os.link(staged, output)
    return {"archive": str(output), "sha256": file_sha256(output), "size_bytes": output.stat().st_size,
            "files": len(rows)+1, "status": "verified", "accuracy_evaluated": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--protocol", default=DEFAULT_PROTOCOL)
    args = parser.parse_args()
    context = load_protocol(args.root, args.protocol)
    print(json.dumps({"status": "verified", "protocol_sha256": context["sha256"],
                      "training_delta": context["candidate_verification"]["training_input_delta"]["report"]["summary"],
                      "training_executed": False}, indent=2))
