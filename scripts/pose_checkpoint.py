#!/usr/bin/env python3
"""Verify a completed paired-training bundle without loading model code."""

import csv
import hashlib
import json
import math
from pathlib import Path
import re


RECIPE = {
    "epochs": 30, "resume": False, "imgsz": 1280, "batch": 4, "rect": True,
    "optimizer": "AdamW", "lr0": 0.0001, "lrf": 0.01, "amp": False,
    "seed": 42, "deterministic": True, "patience": 0, "freeze": 0,
    "weight_decay": 0.0005, "momentum": 0.937, "nbs": 64,
    "warmup_epochs": 3.0, "warmup_momentum": 0.8, "warmup_bias_lr": 0.1,
    "mosaic": 0.0, "mixup": 0.0, "cutmix": 0.0, "copy_paste": 0.0,
    "close_mosaic": 0, "degrees": 5.0, "translate": 0.05, "scale": 0.15,
    "shear": 0.0, "perspective": 0.0, "flipud": 0.0, "fliplr": 0.5,
    "hsv_h": 0.01, "hsv_s": 0.2, "hsv_v": 0.2,
    "box": 7.5, "cls": 0.5, "dfl": 1.5, "pose": 12.0, "kobj": 1.0,
    "save_period": 10, "pretrained": True, "fraction": 1.0,
    "multi_scale": 0.0, "cos_lr": False, "cache": False, "compile": False,
    "single_cls": False, "val": True, "split": "val", "save": True,
}
POSE_CONTRACT = {"task": "pose", "class_names": ["cassette_front"],
                 "kpt_shape": [2, 3], "flip_idx": [1, 0]}
ULTRALYTICS_VERSION = "8.4.182"


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _same(actual, expected):
    return json.dumps(actual, sort_keys=True, allow_nan=False) == json.dumps(
        expected, sort_keys=True, allow_nan=False)


def _sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _digest(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _mapping(value, name):
    _require(isinstance(value, dict), f"{name} must be an object")
    return value


def _read_json(path):
    def unique_fields(pairs):
        result = {}
        for key, value in pairs:
            _require(key not in result, f"duplicate JSON field: {key}")
            result[key] = value
        return result

    def reject_constant(value):
        raise ValueError(f"nonstandard JSON constant: {value}")

    return _mapping(json.loads(Path(path).read_text(encoding="utf-8"),
                               object_pairs_hook=unique_fields, parse_constant=reject_constant), str(path))


def _contained_path(base, relative, *, required=True):
    _require(isinstance(relative, str) and bool(relative), "artifact path must be a nonempty string")
    path = Path(relative)
    _require(not path.is_absolute() and ".." not in path.parts, "artifact path must stay inside its bundle")
    base = Path(base).resolve()
    absolute = base / path
    _require(not any(item.is_symlink() for item in (absolute, *absolute.parents) if item != base
                     and item.is_relative_to(base)), "artifact symlinks are not permitted")
    _require(absolute.resolve().is_relative_to(base), "artifact path escapes its bundle")
    if required:
        _require(absolute.is_file(), f"missing artifact: {absolute}")
    return absolute


def _artifact(base, descriptor, *, checkpoint=False, verify_bytes=True):
    descriptor = _mapping(descriptor, "artifact descriptor")
    fields = {"path", "sha256", "size_bytes"} if checkpoint else {"path", "sha256"}
    _require(set(descriptor) == fields and _digest(descriptor["sha256"]), "invalid artifact descriptor")
    path = _contained_path(base, descriptor["path"], required=verify_bytes)
    if checkpoint:
        _require(path.name == "last.pt", "paired comparison requires the finalized last.pt")
        _require(type(descriptor["size_bytes"]) is int and descriptor["size_bytes"] > 0,
                 "checkpoint size must be a positive integer")
    if verify_bytes:
        _require(_sha256(path) == descriptor["sha256"], f"artifact SHA-256 mismatch: {path}")
        if checkpoint:
            _require(path.stat().st_size == descriptor["size_bytes"], "checkpoint size mismatch")
            with path.open("rb") as stream:
                _require(not stream.read(80).startswith(b"version https://git-lfs.github.com/spec/v1"),
                         "checkpoint is a Git LFS pointer")
    return path


def _check_staging(preflight, staging):
    _require(type(staging["schema_version"]) is int and staging["schema_version"] == 1
             and staging["status"] == "complete", "staging manifest is incomplete or unsupported")
    source = _mapping(staging["source"], "staging source")
    for key, expected in (("reference_manifest_sha256", preflight["manifest_sha256"]),
                          ("model_card_sha256", preflight["model_card_sha256"]),
                          ("initial_checkpoint_sha256", preflight["model_sha256"])):
        _require(source[key] == expected, f"staging {key} differs from frozen reference")
    _require(staging["verification"]["validation_images_and_labels_identical"] is True
             and staging["mechanical_geometry_gate"]["passed"] is True,
             "staging lacks successful validation/geometry verification")
    root = Path(preflight["root"])
    _require(_sha256(root / "data/manifest.json") == preflight["manifest_sha256"], "reference manifest changed after preflight")
    reference = _read_json(root / "data/manifest.json")
    proposal_name = source["candidate_manifest"]
    _require(isinstance(proposal_name, str) and Path(proposal_name).parts[:2] == ("data", "candidates"),
             "candidate manifest must be a repository data/candidates artifact")
    proposal_path = _contained_path(root, proposal_name)
    _require(_sha256(proposal_path) == source["candidate_manifest_sha256"], "staged candidate identity differs")
    proposal = _read_json(proposal_path)
    _require(proposal["source_manifest_sha256"] == preflight["manifest_sha256"]
             and proposal["candidate_id"] == source["candidate_id"], "candidate reference identity differs")
    records = {row["id"]: row for row in reference["records"]}
    _require(len(records) == len(reference["records"]) == 157
             and sum(row["split"] == "train" for row in records.values()) == 137
             and sum(row["split"] == "val" for row in records.values()) == 20,
             "paired training requires the complete 137/20 reference split")
    changes = {}
    for change in proposal["changes"]:
        identifier = change["image_id"]
        _require(identifier not in changes and identifier in records, "duplicate or unknown candidate image")
        record = records[identifier]
        _require(record["split"] == "train" and change["source_label"] == record["label"]
                 and change["source_label_sha256"] == record["label_sha256"]
                 and _digest(change["candidate_label_sha256"]), "candidate changes a different source or split")
        changes[identifier] = change
    variants = _mapping(staging["variants"], "staged variants")
    _require(set(variants) == {"reference", "candidate"}, "staging requires both experiment arms")
    for arm, variant in variants.items():
        expected = {}
        for identifier, record in records.items():
            expected[record["image"]] = record["image_sha256"]
            expected[record["label"]] = (changes[identifier]["candidate_label_sha256"]
                if arm == "candidate" and identifier in changes else record["label_sha256"])
        actual = {}
        for entry in variant["files"]:
            _require(entry["path"] not in actual and entry["source_sha256"] == entry["output_sha256"],
                     "staging has duplicate files or changed copied contents")
            actual[entry["path"]] = entry["output_sha256"]
        _require(actual == expected, f"staged {arm} file inventory differs from declared reference/proposal")
        counts = {split: {("objects" if key == "instances" else key): value for key, value in values.items()}
                  for split, values in reference["split_summary"].items()}
        if arm == "candidate":
            counts = proposal["expected_counts"]
        _require(_same(variant["counts"], counts), f"staged {arm} counts differ")
        _require(variant["data_yaml"] == f"{arm}/data.yaml" and _digest(variant["data_yaml_sha256"]),
                 "invalid staged dataset YAML identity")
    _require(_same(variants["reference"]["counts"]["val"], variants["candidate"]["counts"]["val"]),
             "staged validation counts differ between arms")
    return variants


def _check_results(path, epochs):
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        headers = [name.strip() for name in reader.fieldnames or []]
        required = {"epoch", "train/box_loss", "train/pose_loss", "train/kobj_loss",
                    "val/box_loss", "val/pose_loss", "val/kobj_loss"}
        _require(required <= set(headers) and len(headers) == len(set(headers)), "invalid completed results columns")
        observed = []
        for row in reader:
            _require(None not in row and all(value is not None for value in row.values()), "invalid results row")
            row = {key.strip(): value.strip() for key, value in row.items()}
            _require(re.fullmatch(r"[1-9][0-9]*", row["epoch"]) is not None, "invalid results epoch")
            observed.append(int(row["epoch"]))
            _require(all(math.isfinite(float(value)) for value in row.values()), "nonfinite completed results")
    _require(observed == epochs, "results.csv does not contain the complete predeclared epoch sequence")


def _check_receipt(preflight, manifest, base, receipt, staging):
    _require(type(receipt["schema_version"]) is int and receipt["schema_version"] == 1,
             "unsupported training receipt")
    _require(receipt["purpose"] == "paired-finetune" and receipt["status"] == "complete",
             "only a completed paired-finetune qualifies; resource trials are ineligible")
    _require(receipt["arm"] in ("reference", "candidate"), "invalid training arm")
    for key, expected in (("initial_checkpoint_sha256", preflight["model_sha256"]),
                          ("reference_manifest_sha256", preflight["manifest_sha256"]),
                          ("model_card_sha256", preflight["model_card_sha256"]),
                          ("staging_manifest_sha256", manifest["staging_manifest"]["sha256"])):
        _require(receipt[key] == expected, f"training receipt {key} differs")
    variants = _check_staging(preflight, staging)
    variant = variants[receipt["arm"]]
    _require(receipt["data_yaml_sha256"] == variant["data_yaml_sha256"], "training arm/YAML identity differs")
    settings = _mapping(receipt["training_settings"], "training settings")
    for key, value in RECIPE.items():
        _require(key in settings and _same(settings[key], value), f"training setting differs from paired recipe: {key}")
    _require(settings["data"] == str(Path(staging["output_root"]) / variant["data_yaml"]),
             "training data path differs from its staged arm")
    epochs = list(range(1, RECIPE["epochs"] + 1))
    _require(_same(receipt["completed_epochs"], epochs), "training did not complete every predeclared epoch")
    _require(type(receipt["optimizer_steps"]) is int and receipt["optimizer_steps"] > 0,
             "no optimizer updates recorded")
    _require(receipt["training_returned"] is True and receipt["finalization_completed"] is True
             and receipt["checkpoint_selection"] == "final_epoch", "training/finalization/selection is incomplete")
    execution = _mapping(receipt["execution"], "training execution")
    _require(type(execution["exit_code"]) is int and execution["exit_code"] == 0
             and execution["stop_reason"] is None and "input_verification_error" not in receipt,
             "training execution stopped, failed, or lost verified inputs")
    _require(receipt["optimizer_parameter_change_observed"] is True and receipt["validation_executed"] is True,
             "completed training lacks observed parameter updates or validation")
    _require(_same(receipt["checkpoint"], manifest["checkpoint"]), "receipt identifies a different final checkpoint")
    _require(receipt["checkpoint"]["sha256"] != preflight["model_sha256"], "final checkpoint is the unchanged initialization")
    source = _mapping(receipt["source"], "training source")
    _require(isinstance(source["revision"], str) and re.fullmatch(r"[0-9a-f]{40}", source["revision"]) is not None
             and source["working_tree_status"] == "", "completed comparison requires a clean recorded revision")
    hashes = _mapping(source["files_sha256"], "training code hashes")
    _require({"scripts/pose_checkpoint.py", "scripts/train_pose_comparison.py"} <= set(hashes)
             and all(_digest(value) for value in hashes.values()),
             "training code/recipe hashes are missing or invalid")
    runtime = _mapping(receipt["runtime"], "training runtime")
    _require(runtime["ultralytics"] == ULTRALYTICS_VERSION
             and all(isinstance(runtime[key], str) and runtime[key] for key in ("python", "torch", "device")),
             "training runtime is missing or incompatible")
    _check_results(_artifact(base, receipt["results"]), epochs)
    log = _artifact(base, receipt["log"])
    _require(log.stat().st_size > 0, "training log is empty")


def _validate_bundle(preflight, manifest, base, require_weights):
    try:
        _require("checkpoint_identity" not in preflight, "checkpoint selection requires an unmodified reference preflight")
        expected_fields = {"schema_version", "kind", *POSE_CONTRACT, "checkpoint", "training_receipt", "staging_manifest"}
        _require(set(manifest) == expected_fields and type(manifest["schema_version"]) is int
                 and manifest["schema_version"] == 1 and manifest["kind"] == "baomai-trained-checkpoint",
                 "unsupported checkpoint manifest")
        for key, expected in POSE_CONTRACT.items():
            _require(_same(manifest[key], expected), f"checkpoint {key} differs from baomai")
        weights = _artifact(base, manifest["checkpoint"], checkpoint=True, verify_bytes=require_weights)
        receipt = _read_json(_artifact(base, manifest["training_receipt"]))
        staging = _read_json(_artifact(base, manifest["staging_manifest"]))
        _check_receipt(preflight, manifest, base, receipt, staging)
        return weights, {"kind": manifest["kind"], "purpose": receipt["purpose"], "arm": receipt["arm"],
            "checkpoint_sha256": manifest["checkpoint"]["sha256"], "checkpoint_size_bytes": manifest["checkpoint"]["size_bytes"],
            "training_receipt_sha256": manifest["training_receipt"]["sha256"],
            "staging_manifest_sha256": manifest["staging_manifest"]["sha256"],
            "initial_checkpoint_sha256": receipt["initial_checkpoint_sha256"],
            "completed_epochs": len(receipt["completed_epochs"]), "selection": receipt["checkpoint_selection"]}
    except (KeyError, TypeError, AttributeError, IndexError) as error:
        raise ValueError(f"invalid checkpoint provenance structure: {error}") from error


def select_checkpoint(preflight, manifest_path, *, require_weights=True):
    """Select an explicitly verified bundle; reference validation identity stays fixed."""
    manifest_path = Path(manifest_path).resolve()
    manifest = _read_json(manifest_path)
    weights, identity = _validate_bundle(preflight, manifest, manifest_path.parent, require_weights)
    identity["manifest_sha256"] = _sha256(manifest_path)
    return {**preflight, "reference_model_sha256": preflight["model_sha256"],
            "model_sha256": manifest["checkpoint"]["sha256"], "weights": str(weights),
            "weights_verified": require_weights, "checkpoint_identity": identity}


def write_checkpoint_manifest(run_dir, preflight, receipt_path="training-receipt.json"):
    """Publish only a complete internally consistent bundle, without replacing files."""
    run_dir = Path(run_dir).resolve()
    receipt_file = _contained_path(run_dir, receipt_path)
    receipt = _read_json(receipt_file)
    _require(receipt.get("purpose") == "paired-finetune" and receipt.get("status") == "complete",
             "only a completed paired-finetune qualifies; resource trials are ineligible")
    stage_file = _contained_path(run_dir, "staging-manifest.json")
    manifest = {"schema_version": 1, "kind": "baomai-trained-checkpoint", **POSE_CONTRACT,
                "checkpoint": receipt["checkpoint"],
                "training_receipt": {"path": receipt_path, "sha256": _sha256(receipt_file)},
                "staging_manifest": {"path": "staging-manifest.json", "sha256": _sha256(stage_file)}}
    _validate_bundle(preflight, manifest, run_dir, True)
    output = run_dir / "checkpoint-manifest.json"
    with output.open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return output


def check_loaded_pose_model(model):
    """Check the actual head even if a prediction returns no detections."""
    try:
        shape = model.model.model[-1].kpt_shape
        compatible = (model.task == "pose" and model.names == {0: "cassette_front"}
                      and _same(list(shape), [2, 3]))
    except (AttributeError, IndexError, TypeError):
        compatible = False
    if not compatible:
        raise RuntimeError("loaded checkpoint task/classes/keypoint head differ from baomai [2, 3]")
