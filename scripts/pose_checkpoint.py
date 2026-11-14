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
ORIGINAL_RECIPE_ID = "paired-finetune-v1"
WARMUP_RECIPE_ID = "warmup-bias-lr-1e-4-v1"
BOX_RECIPE_ID = "box-supervision-warmup-v1"
BOX_REPEATABILITY_RECIPES = {
    43: "box-supervision-warmup-seed43-v1",
    44: "box-supervision-warmup-seed44-v1",
}
BOX_RECIPE_IDS = (BOX_RECIPE_ID, *BOX_REPEATABILITY_RECIPES.values())
ANNOTATION_RECIPE_ID = "reviewed-annotations-seed45-v1"
ANNOTATION_CONFIRMATION_RECIPE_ID = "reviewed-annotations-seed46-v1"
ANNOTATION_RECIPE_IDS = (ANNOTATION_RECIPE_ID, ANNOTATION_CONFIRMATION_RECIPE_ID)
ENDPOINT_RECIPE_ID = "reviewed-endpoints-seed46-v1"
COMPOSED_RECIPE_ID = "reviewed-composed-seed46-v1"
OBSERVED_RECIPE_ID = "reviewed-observed-fronts-seed46-v1"
RECIPE_IDS = (ORIGINAL_RECIPE_ID, WARMUP_RECIPE_ID, *BOX_RECIPE_IDS, *ANNOTATION_RECIPE_IDS,
              ENDPOINT_RECIPE_ID, COMPOSED_RECIPE_ID, OBSERVED_RECIPE_ID)

# Resolved defaults observed in the pinned trainer, in addition to RECIPE.
# These are checked, not changed, by the runner. In particular RLE stays at 1.0.
# Schema 1 keeps its original validation rules; schema 2 declares every setting.
PINNED_TRAINER_DEFAULTS = {
    "task": "pose", "mode": "train", "time": None, "workers": 0,
    "name": "fit", "exist_ok": False, "cls_remap": True, "verbose": False,
    "profile": False, "channels_last": None, "overlap_mask": True,
    "mask_ratio": 4, "dropout": 0.0, "save_json": False, "conf": None,
    "iou": 0.7, "max_det": 300, "quantize": None, "dnn": False,
    "plots": False, "nms": None, "source": None, "vid_stride": 1,
    "stream_buffer": False, "visualize": False, "augment": False,
    "agnostic_nms": False, "classes": None, "retina_masks": False,
    "embed": None, "show": False, "save_frames": False, "save_txt": False,
    "save_conf": False, "save_crop": False, "show_labels": True,
    "show_conf": True, "show_boxes": True, "line_width": None,
    "format": "torchscript", "optimize": False, "dynamic": False,
    "simplify": True, "opset": None, "workspace": None,
    "distill_model": None, "dis": 6.0, "cls_pw": 0.0, "rle": 1.0,
    "angle": 1.0, "dlog": 1.0, "dgrad": 0.5, "dlam": 1.0,
    "bgr": 0.0, "copy_paste_mode": "flip", "auto_augment": "randaugment",
    "erasing": 0.4, "cfg": None, "tracker": "tracktrack.yaml",
}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _same(actual, expected):
    return json.dumps(actual, sort_keys=True, allow_nan=False) == json.dumps(
        expected, sort_keys=True, allow_nan=False)


def get_recipe(recipe_id=ORIGINAL_RECIPE_ID):
    """Return an independent fixed recipe; the original RECIPE remains unchanged."""
    _require(recipe_id in RECIPE_IDS, f"unsupported training recipe: {recipe_id}")
    recipe = {**RECIPE, **({"warmup_bias_lr": 0.0001}
                         if recipe_id in (WARMUP_RECIPE_ID, *BOX_RECIPE_IDS, *ANNOTATION_RECIPE_IDS,
                                          ENDPOINT_RECIPE_ID, COMPOSED_RECIPE_ID, OBSERVED_RECIPE_ID) else {})}
    for seed, fixed_id in BOX_REPEATABILITY_RECIPES.items():
        if recipe_id == fixed_id:
            recipe["seed"] = seed
    if recipe_id == ANNOTATION_RECIPE_ID:
        recipe["seed"] = 45
    elif recipe_id in (ANNOTATION_CONFIRMATION_RECIPE_ID, ENDPOINT_RECIPE_ID, COMPOSED_RECIPE_ID,
                       OBSERVED_RECIPE_ID):
        recipe["seed"] = 46
    return recipe


def check_recipe_candidate_kind(recipe_id, kind):
    """Bind each new experiment to its intended overlay; retain legacy contracts."""
    _require(recipe_id in RECIPE_IDS, f"unsupported training recipe: {recipe_id}")
    _require(kind in ("training_label_overlay", "training_box_overlay", "training_annotation_overlay",
                     "training_endpoint_overlay", "training_composed_overlay", "training_observed_review_composition"),
             f"unsupported training candidate kind: {kind}")
    _require((recipe_id == ENDPOINT_RECIPE_ID) == (kind == "training_endpoint_overlay"),
             "reviewed-endpoints recipe and training_endpoint_overlay must be selected together")
    _require((recipe_id == COMPOSED_RECIPE_ID) == (kind == "training_composed_overlay"),
             "reviewed-composed recipe and training_composed_overlay must be selected together")
    _require((recipe_id == OBSERVED_RECIPE_ID) == (kind == "training_observed_review_composition"),
             "observed-fronts recipe and observed review composition must be selected together")
    _require((recipe_id in BOX_RECIPE_IDS) == (kind == "training_box_overlay"),
             "box-supervision recipe and training_box_overlay must be selected together")
    _require((recipe_id in ANNOTATION_RECIPE_IDS) == (kind == "training_annotation_overlay"),
             "reviewed-annotations recipe and training_annotation_overlay must be selected together")


def recipe_identity(recipe_id=ORIGINAL_RECIPE_ID):
    """Hash the full declared fixed settings, excluding execution paths/device.

    The exact payload is {"ultralytics_version": ULTRALYTICS_VERSION,
    "recipe": get_recipe(recipe_id), "resolved_defaults": PINNED_TRAINER_DEFAULTS}.
    SHA-256 covers its UTF-8 JSON with sort_keys=True, separators=(",", ":"),
    allow_nan=False, and no trailing newline. The recipe ID is not in the payload.
    """
    payload = {"ultralytics_version": ULTRALYTICS_VERSION, "recipe": get_recipe(recipe_id),
               "resolved_defaults": PINNED_TRAINER_DEFAULTS}
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"),
                                      allow_nan=False).encode("utf-8")).hexdigest()
    return {"recipe_id": recipe_id, "recipe_sha256": digest}


def check_training_settings(settings, recipe_id, *, model, data, device, project):
    """Check every resolved trainer argument for schema-2 runs, before training."""
    settings = _mapping(settings, "training settings")
    expected = {**PINNED_TRAINER_DEFAULTS, **get_recipe(recipe_id),
                "model": str(model), "data": str(data), "device": device,
                "project": str(project), "save_dir": str(Path(project) / "fit")}
    _require(set(settings) == set(expected),
             "training settings inventory differs from declared recipe: "
             f"missing={sorted(set(expected) - set(settings))}, extra={sorted(set(settings) - set(expected))}")
    for key, value in expected.items():
        _require(_same(settings[key], value), f"training setting differs from declared recipe: {key}")


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
    control, control_path, control_changes = None, None, {}
    if proposal.get("kind") == "training_composed_overlay":
        from verify_pose_composed_candidate import verify_composed_candidate
        verified = verify_composed_candidate(root, proposal_path)
        control = verified["control"]
        _require(source.get("candidate_kind") == "training_composed_overlay"
                 and _same(source.get("control"), control), "staged endpoint control identity/counts differ")
        _require(verified["candidate_manifest_sha256"] == source["candidate_manifest_sha256"]
                 and _same(verified["counts"], proposal["expected_counts"]), "verified composed proposal differs")
        control_path = _contained_path(root, control["manifest"])
        _require(_sha256(control_path) == control["manifest_sha256"], "staged control manifest identity differs")
        control_proposal = _read_json(control_path)
        _require(control_proposal.get("kind") == "training_endpoint_overlay"
                 and control_proposal.get("candidate_id") == control["candidate_id"], "staged control is not the endpoint overlay")
        control_changes = {change["image_id"]: change for change in control_proposal["changes"]}
        _require(len(control_changes) == control["changed_train_labels"], "staged control changed-label count differs")
    elif proposal.get("kind") == "training_observed_review_composition":
        from verify_observed_training_candidate import verify_observed_training_candidate
        verified = verify_observed_training_candidate(root, proposal_path)
        control = verified["control"]
        _require(source.get("candidate_kind") == proposal["kind"]
                 and _same(source.get("control"), control)
                 and source.get("training_protocol") == verified["training_protocol"],
                 "staged PR35 control or observed training protocol differs")
        control_path = _contained_path(root, control["manifest"])
        control_proposal = _read_json(control_path)
        control_changes = {change["image_id"]: change for change in control_proposal["changes"]}
    variants = _mapping(staging["variants"], "staged variants")
    _require(set(variants) == {"reference", "candidate"}, "staging requires both experiment arms")
    for arm, variant in variants.items():
        expected, expected_sources = {}, {}
        arm_changes, overlay_path = (changes, proposal_path) if arm == "candidate" else (control_changes, control_path)
        for identifier, record in records.items():
            expected[record["image"]] = record["image_sha256"]
            change = arm_changes.get(identifier)
            expected[record["label"]] = change["candidate_label_sha256"] if change else record["label_sha256"]
            if control is not None:
                expected_sources[record["image"]] = f"data/{record['image']}"
                expected_sources[record["label"]] = (
                    _contained_path(overlay_path.parent, change["candidate_label"]).relative_to(root).as_posix()
                    if change else f"data/{record['label']}")
        actual = {}
        for entry in variant["files"]:
            _require(entry["path"] not in actual and entry["source_sha256"] == entry["output_sha256"],
                     "staging has duplicate files or changed copied contents")
            actual[entry["path"]] = entry["output_sha256"]
            if control is not None:
                _require(entry.get("source_path") == expected_sources.get(entry["path"]),
                         f"staged {arm} source path differs from composed/control overlay")
        _require(actual == expected, f"staged {arm} file inventory differs from declared reference/proposal")
        counts = {split: {("objects" if key == "instances" else key): value for key, value in values.items()}
                  for split, values in reference["split_summary"].items()}
        if arm == "candidate":
            counts = proposal["expected_counts"]
        elif control is not None:
            counts = control["counts"]
        _require(_same(variant["counts"], counts), f"staged {arm} counts differ")
        _require(variant["data_yaml"] == f"{arm}/data.yaml" and _digest(variant["data_yaml_sha256"]),
                 "invalid staged dataset YAML identity")
    _require(_same(variants["reference"]["counts"]["val"], variants["candidate"]["counts"]["val"]),
             "staged validation counts differ between arms")
    return variants


def training_label_identity(staging, arm):
    """Name the experiment's actual control/treatment labels explicitly."""
    source = _mapping(staging["source"], "staging source")
    if source.get("candidate_kind") not in ("training_composed_overlay", "training_observed_review_composition"):
        return None
    _require(arm in ("reference", "candidate"), "invalid controlled training arm")
    control = _mapping(source["control"], "staged control")
    reference = arm == "reference"
    if source["candidate_kind"] == "training_observed_review_composition":
        return {
            "kind": source["candidate_kind"],
            "role": "pr35_annotation_control" if reference else "observed_fronts_treatment",
            "label_manifest": control["manifest"] if reference else source["candidate_manifest"],
            "label_manifest_sha256": control["manifest_sha256"] if reference else source["candidate_manifest_sha256"],
            "label_candidate_id": control["candidate_id"] if reference else source["candidate_id"],
            "control_manifest": control["manifest"], "control_manifest_sha256": control["manifest_sha256"],
            "observed_manifest": source["candidate_manifest"],
            "observed_manifest_sha256": source["candidate_manifest_sha256"],
            "training_protocol": source["training_protocol"],
        }
    return {
        "kind": "training_composed_overlay",
        "role": "endpoint_control" if reference else "composed_treatment",
        "label_manifest": control["manifest"] if reference else source["candidate_manifest"],
        "label_manifest_sha256": control["manifest_sha256"] if reference else source["candidate_manifest_sha256"],
        "label_candidate_id": control["candidate_id"] if reference else source["candidate_id"],
        "control_manifest": control["manifest"], "control_manifest_sha256": control["manifest_sha256"],
        "composed_manifest": source["candidate_manifest"],
        "composed_manifest_sha256": source["candidate_manifest_sha256"],
    }


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


def _check_runtime_v2(runtime, settings):
    fields = {"python", "torch", "ultralytics", "device", "torch_cuda", "threads", "interop_threads",
              "platform", "packages", "device_name", "device_capability", "backend"}
    _require(set(runtime) == fields, "training runtime inventory differs from schema 2")
    _require(type(runtime["threads"]) is int and runtime["threads"] > 0
             and type(runtime["interop_threads"]) is int and runtime["interop_threads"] == 1
             and isinstance(runtime["platform"], str) and runtime["platform"]
             and (runtime["torch_cuda"] is None or isinstance(runtime["torch_cuda"], str)),
             "training runtime settings are incomplete")
    packages = _mapping(runtime["packages"], "training runtime packages")
    _require(set(packages) == {"torchvision", "numpy", "Pillow", "opencv-python", "psutil"}
             and all(isinstance(value, str) and value for value in packages.values()),
             "training runtime package versions are incomplete")
    device = settings["device"]
    _require(isinstance(device, str) and (device == "cpu" or device.isdecimal()),
             "training device must be cpu or one CUDA index")
    if device == "cpu":
        _require(runtime["device"] == "cpu" and runtime["device_name"] == "cpu"
                 and runtime["device_capability"] is None, "CPU device identity differs")
    else:
        capability = runtime["device_capability"]
        _require(re.fullmatch(r"cuda:\d+", runtime["device"]) is not None
                 and isinstance(runtime["device_name"], str) and runtime["device_name"]
                 and runtime["device_name"] != "cpu" and isinstance(runtime["torch_cuda"], str)
                 and bool(runtime["torch_cuda"])
                 and isinstance(capability, list) and len(capability) == 2
                 and all(type(value) is int and value >= 0 for value in capability) and capability[0] > 0,
                 "resolved CUDA device name/capability is missing or invalid")
    backend = _mapping(runtime["backend"], "training backend")
    boolean_fields = {"model_channels_last", "cuda_matmul_allow_tf32", "cudnn_allow_tf32",
                      "deterministic_algorithms_enabled", "deterministic_warn_only",
                      "cudnn_deterministic", "cudnn_benchmark"}
    _require(set(backend) == boolean_fields | {"float32_matmul_precision", "cudnn_version"}
             and all(type(backend[key]) is bool for key in boolean_fields)
             and backend["float32_matmul_precision"] in ("highest", "high", "medium")
             and (backend["cudnn_version"] is None or
                  type(backend["cudnn_version"]) is int and backend["cudnn_version"] > 0),
             "observed training backend state is missing or invalid")


def _check_receipt(preflight, manifest, base, receipt, staging):
    version = receipt["schema_version"]
    _require(type(version) is int and version in (1, 2) and version == manifest["schema_version"],
             "unsupported training receipt or manifest/receipt schema mismatch")
    identity = {}
    if version == 1:
        _require(not ({"recipe_id", "recipe_sha256"} & set(receipt)),
                 "legacy training receipt cannot declare a versioned recipe")
        recipe = RECIPE
    else:
        identity = recipe_identity(receipt["recipe_id"])
        _require(all(receipt[key] == value and manifest[key] == value for key, value in identity.items()),
                 "training recipe identity differs between declaration, receipt, or manifest")
        recipe = get_recipe(receipt["recipe_id"])
    _require(receipt["purpose"] == "paired-finetune" and receipt["status"] == "complete",
             "only a completed paired-finetune qualifies; resource trials are ineligible")
    _require(receipt["arm"] in ("reference", "candidate"), "invalid training arm")
    _require(not identity or identity["recipe_id"] != WARMUP_RECIPE_ID or receipt["arm"] == "reference",
             "warmup experiment requires the reference label arm")
    _require(not identity or identity["recipe_id"] != ENDPOINT_RECIPE_ID or receipt["arm"] == "candidate",
             "endpoint experiment requires the candidate label arm")
    for key, expected in (("initial_checkpoint_sha256", preflight["model_sha256"]),
                          ("reference_manifest_sha256", preflight["manifest_sha256"]),
                          ("model_card_sha256", preflight["model_card_sha256"]),
                          ("staging_manifest_sha256", manifest["staging_manifest"]["sha256"])):
        _require(receipt[key] == expected, f"training receipt {key} differs")
    variants = _check_staging(preflight, staging)
    proposal = _read_json(_contained_path(Path(preflight["root"]), staging["source"]["candidate_manifest"]))
    check_recipe_candidate_kind(identity.get("recipe_id", ORIGINAL_RECIPE_ID), proposal.get("kind"))
    labels = None
    if proposal.get("kind") in ("training_composed_overlay", "training_observed_review_composition"):
        labels = training_label_identity(staging, receipt["arm"])
        _require(labels is not None and _same(receipt.get("training_labels"), labels),
                 "completed checkpoint control/treatment label identity differs")
    variant = variants[receipt["arm"]]
    _require(receipt["data_yaml_sha256"] == variant["data_yaml_sha256"], "training arm/YAML identity differs")
    settings = _mapping(receipt["training_settings"], "training settings")
    for key, value in recipe.items():
        _require(key in settings and _same(settings[key], value), f"training setting differs from paired recipe: {key}")
    _require(settings["data"] == str(Path(staging["output_root"]) / variant["data_yaml"]),
             "training data path differs from its staged arm")
    if version == 2:
        stage_root = Path(staging["output_root"])
        _require(stage_root.is_absolute() and ".." not in stage_root.parts and stage_root.name == "staged-data",
                 "training staging root differs from the declared run structure")
        _require(isinstance(settings["model"], str), "training model path must be a string")
        model_path = Path(settings["model"])
        relative_model = Path(preflight["weights"]).relative_to(preflight["root"])
        _require(model_path.is_absolute() and ".." not in model_path.parts
                 and model_path.parts[-len(relative_model.parts):] == relative_model.parts,
                 "training model path differs from the frozen initialization")
        check_training_settings(settings, identity["recipe_id"], model=model_path,
                                data=stage_root / variant["data_yaml"], device=settings["device"],
                                project=stage_root.parent)
        _require(receipt["inputs_reverified_after_worker"] is True, "training inputs were not reverified")
    epochs = list(range(1, recipe["epochs"] + 1))
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
    if labels is not None:
        verifier = ("scripts/verify_observed_training_candidate.py"
                    if labels["kind"] == "training_observed_review_composition"
                    else "scripts/verify_pose_composed_candidate.py")
        _require({"scripts/stage_pose_training.py", verifier} <= set(hashes),
                 "controlled training source lacks stager/verifier identities")
    runtime = _mapping(receipt["runtime"], "training runtime")
    _require(runtime["ultralytics"] == ULTRALYTICS_VERSION
             and all(isinstance(runtime[key], str) and runtime[key] for key in ("python", "torch", "device")),
             "training runtime is missing or incompatible")
    if version == 2:
        _check_runtime_v2(runtime, settings)
    _check_results(_artifact(base, receipt["results"]), epochs)
    log = _artifact(base, receipt["log"])
    _require(log.stat().st_size > 0, "training log is empty")
    return {**identity, **({"training_labels": labels} if labels is not None else {})}


def _validate_bundle(preflight, manifest, base, require_weights):
    try:
        _require("checkpoint_identity" not in preflight, "checkpoint selection requires an unmodified reference preflight")
        version = manifest["schema_version"]
        expected_fields = {"schema_version", "kind", *POSE_CONTRACT, "checkpoint", "training_receipt", "staging_manifest"}
        if type(version) is int and version == 2:
            expected_fields |= {"recipe_id", "recipe_sha256"}
        _require(set(manifest) == expected_fields and type(manifest["schema_version"]) is int
                 and version in (1, 2) and manifest["kind"] == "baomai-trained-checkpoint",
                 "unsupported checkpoint manifest")
        for key, expected in POSE_CONTRACT.items():
            _require(_same(manifest[key], expected), f"checkpoint {key} differs from baomai")
        weights = _artifact(base, manifest["checkpoint"], checkpoint=True, verify_bytes=require_weights)
        receipt = _read_json(_artifact(base, manifest["training_receipt"]))
        staging = _read_json(_artifact(base, manifest["staging_manifest"]))
        recipe = _check_receipt(preflight, manifest, base, receipt, staging)
        return weights, {"kind": manifest["kind"], "purpose": receipt["purpose"], "arm": receipt["arm"],
            "checkpoint_sha256": manifest["checkpoint"]["sha256"], "checkpoint_size_bytes": manifest["checkpoint"]["size_bytes"],
            "training_receipt_sha256": manifest["training_receipt"]["sha256"],
            "staging_manifest_sha256": manifest["staging_manifest"]["sha256"],
            "initial_checkpoint_sha256": receipt["initial_checkpoint_sha256"],
            "completed_epochs": len(receipt["completed_epochs"]), "selection": receipt["checkpoint_selection"], **recipe}
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
    version = receipt.get("schema_version")
    _require(type(version) is int and version in (1, 2), "unsupported training receipt")
    recipe = ({key: receipt.get(key) for key in ("recipe_id", "recipe_sha256")} if version == 2 else {})
    manifest = {"schema_version": version, "kind": "baomai-trained-checkpoint", **POSE_CONTRACT, **recipe,
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
