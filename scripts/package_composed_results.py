#!/usr/bin/env python3
"""Verify and package the fixed endpoint-control versus composed-fronts pair."""

import sys
sys.dont_write_bytecode = True

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import tempfile
import zipfile

from evaluate_pose import _relative_file, file_sha256, preflight_dataset
from package_gpu_results import (INDEX_NAME, check_pair, collect_arm, copy_member,
                                 shared_settings, verify_archive)
from pose_checkpoint import _same, recipe_identity
from restore_gpu_inputs import read_json, require
from verify_pose_composed_candidate import verify_composed_candidate


ROLES = ("reference", "candidate")
DEFAULT_PROTOCOL = "experiments/composed-fronts-2026-11-12.json"
CANDIDATE = "data/candidates/composed-fronts-2026-11-12/manifest.json"
CONTROL = "data/candidates/endpoint-visibility-2026-11-12/manifest.json"
CONTROL_SHA256 = "9c489bf57a526509de73e40cc73b02117398fba4ec97107bf047aa71e4951c41"
RECIPE_ID = "reviewed-composed-seed46-v1"
REFERENCE = "data/references/visible-front-adjudication-2026-11-12/manifest.json"
REFERENCE_SHA256 = "75e8689f65f7f32ca7f4657f27d214aa1ba371347812de780c6e5e6ac21af61e"
OPERATION_SPEC = "results/review/next-training-20261112/combined-operations-spec.json"
OPERATION_SHA256 = "1492c39b5a6b3b890ab6807fc51a1ed1524f1bf57844dd285341b6b2344852b0"
EVALUATION_SETTINGS = {"primary_box_confidence": 0.13, "diagnostic_box_confidences": [],
                       "imgsz": 1280, "nms_iou": 0.70, "matching_iou": 0.50,
                       "endpoint_confidence": 0.50, "radii_px": [5, 10]}
EXPECTED_DELTA = {"changed_train_images": ["new_000023", "new_000031", "new_000045"],
                  "box_replacements": 17, "deleted_objects": 1,
                  "deleted_source_rows": {"new_000023": [22]},
                  "new_endpoint_slot_changes": 0, "preserved_endpoint_slot_edits": 10}
DECISION = ("At box confidence 0.13, require an original-image-supported improvement in object completeness "
            "or reduction of duplicate/false detections against the matched endpoint control, with no higher "
            "miss count or additional supported lost objects. Preserve locatable coverage and existing 5px "
            "and 10px pair successes; empty scenes must have no added outputs. Inspect all gained/lost match "
            "and success identities and distinguish actual recovery from IoU or assignment crossings. "
            "Assess replacement of the exact operational PR35 separately using the same preservation "
            "criteria. Reused development views and uncertain labels do not establish unseen-scene "
            "reliability. No automatic promotion or threshold/seed/epoch/subset search.")


def validate_protocol(root, protocol_path=DEFAULT_PROTOCOL):
    root = Path(root).resolve()
    protocol_path = Path(protocol_path)
    if not protocol_path.is_absolute():
        protocol_path = _relative_file(root, str(protocol_path))
    document = read_json(protocol_path)
    require(type(document.get("schema_version")) is int and document["schema_version"] == 1
            and document.get("kind") == "baomai-composed-comparison-protocol"
            and document.get("experiment_id") == "composed-fronts-seed46-v1"
            and document.get("status") == "prospectively_declared", "wrong composed protocol identity")
    require(datetime.fromisoformat(document["declared_at_utc"]).utcoffset() is not None,
            "protocol declaration must have a timezone")
    candidate_path = _relative_file(root, CANDIDATE)
    receipt = verify_composed_candidate(root, candidate_path)
    proposal = read_json(candidate_path)
    control_path = _relative_file(root, CONTROL)
    require(file_sha256(control_path) == CONTROL_SHA256, "control manifest identity changed")
    control = read_json(control_path)
    require(_same(document.get("control"), {"manifest": CONTROL, "sha256": CONTROL_SHA256}),
            "protocol must use the current ten-slot endpoint control")
    require(_same(document.get("candidate"), {"manifest": CANDIDATE,
            "sha256": receipt["candidate_manifest_sha256"], "candidate_id": receipt["candidate_id"]}),
            "protocol composed candidate identity differs")
    require(document.get("source_manifest_sha256") == receipt["source_manifest_sha256"],
            "protocol original manifest differs")
    require(document.get("candidate_source_revision") == proposal["source_revision"],
            "protocol candidate source revision differs")
    recipe = recipe_identity(RECIPE_ID)
    require(document.get("recipe_sha256") == recipe["recipe_sha256"], "protocol recipe differs")
    expected_counts = {"reference": control["expected_counts"], "candidate": proposal["expected_counts"]}
    expected_roles = {arm: {"arm": arm, "supervision": "endpoint_control" if arm == "reference" else "composed_treatment",
                           "candidate_manifest": CANDIDATE,
                           "candidate_manifest_sha256": receipt["candidate_manifest_sha256"],
                           "candidate_id": receipt["candidate_id"], **recipe,
                           "counts": expected_counts[arm]} for arm in ROLES}
    require(_same(document.get("roles"), expected_roles), "protocol role, arm, supervision or counts differ")
    require(_same(document.get("incremental_delta"), EXPECTED_DELTA), "protocol incremental delta differs")
    require(_same(document.get("reviewed_operation_spec"), {"path": OPERATION_SPEC, "sha256": OPERATION_SHA256})
            and file_sha256(_relative_file(root, OPERATION_SPEC)) == OPERATION_SHA256,
            "protocol operation evidence differs")
    original = read_json(_relative_file(root, "data/manifest.json"))
    require(document.get("initial_checkpoint_sha256") == original["model_sha256"],
            "protocol initialization differs")
    execution = document.get("execution", {})
    expected_execution = {"seed": 46, "epochs": 30, "independent_epoch173_initialization": True,
                          "resume": False, "checkpoint_selection": "final_epoch",
                          "checkpoint_path": "fit/weights/last.pt",
                          "resource_trial": {"role": "candidate", "batches": 4, "initializes_full_runs": False},
                          "require_same_clean_source": True, "require_same_recorded_gpu_runtime": True,
                          "require_same_resolved_numerical_settings": True,
                          "reject_incompatible_restores": True, "automatic_promotion": False}
    require(_same(execution, expected_execution), "protocol execution design differs")
    evaluation = document.get("evaluation", {})
    require(all(_same(evaluation.get(k), v) for k, v in EVALUATION_SETTINGS.items()),
            "protocol evaluation settings differ")
    require(evaluation.get("primary_reference") == REFERENCE
            and evaluation.get("primary_reference_sha256") == REFERENCE_SHA256,
            "protocol must freeze the adjudicated development reference")
    reference_path = _relative_file(root, REFERENCE)
    require(file_sha256(reference_path) == REFERENCE_SHA256, "evaluation reference manifest changed")
    reference = read_json(reference_path)
    for item in reference["files"]:
        require(file_sha256(_relative_file(reference_path.parent, item["label_path"])) == item["label_sha256"],
                "evaluation reference label changed")
    require(evaluation.get("all_object_denominator") == reference["counts"]["objects"] == 663
            and evaluation.get("locatable_endpoint_denominator") == reference["counts"]["visible_endpoints"] == 1227
            and evaluation.get("both_corner_denominator") == reference["objects_by_locatable_corners"]["2"] == 584,
            "protocol evaluation denominator differs")
    require(evaluation.get("training_validation_unchanged") is True
            and evaluation.get("references_reported_separately") is True,
            "development reference must remain separate from training validation")
    pr35 = evaluation.get("PR35", {})
    require(pr35.get("checkpoint_sha256") == "420c00f73eeba24e587342400878b7d96371d3e745772113809aa705ec8163b1",
            "operational PR35 comparator differs")
    require(evaluation.get("decision") == DECISION, "protocol promotion decision rule differs")
    inventories = {}
    for arm, overlay in (("reference", control), ("candidate", proposal)):
        changes = {change["image_id"]: change for change in overlay["changes"]}
        inventory = {}
        for record in original["records"]:
            inventory[record["image"]] = record["image_sha256"]
            change = changes.get(record["id"])
            inventory[record["label"]] = change["candidate_label_sha256"] if change else record["label_sha256"]
        inventories[arm] = inventory
    changed = sorted(p for p in inventories["reference"]
                     if inventories["reference"][p] != inventories["candidate"][p])
    require(changed == [f"labels/train/{name}.txt" for name in EXPECTED_DELTA["changed_train_images"]],
            "effective control/treatment file delta differs")
    return {"document": document, "sha256": file_sha256(protocol_path), "path": protocol_path.resolve(),
            "candidate_receipt": receipt, "inventories": inventories,
            "delta_train_image_ids": EXPECTED_DELTA["changed_train_images"]}


def collect_composed_run(preflight, run, role, context):
    require(role in ROLES, "unknown composed role")
    bundle = collect_arm(preflight, Path(run).resolve(), role)
    receipt, staging = bundle["receipt"], bundle["staging"]
    spec = context["document"]["roles"][role]
    require(all(receipt.get(k) == spec[k] for k in ("recipe_id", "recipe_sha256")),
            "completed run recipe differs from the protocol")
    require(receipt["source"].get("working_tree_status") == "", "completed source must be clean")
    require(receipt["runtime"]["device"].startswith("cuda:"), "composed comparison requires CUDA runs")
    require(receipt["initial_checkpoint_sha256"] == context["document"]["initial_checkpoint_sha256"],
            "completed initialization differs from the protocol")
    require(all(staging["source"].get(k) == spec[k] for k in
                ("candidate_manifest", "candidate_manifest_sha256", "candidate_id")),
            "completed proposal identity differs from the protocol")
    control = staging["source"].get("control", {})
    require(control.get("manifest") == CONTROL and control.get("manifest_sha256") == CONTROL_SHA256,
            "completed control identity differs")
    for arm in ROLES:
        variant = staging["variants"][arm]
        require(_same(variant["counts"], context["document"]["roles"][arm]["counts"]),
                "completed staging counts differ")
        rows = variant["files"]
        require(len(rows) == len(context["inventories"][arm])
                and len({row["path"] for row in rows}) == len(rows), "completed file inventory differs")
        require({row["path"]: row["output_sha256"] for row in rows} == context["inventories"][arm]
                and all(row["source_sha256"] == row["output_sha256"] for row in rows),
                "completed supervision or image hashes differ")
    return bundle


def check_composed_pair(bundles, context):
    require(set(bundles) == set(ROLES), "return both composed roles")
    check_pair(bundles["reference"], bundles["candidate"])
    require(all(bundles[role]["receipt"]["arm"] == role for role in ROLES),
            "composed roles must retain their actual runner arms")
    require(all(bundles[role]["receipt"]["recipe_sha256"] == context["document"]["recipe_sha256"]
                for role in ROLES), "paired numerical recipe differs")


def package_results(root, reference, candidate, output, *, protocol=DEFAULT_PROTOCOL, verify_existing=False):
    root, output = Path(root).resolve(), Path(output).absolute()
    runs = {"reference": Path(reference), "candidate": Path(candidate)}
    require(all(path.is_dir() and not path.is_symlink() for path in runs.values()),
            "both completed run paths must be regular directories")
    runs = {arm: path.resolve() for arm, path in runs.items()}
    require(not runs["reference"].is_relative_to(runs["candidate"])
            and not runs["candidate"].is_relative_to(runs["reference"]), "run directories overlap")
    require(not output.is_symlink() and output.suffix.lower() == ".zip", "choose a nonsymlink ZIP output")
    output = output.resolve()
    require(not any(output.is_relative_to(path) for path in (root, *runs.values())),
            "output must be outside source and runs")
    require(output.is_file() if verify_existing else not output.exists(),
            "verification requires an existing ZIP; packaging requires a fresh output")
    context = validate_protocol(root, protocol)
    preflight = preflight_dataset(root, require_weights=False)
    bundles = {arm: collect_composed_run(preflight, path, arm, context) for arm, path in runs.items()}
    check_composed_pair(bundles, context)
    rows = [row for bundle in bundles.values() for row in bundle["files"]]
    rows.append({"path": "protocol.json", "sha256": context["sha256"],
                 "size_bytes": context["path"].stat().st_size, "source_path": context["path"]})
    rows.sort(key=lambda row: row["path"])
    created = datetime.now(timezone.utc).isoformat()
    if verify_existing:
        with zipfile.ZipFile(output) as archive:
            created = json.loads(archive.read(INDEX_NAME))["created_at_utc"]
        require(datetime.fromisoformat(created).utcoffset() is not None, "invalid archive creation time")
    index = {"schema_version": 1, "kind": "baomai-composed-training-results",
             "created_at_utc": created, "experiment_id": context["document"]["experiment_id"],
             "protocol": {"path": "protocol.json", "sha256": context["sha256"]},
             "source": bundles["reference"]["staging"]["source"],
             "training_source": bundles["reference"]["receipt"]["source"],
             "runtime": bundles["reference"]["receipt"]["runtime"],
             "shared_training_settings": shared_settings(bundles["reference"]["receipt"]),
             "arms": {arm: {"checkpoint_manifest": f"{arm}/checkpoint-manifest.json",
                            "checkpoint_identity": bundle["identity"]} for arm, bundle in bundles.items()},
             "files": [{k: v for k, v in row.items() if k != "source_path"} for row in rows],
             "verification_scope": "Completed matched-run artifacts and frozen protocol; no model loading, resource-trial proof or accuracy approval."}
    index_bytes = (json.dumps(index, indent=2, allow_nan=False) + "\n").encode("utf-8")
    if not verify_existing:
        output.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".composed-results-", dir=output.parent) as temporary:
            archive_path = Path(temporary) / "results.zip"
            with zipfile.ZipFile(archive_path, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=1) as archive:
                archive.writestr(INDEX_NAME, index_bytes)
                for row in rows:
                    copy_member(archive, row)
            verify_archive(archive_path, index_bytes, rows)
            os.link(archive_path, output)
    verify_archive(output, index_bytes, rows)
    return {"status": "verified" if verify_existing else "packaged", "archive": str(output),
            "sha256": file_sha256(output), "size_bytes": output.stat().st_size,
            "files": len(rows) + 1, "protocol_sha256": context["sha256"],
            "model_accuracy_evaluated": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--protocol", default=DEFAULT_PROTOCOL)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--verify-existing", action="store_true")
    args = parser.parse_args()
    try:
        result = package_results(args.root, args.reference, args.candidate, args.output,
                                 protocol=args.protocol, verify_existing=args.verify_existing)
    except (ValueError, OSError, KeyError, TypeError, zipfile.BadZipFile) as error:
        parser.exit(1, f"Composed return package refused: {error}\n")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
