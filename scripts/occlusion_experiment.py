#!/usr/bin/env python3
"""Bind the fixed pixel-corruption experiment, PR35 labels and completed GPU pair."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import zipfile

sys.dont_write_bytecode = True

from evaluate_pose import _relative_file, file_sha256, parse_pose_labels
from package_gpu_results import INDEX_NAME, check_pair, collect_arm, copy_member, shared_settings, verify_archive
from pose_checkpoint import _same, recipe_identity
from predict_pose_folder import BASELINE_SHA256, _reference_model
from restore_gpu_inputs import read_json, require
from stage_pose_training import MANIFEST, check_png, checked_output, manifest_fields, verify_outputs, yaml_bytes
from verify_pose_candidate import check_geometry, read_reference, totals
from verify_pose_composed_candidate import PARENT, policy_compatibility, verify_historical_control


DEFAULT_PROTOCOL = "experiments/occlusion-training-2026-11-13.json"
PROTOCOL_SHA256 = "c70563034d3ac969ff7837a594d53bc6316fda3897e1557b5d7828be52e94a62"
ROLES = ("control", "candidate")


def digest_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def load_protocol(root, path=DEFAULT_PROTOCOL):
    from pose_input_corruption import DEFAULT_SPEC
    root = Path(root).resolve()
    path = Path(path)
    path = path.resolve() if path.is_absolute() else _relative_file(root, str(path))
    require(path.is_relative_to(root), "protocol must be inside the source checkout")
    require(file_sha256(path) == PROTOCOL_SHA256, "fixed occlusion protocol differs")
    document = read_json(path)
    require(document["kind"] == "baomai-occlusion-training-protocol"
            and document["experiment_id"] == "occlusion-seed46-v1", "wrong experiment")
    require(_same(document["augmentation"], DEFAULT_SPEC), "corruption implementation/specification differ")
    identity = recipe_identity(document["recipe_id"])
    require(all(document[k] == v for k, v in identity.items()), "base numerical recipe differs")
    require(document["candidate_manifest"] == PARENT["manifest"]
            and document["candidate_manifest_sha256"] == PARENT["sha256"], "exact PR35 labels required")
    for name, expected in ((document["candidate_manifest"], document["candidate_manifest_sha256"]),
                           ("data/manifest.json", document["source_manifest_sha256"]),
                           (document["current_policy"]["path"], document["current_policy"]["sha256"]),
                           (document["historical_policy"]["path"], document["historical_policy"]["sha256"])):
        require(file_sha256(_relative_file(root, name)) == expected, "fixed source identity differs: " + name)
    reference_spec = document["evaluation"]["primary_reference"]
    reference_path = _relative_file(root, reference_spec["path"])
    require(file_sha256(reference_path) == reference_spec["sha256"], "development reference differs")
    for entry in read_json(reference_path)["files"]:
        require(file_sha256(_relative_file(reference_path.parent, entry["label_path"])) == entry["label_sha256"],
                "development reference label changed")
    probe = document["evaluation"]["synthetic_probe"]
    for key in ("protocol", "archive"):
        require(file_sha256(_relative_file(root, probe[key + "_path"])) == probe[key + "_sha256"],
                "PR60 diagnostic identity differs")
    with zipfile.ZipFile(root / probe["archive_path"]) as archive:
        require(hashlib.sha256(archive.read(probe["cases_member"])).hexdigest() == probe["cases_sha256"],
                "PR60 cases changed")
    for name, expected in read_json(root / probe["protocol_path"])["source_files"].items():
        require(file_sha256(_relative_file(root, name)) == expected, "historical replay helper changed: " + name)
    require(document["initial_checkpoint_sha256"] == BASELINE_SHA256, "epoch173 initialization differs")
    return {"document": document, "sha256": PROTOCOL_SHA256, "path": path}


def source_identity(root):
    from train_pose_comparison import source_identity as base_source
    root = Path(root).resolve()
    result = base_source(root)
    for name in ("scripts/occlusion_experiment.py", "scripts/train_pose_occlusion.py",
                 "scripts/pose_input_corruption.py", DEFAULT_PROTOCOL):
        result["files_sha256"][name] = file_sha256(_relative_file(root, name))
    return result


def experiment_identity(context, role):
    require(role in ROLES, "unknown experimental role")
    document = context["document"]
    return {"id": document["experiment_id"], "role": role, "protocol_sha256": context["sha256"],
            "base_recipe_id": document["recipe_id"], "base_recipe_sha256": document["recipe_sha256"],
            "label_arm": "candidate", "label_manifest": document["candidate_manifest"],
            "label_manifest_sha256": document["candidate_manifest_sha256"],
            "corruption_spec_sha256": digest_json(document["augmentation"]),
            "target_contract_sha256": digest_json(document["target_contract"])}


def verified_label_inputs(root, context=None):
    root = Path(root).resolve()
    context = context or load_protocol(root)
    document = context["document"]
    values = read_reference(root, document["source_manifest_sha256"])
    _, records, _, source_bytes, original_objects, original_counts = values
    policy_compatibility(root)
    # This unchanged strict chain validates PR35 as the parent, using its exact
    # historical policy in a temporary snapshot. No source policy is rewritten.
    historical = verify_historical_control(root, records, source_bytes)
    require(historical["parent_manifest_sha256"] == document["candidate_manifest_sha256"], "historical PR35 differs")
    proposal = read_json(root / document["candidate_manifest"])
    changes = {change["image_id"]: change for change in proposal["changes"]}
    effective_objects, labels, inventory = {}, {}, {}
    for record in records:
        image_id = record["id"]
        change = changes.get(image_id)
        path = (_relative_file((root / document["candidate_manifest"]).parent, change["candidate_label"])
                if change else _relative_file(root / "data", record["label"]))
        expected = change["candidate_label_sha256"] if change else record["label_sha256"]
        require(file_sha256(path) == expected, "PR35 effective label differs")
        objects = parse_pose_labels(path.read_text(), *record["original_size"], image_id)
        check_geometry(objects, *record["original_size"], image_id)
        effective_objects[image_id], labels[image_id] = objects, (path, expected)
        inventory[record["image"]], inventory[record["label"]] = record["image_sha256"], expected
    require(_same(totals(records, effective_objects), document["training_counts"]), "effective training counts differ")
    return {"records": records, "proposal": proposal, "labels": labels, "inventory": inventory,
            "original_counts": original_counts, "candidate_counts": document["training_counts"],
            "original_objects": original_objects}


def stage_pr35(root, output):
    root = Path(root).resolve()
    context = load_protocol(root)
    document = context["document"]
    candidate = root / document["candidate_manifest"]
    output = checked_output(output, root, candidate, verify=False)
    verified = verified_label_inputs(root, context)
    plans = {"reference": {}, "candidate": {}}
    for record in verified["records"]:
        image = _relative_file(root / "data", record["image"])
        check_png(image, record["image_sha256"], record["original_size"])
        for arm in plans:
            label, label_hash = (verified["labels"][record["id"]] if arm == "candidate" else
                                 (_relative_file(root / "data", record["label"]), record["label_sha256"]))
            for kind, path, expected in (("image", image, record["image_sha256"]), ("label", label, label_hash)):
                plans[arm][record[kind]] = {"source": path, "sha256": expected, "kind": kind, "record": record}
    proposal = verified["proposal"]
    source = {"reference_manifest_sha256": document["source_manifest_sha256"],
              "model_card_sha256": file_sha256(root / "metadata/model_card.json"),
              "initial_checkpoint_sha256": document["initial_checkpoint_sha256"], "checkpoint_bytes_verified": False,
              "candidate_id": proposal["candidate_id"], "candidate_manifest": document["candidate_manifest"],
              "candidate_manifest_sha256": document["candidate_manifest_sha256"],
              "declared_candidate_source_revision": proposal["source_revision"], "source_revision_git_object_checked": False,
              "historical_policy_validation": {"isolated_snapshot": True, "live_policy_unchanged": True,
                                               "historical_policy_sha256": document["historical_policy"]["sha256"]}}
    state = {"root": root, "records": verified["records"], "plans": plans,
             "counts": {"reference": verified["original_counts"], "candidate": verified["candidate_counts"]},
             "source": source, "changed_train_labels": len(proposal["changes"]),
             "removed_objects": sum(len(change["deletions"]) for change in proposal["changes"])}
    output.mkdir(parents=True)
    for arm, entries in plans.items():
        for name, item in entries.items():
            destination = output / arm / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(item["source"], destination)
        (output / arm / "data.yaml").write_bytes(yaml_bytes(output / arm))
    manifest = manifest_fields(output, state, verify_outputs(output, state))
    manifest["created_at_utc"] = datetime.now(timezone.utc).isoformat()
    (output / MANIFEST).write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    return manifest


def metadata_preflight(root):
    root = Path(root).resolve()
    weights, card_hash = _reference_model(root)
    return {"root": str(root), "weights": str(weights), "model_sha256": BASELINE_SHA256,
            "model_card_sha256": card_hash, "manifest_sha256": file_sha256(root / "data/manifest.json")}


def verify_run(root, run, role, context=None, require_weights=True):
    from pose_input_corruption import validate_state
    root, run = Path(root).resolve(), Path(run).resolve()
    require(require_weights is True, "completed experiment verification requires actual returned weights")
    context = context or load_protocol(root)
    receipt = read_json(run / "training-receipt.json")
    require(_same(receipt.get("experiment"), experiment_identity(context, role)), "completed experiment/role binding differs")
    state = receipt.get("input_corruption")
    validate_state(state, context["document"]["augmentation"], role)
    counters = state["counters"]
    require(counters["images_seen"] == receipt["images_processed"]
            and counters["batches_seen"] == receipt["batches_completed"], "hook did not observe every training batch")
    require(counters["images_seen"] > 0 and counters["images_planned"] > 0, "no valid corruption exposure recorded")
    if role == "candidate":
        require(counters["images_applied"] > 0 and counters["known_endpoints_covered_applied"] > 0,
                "candidate input corruption never ran")
    else:
        require(counters["images_applied"] == 0 and counters["known_endpoints_covered_applied"] == 0,
                "control pixels were corrupted")
    expected_sources = source_identity(root)["files_sha256"]
    require(receipt["source"]["files_sha256"] == expected_sources, "completed implementation differs from this checkout")
    bundle = collect_arm(metadata_preflight(root), run, "candidate")
    document, staging = context["document"], bundle["staging"]
    require(receipt["training_settings"]["device"].isdecimal()
            and receipt["runtime"]["device"].startswith("cuda:")
            and receipt["runtime"]["threads"] == 4, "completed experiment requires its fixed CUDA/four-thread runtime")
    expected_images = document["training_counts"]["train"]["images"] * document["execution"]["epochs"]
    expected_batches = ((document["training_counts"]["train"]["images"] + 3) // 4) * document["execution"]["epochs"]
    require(counters["images_seen"] == expected_images and counters["batches_seen"] == expected_batches,
            "completed transform exposure differs from all 30 epochs")
    require(receipt["recipe_id"] == document["recipe_id"]
            and receipt["recipe_sha256"] == document["recipe_sha256"], "completed base recipe differs")
    require(staging["source"]["candidate_manifest"] == document["candidate_manifest"]
            and staging["source"]["candidate_manifest_sha256"] == document["candidate_manifest_sha256"], "completed label composition differs")
    expected_inventory = verified_label_inputs(root, context)["inventory"]
    require({f["path"]: f["output_sha256"] for f in staging["variants"]["candidate"]["files"]} == expected_inventory,
            "actual training inventory differs from PR35")
    for entry in bundle["files"]:
        entry["path"] = role + "/" + entry["path"].split("/", 1)[1]
    return bundle


def package_pair(root, control, candidate, output, verify_existing=False, protocol=DEFAULT_PROTOCOL):
    root = Path(root).resolve()
    context = load_protocol(root, protocol)
    runs = {"control": Path(control).resolve(), "candidate": Path(candidate).resolve()}
    require(all(path.is_dir() and not Path(raw).is_symlink() for raw, path in
                ((control, runs["control"]), (candidate, runs["candidate"]))), "run directories must be regular")
    require(not runs["control"].is_relative_to(runs["candidate"])
            and not runs["candidate"].is_relative_to(runs["control"]), "run directories overlap")
    bundles = {role: verify_run(root, run, role, context) for role, run in runs.items()}
    check_pair(bundles["control"], bundles["candidate"])
    control_state = bundles["control"]["receipt"]["input_corruption"]
    candidate_state = bundles["candidate"]["receipt"]["input_corruption"]
    require(control_state["decision_sha256"] == candidate_state["decision_sha256"],
            "paired mask planning differed")
    rows = [entry for bundle in bundles.values() for entry in bundle["files"]]
    rows.append({"path": "experiment-protocol.json", "sha256": context["sha256"],
                 "size_bytes": context["path"].stat().st_size, "source_path": context["path"]})
    rows.sort(key=lambda row: row["path"])
    receipt = bundles["control"]["receipt"]
    index = {"schema_version": 1, "kind": "baomai-occlusion-training-results",
             "created_at_utc": datetime.now(timezone.utc).isoformat(), "protocol_sha256": context["sha256"],
             "source": receipt["source"], "runtime": receipt["runtime"], "shared_base_settings": shared_settings(receipt),
             "roles": {role: {"checkpoint_manifest": f"{role}/checkpoint-manifest.json",
                              "experiment": bundle["receipt"]["experiment"],
                              "input_corruption": bundle["receipt"]["input_corruption"],
                              "checkpoint_identity": bundle["identity"]} for role, bundle in bundles.items()},
             "files": [{key: value for key, value in row.items() if key != "source_path"} for row in rows],
             "scope": "Completed artifact and controlled-comparison eligibility only; no accuracy approval or model promotion."}
    output = Path(output).absolute()
    require(not output.is_symlink() and output.suffix.lower() == ".zip", "output must be a regular ZIP path")
    output = output.resolve()
    require(all(not output.is_relative_to(path) for path in (root, *runs.values())), "ZIP overlaps source/run directories")
    if verify_existing:
        with zipfile.ZipFile(output) as archive:
            old = json.loads(archive.read(INDEX_NAME))
        index["created_at_utc"] = old["created_at_utc"]
        require(_same(old, index), "existing return index differs from the verified runs")
    else:
        require(not output.exists(), "return ZIP already exists")
    index_bytes = (json.dumps(index, indent=2, allow_nan=False) + "\n").encode()
    if verify_existing:
        verify_archive(output, index_bytes, rows)
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".occlusion-return-", dir=output.parent) as temporary:
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
        verified = verified_label_inputs(args.root, context)
        result = {"status": "verified", "protocol_sha256": context["sha256"],
                  "training_counts": verified["candidate_counts"], "original_image_bytes_verified": False,
                  "initial_weight_bytes_verified": False, "training_executed": False}
    else:
        result = package_pair(args.root, args.control, args.candidate, args.output, args.verify_existing, args.protocol)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
