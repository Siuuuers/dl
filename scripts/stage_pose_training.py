#!/usr/bin/env python3
"""Copy and verify paired baomai training datasets without running training."""

import sys
sys.dont_write_bytecode = True

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import struct

from evaluate_pose import PNG_SIGNATURE, _relative_file, file_sha256, parse_pose_labels
from verify_pose_candidate import (DEFAULT_CANDIDATE, GEOMETRY_EPSILON_PX,
                                   check_geometry, require, totals, verify_candidate)


ARMS = ("reference", "candidate")
MANIFEST = "staging-manifest.json"


def check_png(path, expected_hash, dimensions):
    with path.open("rb") as source:
        header = source.read(24)
    require(len(header) == 24 and header[:8] == PNG_SIGNATURE and header[12:16] == b"IHDR",
            f"expected actual original PNG bytes: {path}")
    require(list(struct.unpack(">II", header[16:24])) == dimensions,
            f"original PNG dimensions mismatch: {path}")
    actual_hash = file_sha256(path)
    require(actual_hash == expected_hash, f"PNG SHA-256 mismatch: {path}")
    return actual_hash


def source_name(path, root):
    return path.relative_to(root).as_posix() if path.is_relative_to(root) else str(path)


def source_state(root, candidate):
    """Verify all reference/proposal labels and every original before any copies."""
    proposal = json.loads(candidate.read_text(encoding="utf-8"))
    require(isinstance(proposal, dict), "candidate manifest must be an object")
    if proposal.get("kind") == "training_box_overlay":
        from verify_pose_box_candidate import verify_box_candidate
        receipt = verify_box_candidate(root, candidate)
    elif proposal.get("kind") == "training_annotation_overlay":
        from verify_pose_annotation_candidate import verify_annotation_candidate
        receipt = verify_annotation_candidate(root, candidate)
    elif proposal.get("kind") == "training_endpoint_overlay":
        from verify_pose_endpoint_candidate import verify_endpoint_candidate
        receipt = verify_endpoint_candidate(root, candidate)
    elif proposal.get("kind") == "training_composed_overlay":
        from verify_pose_composed_candidate import verify_composed_candidate
        receipt = verify_composed_candidate(root, candidate)
    elif proposal.get("kind") == "training_observed_review_composition":
        from verify_observed_training_candidate import verify_observed_training_candidate
        receipt = verify_observed_training_candidate(root, candidate)
    elif proposal.get("kind") == "training_incremental_observed_overlay":
        from verify_incremental_pose_candidate import verify_incremental_candidate
        receipt = verify_incremental_candidate(root, candidate)
    else:
        receipt = verify_candidate(root, candidate)
    reference = json.loads((root / "data/manifest.json").read_text(encoding="utf-8"))
    require(file_sha256(candidate) == receipt["candidate_manifest_sha256"], "proposal changed during preflight")
    card_path = _relative_file(root, "metadata/model_card.json")
    card = json.loads(card_path.read_text(encoding="utf-8"))
    require(card["task"] == "pose" and card["class_names"] == ["cassette_front"]
            and card["kpt_shape"] == reference["kpt_shape"] == [2, 3]
            and card["flip_idx"] == [1, 0] and card["sha256"] == reference["model_sha256"],
            "reference metadata differs from the baomai two-corner contract")
    records = sorted(reference["records"], key=lambda row: (row["split"], row["id"]))
    changes = {change["image_id"]: change for change in proposal["changes"]}
    control_path, control_changes = None, {}
    if proposal.get("kind") in ("training_composed_overlay", "training_observed_review_composition",
                                "training_incremental_observed_overlay"):
        control = receipt["control"]
        if proposal["kind"] == "training_composed_overlay":
            require(proposal["control"] == {"manifest": control["manifest"], "sha256": control["manifest_sha256"]},
                    "composed control identity differs from its verification receipt")
        control_path = _relative_file(root, control["manifest"])
        require(file_sha256(control_path) == control["manifest_sha256"], "control manifest changed during preflight")
        control_proposal = json.loads(control_path.read_text(encoding="utf-8"))
        expected_control_kind = ("training_endpoint_overlay" if proposal["kind"] == "training_composed_overlay"
                                 else "training_annotation_overlay")
        require(control_proposal["kind"] == expected_control_kind
                and control_proposal["candidate_id"] == control["candidate_id"], "control is not the verified supervision overlay")
        control_changes = {change["image_id"]: change for change in control_proposal["changes"]}
        require(len(control_changes) == control["changed_train_labels"], "control changed-label count differs")
    plans, parsed = {arm: {} for arm in ARMS}, {arm: {} for arm in ARMS}
    for record in records:
        image = _relative_file(root / "data", record["image"])
        check_png(image, record["image_sha256"], record["original_size"])
        for arm in ARMS:
            arm_changes, overlay = (changes, candidate) if arm == "candidate" else (control_changes, control_path)
            change = arm_changes.get(record["id"])
            label = (_relative_file(overlay.parent, change["candidate_label"]) if change else
                     _relative_file(root / "data", record["label"]))
            label_hash = change["candidate_label_sha256"] if change else record["label_sha256"]
            require(file_sha256(label) == label_hash, f"label changed during preflight: {label}")
            objects = parse_pose_labels(label.read_text(encoding="utf-8"), *record["original_size"], str(label))
            check_geometry(objects, *record["original_size"], str(label))
            parsed[arm][record["id"]] = objects
            for kind, path, digest in (("image", image, record["image_sha256"]), ("label", label, label_hash)):
                plans[arm][record[kind]] = {"source": path, "sha256": digest, "kind": kind, "record": record}
    counts = {arm: totals(records, parsed[arm]) for arm in ARMS}
    require(counts["candidate"] == receipt["counts"], "candidate counts changed during preflight")
    if control_path is not None:
        require(counts["reference"] == receipt["control"]["counts"], "endpoint control counts changed during preflight")
    require(counts["reference"]["val"] == counts["candidate"]["val"], "validation counts differ")
    source = {"reference_manifest_sha256": receipt["source_manifest_sha256"],
              "model_card_sha256": file_sha256(card_path), "initial_checkpoint_sha256": card["sha256"],
              "checkpoint_bytes_verified": False, "candidate_id": receipt["candidate_id"],
              "candidate_manifest": source_name(candidate, root),
              "candidate_manifest_sha256": receipt["candidate_manifest_sha256"],
              "declared_candidate_source_revision": receipt["declared_source_revision"],
              "source_revision_git_object_checked": False}
    if control_path is not None:
        source.update(candidate_kind=proposal["kind"], control=receipt["control"])
    if proposal.get("kind") in ("training_observed_review_composition", "training_incremental_observed_overlay"):
        source["training_protocol"] = receipt["training_protocol"]
    if proposal.get("kind") == "training_incremental_observed_overlay":
        source["retained_legacy_image_ids"] = receipt["retained_legacy_image_ids"]
    return {"root": root, "records": records, "plans": plans, "counts": counts, "source": source,
            "changed_train_labels": receipt["changed_train_labels"], "removed_objects": receipt["removed_objects"]}


def checked_output(path, root, candidate, *, verify):
    path = Path(path).absolute()
    require(not path.is_symlink(), "staging output must not be a symlink")
    path = path.resolve()
    for protected in ((root / "data").resolve(), candidate.parent.resolve()):
        require(not path.is_relative_to(protected) and not protected.is_relative_to(path),
                "staging output overlaps frozen data or candidate source")
    if verify:
        require(path.is_dir(), "verification requires an existing staging directory")
    else:
        require(not path.exists(), "staging output already exists; choose a fresh directory")
    return path


def yaml_bytes(arm_root):
    return (f"path: {json.dumps(str(arm_root), ensure_ascii=False)}\n"
            "train: images/train\nval: images/val\nnames:\n  0: cassette_front\n"
            "kpt_shape: [2, 3]\nflip_idx: [1, 0]\nkpt_names:\n"
            "  0: [front_lower_left, front_lower_right]\n").encode("utf-8")


def verify_outputs(output, state):
    expected_files = {f"{arm}/{relative}" for arm in ARMS for relative in state["plans"][arm]}
    expected_files.update(f"{arm}/data.yaml" for arm in ARMS)
    expected_dirs = {str(parent) for name in expected_files for parent in Path(name).parents if str(parent) != "."}
    actual_files, actual_dirs = set(), set()
    for path in output.rglob("*"):
        require(not path.is_symlink(), f"symlinks are forbidden in staged data: {path}")
        relative = path.relative_to(output).as_posix()
        if path.is_dir():
            actual_dirs.add(relative)
        else:
            require(path.is_file(), f"unexpected nonregular staged path: {path}")
            actual_files.add(relative)
    require(actual_files - {MANIFEST} == expected_files and actual_dirs == expected_dirs,
            "unexpected or missing staged files/directories (including caches)")
    variants = {}
    for arm in ARMS:
        arm_root, objects, files = output / arm, {}, []
        yaml_path = arm_root / "data.yaml"
        require(yaml_path.read_bytes() == yaml_bytes(arm_root), f"dataset YAML/root mismatch: {arm}")
        for relative, item in sorted(state["plans"][arm].items()):
            path, record = arm_root / relative, item["record"]
            actual_hash = file_sha256(path)
            require(actual_hash == item["sha256"], f"staged file SHA-256 mismatch: {path}")
            if item["kind"] == "image":
                check_png(path, item["sha256"], record["original_size"])
            else:
                objects[record["id"]] = parse_pose_labels(path.read_text(encoding="utf-8"),
                                                         *record["original_size"], str(path))
                check_geometry(objects[record["id"]], *record["original_size"], str(path))
            files.append({"path": relative, "source_path": source_name(item["source"], state["root"]),
                          "source_sha256": item["sha256"], "output_sha256": actual_hash})
        actual_counts = totals(state["records"], objects)
        require(actual_counts == state["counts"][arm], f"staged split totals mismatch: {arm}")
        variants[arm] = {"data_yaml": f"{arm}/data.yaml", "data_yaml_sha256": file_sha256(yaml_path),
                         "counts": actual_counts, "files": files}
    return variants


def code_hashes(*, composed=False, observed=False, incremental=False):
    scripts = Path(__file__).resolve().parent
    names = ("stage_pose_training.py", "verify_pose_candidate.py", "verify_pose_box_candidate.py",
             "verify_pose_annotation_candidate.py", "verify_pose_endpoint_candidate.py",
             "evaluate_pose.py", "pose_scoring.py")
    if composed:
        names += ("verify_pose_composed_candidate.py",)
    if observed or incremental:
        names += ("verify_observed_training_candidate.py", "compose_observed_pose_review.py",
                  "verify_pose_composed_candidate.py")
    if incremental:
        names += ("verify_incremental_pose_candidate.py", "compose_incremental_pose_candidate.py")
    return {f"scripts/{name}": file_sha256(scripts / name) for name in names}


def manifest_fields(output, state, variants):
    composed = state["source"].get("candidate_kind") == "training_composed_overlay"
    observed = state["source"].get("candidate_kind") == "training_observed_review_composition"
    incremental = state["source"].get("candidate_kind") == "training_incremental_observed_overlay"
    result = {"schema_version": 1, "status": "complete", "output_root": str(output),
            "source": state["source"], "code_sha256": code_hashes(composed=composed, observed=observed, incremental=incremental), "variants": variants,
            "verification": {"source_labels_verified": 157, "actual_original_pngs_verified": 157,
                             "copied_images_verified": 314, "copied_labels_verified": 314,
                             "changed_train_labels": state["changed_train_labels"],
                             "removed_training_objects": state["removed_objects"],
                             "validation_images_and_labels_identical": True,
                             "source_files_rewritten": False, "symlinks_present": False,
                             "trainer_loader_executed": False},
            "mechanical_geometry_gate": {"passed": True, "serialization_tolerance_original_px": GEOMETRY_EPSILON_PX,
                                        "box_edges_within_closed_image_bounds": True,
                                        "visible_endpoints_within_boxes": True,
                                        "visible_endpoints_strictly_in_frame": True,
                                        "coordinates_clamped_or_rewritten": False},
            "physical_policy_conformance": {"visually_certified": False,
                                            "scope": "Mechanical schema/geometry and proposal provenance only; no new image adjudication."},
            "training_run": False}
    if composed:
        result["verification"].update(
            reference_labels_are_verified_endpoint_control=True,
            control_changed_train_labels=state["source"]["control"]["changed_train_labels"],
            removed_training_objects_vs_control=state["counts"]["reference"]["train"]["objects"]
                - state["counts"]["candidate"]["train"]["objects"],
            removed_training_objects_denominator="original source labels")
    if observed or incremental:
        result["verification"].update(
            reference_labels_are_verified_pr35_control=True,
            control_changed_train_labels=state["source"]["control"]["changed_train_labels"],
            net_removed_training_objects_vs_control=state["counts"]["reference"]["train"]["objects"]
                - state["counts"]["candidate"]["train"]["objects"],
            training_readiness_verified=True)
    if incremental:
        result["verification"].update(
            retained_photo_labels_identical_between_arms=True,
            retained_legacy_images=len(state["source"]["retained_legacy_image_ids"]),
            full_dataset_certified=False,
            training_readiness_scope="controlled_incremental_experiment_only")
    return result


def stage_datasets(root, candidate, output):
    root, candidate = Path(root).resolve(), Path(candidate).resolve()
    output = checked_output(output, root, candidate, verify=False)
    state = source_state(root, candidate)
    output.mkdir(parents=True, exist_ok=False)
    for arm in ARMS:
        for relative, item in state["plans"][arm].items():
            target = output / arm / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(item["source"], target)
        (output / arm / "data.yaml").write_bytes(yaml_bytes(output / arm))
    variants = verify_outputs(output, state)
    manifest = manifest_fields(output, state, variants)
    manifest["created_at_utc"] = datetime.now(timezone.utc).isoformat()
    with (output / MANIFEST).open("x", encoding="utf-8") as destination:
        json.dump(manifest, destination, indent=2, allow_nan=False)
        destination.write("\n")
    return result_summary("staged", output, manifest)


def verify_staging(root, candidate, output):
    root, candidate = Path(root).resolve(), Path(candidate).resolve()
    output = checked_output(output, root, candidate, verify=True)
    state = source_state(root, candidate)
    variants = verify_outputs(output, state)
    manifest = json.loads((output / MANIFEST).read_text(encoding="utf-8"))
    expected = manifest_fields(output, state, variants)
    require(isinstance(manifest, dict) and set(manifest) == set(expected) | {"created_at_utc"},
            "invalid staging manifest fields")
    require(isinstance(manifest["created_at_utc"], str)
            and datetime.fromisoformat(manifest["created_at_utc"]).utcoffset() is not None,
            "staging manifest requires a timezone-aware creation time")
    actual_fields = {key: manifest[key] for key in expected}
    require(json.dumps(actual_fields, sort_keys=True, allow_nan=False)
            == json.dumps(expected, sort_keys=True, allow_nan=False),
            "staging manifest differs from freshly verified source, output, configuration or code")
    return result_summary("verified", output, manifest)


def result_summary(status, output, manifest):
    return {"status": status, "output": str(output), "staging_manifest_sha256": file_sha256(output / MANIFEST),
            "counts": {arm: manifest["variants"][arm]["counts"] for arm in ARMS},
            "actual_original_pngs_verified": 157, "geometry_gate_passed": True,
            "physical_policy_visually_certified": False, "training_run": False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--candidate", type=Path, default=Path(DEFAULT_CANDIDATE),
                        help="proposal manifest; relative paths resolve from --root")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--output", type=Path, help="fresh directory for complete copied reference/candidate datasets")
    mode.add_argument("--verify", type=Path, metavar="OUTPUT", help="read-only verification of an existing staging tree")
    args = parser.parse_args(argv)
    candidate = args.candidate if args.candidate.is_absolute() else args.root / args.candidate
    try:
        result = (verify_staging(args.root, candidate, args.verify) if args.verify is not None else
                  stage_datasets(args.root, candidate, args.output))
    except (OSError, ValueError, KeyError, TypeError, IndexError, AttributeError) as error:
        print(json.dumps({"status": "invalid", "error": str(error)}, allow_nan=False))
        return 1
    print(json.dumps(result, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
