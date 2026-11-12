#!/usr/bin/env python3
"""Verify a reviewed endpoint-only extension of the exact PR35 label overlay.

The inherited annotation contract remains strict and unchanged. This separate
kind binds each new row and both endpoint slots to recorded review evidence;
it does not establish visual correctness or authorize a training recipe.
"""

import sys
sys.dont_write_bytecode = True

import argparse
import json
from pathlib import Path
import re

from evaluate_pose import _relative_file, file_sha256, parse_pose_labels
from verify_pose_annotation_candidate import (ANNOTATION_CANDIDATE, nonempty,
    read_json, row_text, verify_annotation_candidate)
from verify_pose_candidate import COUNT_KEYS, check_geometry, read_reference, require, totals


ENDPOINT_KIND = "training_endpoint_overlay"
ENDPOINT_CANDIDATE = "data/candidates/endpoint-visibility-2026-11-12/manifest.json"
PARENT_SHA256 = "28fc8f5da95605089d61fb062132200ed296c3a10b216f1a61dbda1798ea4ba7"
SIDES = ("left", "right")
BOUND_FIELDS = ("operation", "original_row", "candidate_row", "source_row_text",
                "candidate_row_text", "changed_fields", "endpoints", "review_status", "reason")


def same(left, right):
    """Compare JSON values without accepting bools as integer substitutes."""
    return json.dumps(left, sort_keys=True, allow_nan=False) == json.dumps(right, sort_keys=True, allow_nan=False)


def descriptor_file(base, descriptor, name):
    require(isinstance(descriptor, dict), f"{name} descriptor must be an object")
    path = _relative_file(base, descriptor["path"])
    require(file_sha256(path) == descriptor["sha256"], f"{name} SHA-256 mismatch")
    return path


def verify_endpoint_candidate(root, candidate):
    root, candidate = Path(root).resolve(), Path(candidate).resolve()
    proposal = read_json(candidate)
    require(type(proposal.get("schema_version")) is int and proposal["schema_version"] == 1,
            "unsupported endpoint candidate schema_version")
    require(proposal.get("status") == "proposed" and proposal.get("kind") == ENDPOINT_KIND,
            "expected a proposed training_endpoint_overlay")
    require(nonempty(proposal.get("candidate_id")), "candidate_id must be a nonempty string")
    require(isinstance(proposal.get("source_revision"), str)
            and re.fullmatch(r"[0-9a-f]{40}", proposal["source_revision"]), "expected full source revision SHA")
    require(same(proposal.get("parent"), {"manifest": ANNOTATION_CANDIDATE, "sha256": PARENT_SHA256}),
            "endpoint parent must be the exact PR35 annotation manifest")
    parent_path = _relative_file(root, ANNOTATION_CANDIDATE)
    require(file_sha256(parent_path) == PARENT_SHA256, "parent manifest SHA-256 mismatch")
    parent_receipt = verify_annotation_candidate(root, parent_path)
    parent = read_json(parent_path)
    for field in ("source_manifest_sha256", "annotation_policy"):
        require(same(proposal[field], parent[field]), f"endpoint candidate changed parent {field}")
    require(proposal["candidate_id"] != parent["candidate_id"], "endpoint candidate needs a separate identity")

    review_path = descriptor_file(candidate.parent, proposal["review"], "review evidence")
    review = read_json(review_path)
    require(type(review.get("schema_version")) is int and review["schema_version"] == 1
            and review.get("candidate_id") == proposal["candidate_id"], "review candidate identity differs")
    require(same(review.get("parent"), proposal["parent"])
            and review.get("source_revision") == proposal["source_revision"], "review source/parent identity differs")
    evidence_path = descriptor_file(root, review["evidence"], "recovered endpoint evidence")
    evidence = read_json(evidence_path)
    recovered = {}
    require(isinstance(evidence.get("endpoint_rows"), list) and evidence["endpoint_rows"],
            "recovered evidence needs endpoint rows")
    for item in evidence["endpoint_rows"]:
        require(isinstance(item, dict) and nonempty(item.get("image_id"))
                and type(item.get("original_row")) is int, "invalid recovered endpoint row")
        key = (item["image_id"], item["original_row"])
        require(key not in recovered, "repeated recovered endpoint row")
        recovered[key] = item

    parent_review = read_json(_relative_file(parent_path.parent, parent["review"]["path"]))
    inherited_decisions = parent_review["decisions"]
    decisions_list = review.get("decisions")
    require(isinstance(decisions_list, list)
            and same(decisions_list[:len(inherited_decisions)], inherited_decisions),
            "inherited review decisions differ from PR35")
    decisions = {}
    for decision in decisions_list:
        require(isinstance(decision, dict) and nonempty(decision.get("id"))
                and decision["id"] not in decisions, "invalid or repeated review decision ID")
        decisions[decision["id"]] = decision
    added_decisions = {item["id"] for item in decisions_list[len(inherited_decisions):]}

    reference_hash, records, by_id, source_bytes, gold, source_counts = read_reference(
        root, proposal["source_manifest_sha256"])
    changes = proposal.get("changes")
    inherited = parent["changes"]
    require(isinstance(changes, list) and len(changes) > len(inherited)
            and same(changes[:len(inherited)], inherited), "inherited change records differ from PR35")
    effective, changed, paths = dict(gold), set(), set()
    for change in inherited:
        image_id = change["image_id"]
        path = _relative_file(candidate.parent, change["candidate_label"])
        require(path.read_bytes() == _relative_file(parent_path.parent, change["candidate_label"]).read_bytes(),
                f"inherited label bytes differ from PR35: {image_id}")
        effective[image_id] = parse_pose_labels(path.read_text(encoding="utf-8"), *by_id[image_id]["original_size"], str(path))
        changed.add(image_id)
        paths.add(change["candidate_label"])

    used_decisions, used_rows = set(), set()
    delta = dict.fromkeys(("changed_train_files", "replaced_rows", "changed_endpoint_slots",
        "visible_to_missing", "missing_to_visible", "visible_coordinate_corrections",
        "retained_missing_slots", "net_visible_endpoint_change", "net_object_change"), 0)
    for change in changes[len(inherited):]:
        require(isinstance(change, dict), "endpoint change must be an object")
        image_id = change["image_id"]
        require(isinstance(image_id, str) and image_id in by_id, "unknown endpoint image")
        record = by_id[image_id]
        require(record["split"] == "train", f"validation changes are forbidden: {image_id}")
        require(image_id not in changed, f"endpoint images must be disjoint from parent and each other: {image_id}")
        changed.add(image_id)
        require(change["source_label"] == change["candidate_label"] == record["label"],
                f"candidate/source label mapping mismatch: {image_id}")
        require(change["source_label_sha256"] == record["label_sha256"]
                and change["source_image_sha256"] == record["image_sha256"], f"source identity mismatch: {image_id}")
        require(change.get("deletions") == [] and change.get("additions") == [],
                f"endpoint overlays cannot add or delete rows: {image_id}")
        lines = source_bytes[image_id].splitlines(keepends=True)
        require(len(lines) == len(gold[image_id]) and all(line.strip() for line in lines),
                f"endpoint overlay needs one nonblank source line per object: {image_id}")
        require(all(type(change.get(key)) is int and change[key] == len(lines)
                    for key in ("source_objects", "candidate_objects")), f"endpoint object count changed: {image_id}")
        identity_map = [{"original_row": n, "candidate_row": n} for n in range(1, len(lines) + 1)]
        require(same(change.get("row_map"), identity_map), f"endpoint row map must preserve every row identity: {image_id}")
        operations = change.get("replacements")
        require(isinstance(operations, list) and operations, f"endpoint changes need replacements: {image_id}")
        expected = list(lines)
        for operation in operations:
            require(isinstance(operation, dict) and operation.get("operation") == "endpoint_amendment",
                    "expected endpoint_amendment operation")
            number = operation.get("original_row")
            require(type(number) is int and 1 <= number <= len(lines)
                    and type(operation.get("candidate_row")) is int and operation["candidate_row"] == number,
                    f"endpoint amendment must preserve original row number: {image_id}")
            key = (image_id, number)
            require(key not in used_rows and key in recovered, f"undeclared or repeated recovered row: {image_id}:{number}")
            used_rows.add(key)
            identifier = operation.get("review_id")
            require(identifier in added_decisions and identifier not in used_decisions,
                    f"endpoint amendment needs one new accepted review: {image_id}:{number}")
            used_decisions.add(identifier)
            decision = decisions[identifier]
            require(decision.get("image_id") == image_id
                    and decision.get("source_image_sha256") == record["image_sha256"]
                    and decision.get("review_status") == "accepted_incremental" and nonempty(decision.get("reason"))
                    and all(same(decision.get(field), operation.get(field)) for field in BOUND_FIELDS),
                    f"endpoint operation differs from accepted review: {image_id}:{number}")
            require(same(decision.get("evidence"), review["evidence"]), "endpoint review evidence identity differs")
            require(operation.get("donor_endpoints") == [], "endpoint amendments cannot transfer donor points")
            original_text = lines[number - 1].decode("utf-8").rstrip("\r\n")
            require(operation.get("source_row_text") == original_text, f"source row audit text differs: {image_id}:{number}")
            before = original_text.split()
            after = row_text(operation["candidate_row_text"], record["original_size"], f"{image_id}:{number}")
            require(before[:5] == after[:5], f"endpoint amendments must preserve class and bbox tokens: {image_id}:{number}")
            fields = [f"{side}_endpoint" for i, side in enumerate(SIDES)
                      if list(map(float, before[5 + 3*i:8 + 3*i])) != list(map(float, after[5 + 3*i:8 + 3*i]))]
            require(fields and operation.get("changed_fields") == fields,
                    f"changed endpoint field inventory differs: {image_id}:{number}")
            endpoints = operation.get("endpoints")
            require(isinstance(endpoints, list) and len(endpoints) == 2
                    and all(isinstance(item, dict) for item in endpoints)
                    and [item.get("side") for item in endpoints] == list(SIDES), "endpoint slot inventory must be left then right")
            for i, endpoint in enumerate(endpoints):
                a, b = before[5 + 3*i:8 + 3*i], after[5 + 3*i:8 + 3*i]
                require(endpoint.get("before") == a and endpoint.get("after") == b
                        and nonempty(endpoint.get("reason")), f"endpoint slot audit differs: {image_id}:{number}:{SIDES[i]}")
                point = recovered[key][f"{SIDES[i]}_original_pixels"]
                ranges = recovered[key]["visible_point_descriptive_ranges_xy"]
                uncertainty = None if point is None else {"x_range": ranges[0], "y_range": ranges[1]}
                require(same(endpoint.get("accepted_original_pixels"), point)
                        and same(endpoint.get("coordinate_uncertainty_pixels"), uncertainty),
                        f"endpoint coordinate provenance differs: {image_id}:{number}:{SIDES[i]}")
                if f"{SIDES[i]}_endpoint" not in fields:
                    require(a == b, "unchanged endpoint tokens differ")
                old_visible, new_visible = float(a[2]) == 2, float(b[2]) == 2
                delta["changed_endpoint_slots"] += a != b
                delta["visible_to_missing"] += old_visible and not new_visible
                delta["missing_to_visible"] += not old_visible and new_visible
                delta["visible_coordinate_corrections"] += old_visible and new_visible and a != b
                delta["retained_missing_slots"] += not old_visible and not new_visible
                delta["net_visible_endpoint_change"] += int(new_visible) - int(old_visible)
            require(before[5:] == recovered[key]["before_tokens"] and after[5:] == recovered[key]["after_tokens"],
                    f"endpoint row differs from recovered accepted tokens: {image_id}:{number}")
            ending = lines[number - 1][len(original_text.encode("utf-8")):]
            require(ending in (b"", b"\n", b"\r\n"), "unsupported source line ending")
            expected[number - 1] = operation["candidate_row_text"].encode("utf-8") + ending
            delta["replaced_rows"] += 1
        path = _relative_file(candidate.parent, change["candidate_label"])
        require(file_sha256(path) == change["candidate_label_sha256"], f"candidate label SHA-256 mismatch: {image_id}")
        require(path.read_bytes() == b"".join(expected), f"unlisted rows/bytes changed in endpoint candidate: {image_id}")
        effective[image_id] = parse_pose_labels(path.read_text(encoding="utf-8"), *record["original_size"], str(path))
        check_geometry(effective[image_id], *record["original_size"], str(path))
        paths.add(change["candidate_label"])
        delta["changed_train_files"] += 1
    require(used_decisions == added_decisions and used_rows == set(recovered),
            "endpoint review/recovery row inventory differs from applied operations")

    labels = candidate.parent / "labels"
    require(not labels.is_symlink(), "overlay labels directory must not be a symlink")
    actual_paths = set()
    for path in labels.rglob("*"):
        require(not path.is_symlink() and (path.is_file() or path.is_dir()), "overlay labels must contain only regular files/directories")
        if path.is_file():
            actual_paths.add(path.relative_to(candidate.parent).as_posix())
    require(actual_paths == paths, "unexpected or missing endpoint overlay labels")
    counts = totals(records, effective)
    expected_counts = proposal["expected_counts"]
    require(isinstance(expected_counts, dict) and set(expected_counts) == {"train", "val"}
            and all(isinstance(expected_counts[split], dict) and set(expected_counts[split]) == set(COUNT_KEYS)
                    and all(type(value) is int and value >= 0 for value in expected_counts[split].values())
                    for split in ("train", "val")), "invalid expected counts")
    require(counts == expected_counts and counts["val"] == source_counts["val"], "endpoint candidate totals differ")
    parent_counts = parent_receipt["counts"]["train"]
    require(all(counts["train"][key] == parent_counts[key] for key in ("images", "objects", "empty_images"))
            and counts["train"]["visible_endpoints"] == parent_counts["visible_endpoints"] + delta["net_visible_endpoint_change"]
            and counts["train"]["missing_endpoints"] == parent_counts["missing_endpoints"] - delta["net_visible_endpoint_change"],
            "endpoint-only count delta differs")
    edits = {"changed_train_files": len(changed), "added_rows": parent_receipt["added_objects"],
        "deleted_rows": parent_receipt["removed_objects"],
        "replaced_rows": parent_receipt["replaced_objects"] + delta["replaced_rows"],
        "donor_endpoints_transferred": parent_receipt["merged_endpoints"],
        "net_object_change": counts["train"]["objects"] - source_counts["train"]["objects"]}
    require(same(proposal.get("expected_edit_counts"), edits)
            and same(proposal.get("new_endpoint_edit_counts"), delta), "endpoint edit totals differ")
    scripts = Path(__file__).resolve().parent
    return {"status": "verified", "kind": ENDPOINT_KIND, "candidate_id": proposal["candidate_id"],
        "candidate_manifest_sha256": file_sha256(candidate), "declared_source_revision": proposal["source_revision"],
        "source_revision_git_object_checked": False, "source_manifest_sha256": reference_hash,
        "parent_manifest_sha256": PARENT_SHA256, "parent_verified_by": "verify_pose_annotation_candidate.py",
        "review_sha256": proposal["review"]["sha256"], "annotation_policy_sha256": proposal["annotation_policy"]["sha256"],
        "reference_labels_verified": len(records), "changed_train_labels": len(changed),
        "added_objects": edits["added_rows"], "removed_objects": edits["deleted_rows"],
        "replaced_objects": edits["replaced_rows"], "merged_endpoints": edits["donor_endpoints_transferred"],
        "counts": counts, "new_endpoint_edit_counts": delta, "validation_labels_unchanged": True,
        "parent_labels_and_reviews_preserved": True, "unchanged_rows_byte_preserved": True,
        "full_row_maps_verified": True, "class_and_boxes_preserved_in_endpoint_delta": True,
        "image_bytes_verified": False, "images_staged_by_check": False, "dataset_built": False,
        "visual_correctness_verified": False, "mechanical_geometry_verified": True, "training_recipe_supported": False,
        "code_sha256": {f"scripts/{name}": file_sha256(scripts / name) for name in
            ("verify_pose_endpoint_candidate.py", "verify_pose_annotation_candidate.py", "verify_pose_candidate.py",
             "evaluate_pose.py", "pose_scoring.py")}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--candidate", type=Path, default=Path(ENDPOINT_CANDIDATE))
    args = parser.parse_args()
    candidate = args.candidate if args.candidate.is_absolute() else args.root / args.candidate
    try:
        result = verify_endpoint_candidate(args.root, candidate)
    except (OSError, ValueError, KeyError, TypeError, IndexError, AttributeError) as error:
        print(json.dumps({"status": "invalid", "error": str(error)}, allow_nan=False))
        return 1
    print(json.dumps(result, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
