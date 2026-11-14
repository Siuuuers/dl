#!/usr/bin/env python3
"""Compose explicit observed-label edits on the frozen fixed-geometry parent.

This review tool writes a new training overlay, preserves all unreviewed tokens,
and verifies the historical parent through its unchanged verifier. It cannot
publish a training recipe, edit validation, or infer labels from rack spacing.
"""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
from evaluate_pose import parse_pose_labels
from refine_reconciled_pose_labels import box, revision as parent_revision
from verify_pose_candidate import check_geometry, read_reference, totals
from verify_reconciled_pose_labels import bound, digest, encoded, read_json, require, sha, verify_tree

KIND = "training_observed_review_composition"
PARENT = {
    "path": "data/candidates/fixed-geometry-labels-2026-11-14/manifest.json",
    "sha256": "509a6d755b4ac2b1af679679debccbf2b038e60fc98ebfa7843a38bddc4ad05d",
}
BOX_FIELDS = tuple("bbox." + edge for edge in ("left", "top", "right", "bottom"))
POINT_FIELDS = {f"{side}_endpoint.{axis}": index + offset
                for side, index in (("left", 5), ("right", 8))
                for axis, offset in (("x", 0), ("y", 1), ("visibility", 2))}
ZERO_POINTS = ["0.00000000", "0.00000000", "0", "0.00000000", "0.00000000", "0"]


def identity(item):
    original, addition = item.get("original_row"), item.get("addition_id")
    require((type(original) is int and original > 0 and addition is None)
            or (original is None and isinstance(addition, str) and addition.strip()),
            "one stable original row or addition ID required")
    return (original, addition)


def one_line(text, source):
    require(isinstance(text, str) and text and text.splitlines() == [text]
            and text == text.strip(), f"one nonblank physical line required: {source}")
    tokens = text.split()
    require(len(tokens) == 11, f"eleven tokens required: {source}")
    return tokens


def box_iou(first, second):
    intersection = max(0, min(first[2], second[2]) - max(first[0], second[0])) * max(
        0, min(first[3], second[3]) - max(first[1], second[1]))
    union = ((first[2] - first[0]) * (first[3] - first[1])
             + (second[2] - second[0]) * (second[3] - second[1]) - intersection)
    return intersection / union if union > 0 else 0


def replacement(op, size):
    old = one_line(op["before_text"], "before")
    new = one_line(op["after_text"], "after")
    require(new[7] in ("0", "2") and new[10] in ("0", "2"),
            "replacement visibility tokens must be literal 0 or 2")
    fields = op["changed_fields"]
    require(isinstance(fields, list) and fields and len(fields) == len(set(fields))
            and set(fields) <= set(BOX_FIELDS) | set(POINT_FIELDS), "invalid changed fields")
    require(new[0] == old[0], "class changed")
    old_box, new_box = box(old, size), box(new, size)
    for name, before, after in zip(BOX_FIELDS, old_box, new_box):
        if name in fields:
            require(abs(before - after) > 0.00002, "ineffective declared box bound")
        else:
            require(abs(before - after) <= 0.00002, "undeclared box bound changed")
    if not set(fields) & set(BOX_FIELDS):
        require(old[1:5] == new[1:5], "unreviewed bbox tokens changed")
    else:
        target = op["after_bbox_xyxy_px"]
        require(len(target) == 4 and all(abs(a - b) <= 0.00002 for a, b in zip(target, new_box)),
                "encoded box differs from reviewed pixels")
    if not set(fields) & {"bbox.left", "bbox.right"}:
        require(old[1] == new[1] and old[3] == new[3], "unreviewed horizontal bbox tokens changed")
    if not set(fields) & {"bbox.top", "bbox.bottom"}:
        require(old[2] == new[2] and old[4] == new[4], "unreviewed vertical bbox tokens changed")
    for name, index in POINT_FIELDS.items():
        require((old[index] != new[index]) == (name in fields),
                "endpoint token change does not match declaration")
    for side, start in (("left", 5), ("right", 8)):
        if not any(name.startswith(side + "_endpoint.") for name in fields):
            continue
        target = op["after_keypoints_xyv_px"][side]
        require(len(target) == 3 and target[2] in (0, 2), "invalid observed point state")
        require(abs(float(new[start]) * size[0] - target[0]) <= 0.00002
                and abs(float(new[start + 1]) * size[1] - target[1]) <= 0.00002
                and int(new[start + 2]) == target[2], "encoded point differs from reviewed pixels")
        if new[start + 2] == "0":
            require(new[start:start + 3] == ["0.00000000", "0.00000000", "0"],
                    "missing endpoint is not canonical")
        if old[start + 2] == "0" and new[start + 2] == "2":
            require(op.get("new_visible_point_evidence", {}).get(side) == "actual_corner_visible_in_original",
                    "hidden point cannot become visible from geometry alone")
    return op["after_text"] + "\n"


def compose(root, review_path):
    review = read_json(review_path)
    require(review.get("schema_version") == 1 and review.get("kind") == KIND, "unexpected review schema")
    require(review["parent"] == PARENT, "unexpected parent")
    require(review["release_ready"] is False and review["training_recipe_id"] is None,
            "review cannot claim training release readiness")
    for descriptor in review["frozen_code"] + review["evidence"]:
        bound(root, descriptor)
    parent_path = bound(root, PARENT)
    parent = read_json(parent_path)
    inherited, reconstructed = parent_revision(root, bound(root, parent["review"]))
    verify_tree(parent_path.parent, inherited)
    require(parent == reconstructed, "historical parent reconstruction differs")
    _, records, by_id, source_bytes, _, _ = read_reference(root, parent["source_manifest_sha256"])
    for record in records:
        require(sha(root / "data" / record["image"]) == record["image_sha256"], "original image changed")
    grouped, seen = defaultdict(list), set()
    require(isinstance(review["operations"], list) and review["operations"], "no accepted operations")
    for op in review["operations"]:
        key = (op["image_id"], *identity(op))
        require(key not in seen, "repeated object operation; combine its reviewed fields")
        seen.add(key)
        require(op["status"] == "accepted" and op["action"] in ("replace", "add", "delete"),
                "unsupported or unaccepted operation")
        require(op["coordinate_evidence"] == "original_image_review", "inferred geometry is ineligible")
        require(isinstance(op["reason"], str) and op["reason"].strip(), "physical evidence required")
        record = by_id[op["image_id"]]
        require(record["split"] == "train", "validation edit forbidden")
        require(op["source_image_sha256"] == record["image_sha256"]
                and op["source_label_sha256"] == record["label_sha256"], "original identity differs")
        grouped[op["image_id"]].append(op)
    parent_changes = {c["image_id"]: c for c in parent["changes"]}
    payloads, changes, effective = {}, [], {}
    for image_id, record in sorted(by_id.items()):
        relative, original = record["label"], source_bytes[image_id]
        before = inherited.get(relative, original)
        lines = before.decode().splitlines(keepends=True)
        prior = parent_changes.get(image_id)
        mapping = (prior["row_map"] if prior else
                   [{"original_row": i, "candidate_row": i} for i in range(1, len(lines) + 1)])
        rows = {identity(item): item["candidate_row"] for item in mapping}
        require(len(rows) == len(mapping), "duplicate stable identity in parent map")
        remove, replace, append = set(), {}, []
        for op in grouped[image_id]:
            key = identity(op)
            if op["action"] == "add":
                require(key[0] is None and key not in rows and op["parent_row"] is None
                        and op["before_text"] is None, "addition reuses a historical identity")
                added_tokens = one_line(op["after_text"], "addition")
                require(op["instance_evidence"] == "distinct_front_visible_in_original",
                        "addition requires an observed distinct front")
                require(added_tokens[7] in ("0", "2") and added_tokens[10] in ("0", "2"),
                        "new visibility tokens must be literal 0 or 2")
                target_box = op["after_bbox_xyxy_px"]
                require(len(target_box) == 4 and all(abs(a - b) <= 0.00002 for a, b in
                        zip(target_box, box(added_tokens, record["original_size"]))),
                        "added box differs from reviewed pixels")
                overlapping = {index: line.rstrip("\r\n") for index, line in enumerate(lines, 1)
                               if box_iou(target_box, box(line.split(), record["original_size"])) >= 0.75}
                dispositions = op.get("overlap_adjudications", [])
                require(isinstance(dispositions, list)
                        and len(dispositions) == len(overlapping)
                        and {item["parent_row"] for item in dispositions} == set(overlapping),
                        "addition overlaps a parent instance; review the complete parent labels, including inherited additions")
                for item in dispositions:
                    require(item["before_text"] == overlapping[item["parent_row"]]
                            and item["disposition"] == "distinct_front_in_original_image"
                            and isinstance(item["reason"], str) and item["reason"].strip(),
                            "overlap needs explicit distinct-instance evidence; refine the existing row for the same front")
                for side, start in (("left", 5), ("right", 8)):
                    target = op["after_keypoints_xyv_px"][side]
                    require(len(target) == 3 and target[2] in (0, 2)
                            and abs(float(added_tokens[start]) * record["original_size"][0] - target[0]) <= 0.00002
                            and abs(float(added_tokens[start + 1]) * record["original_size"][1] - target[1]) <= 0.00002
                            and int(added_tokens[start + 2]) == target[2],
                            "added endpoint differs from reviewed pixels")
                    if added_tokens[start + 2] == "2":
                        require(op.get("new_visible_point_evidence", {}).get(side)
                                == "actual_corner_visible_in_original", "added visible point needs original evidence")
                append.append(op)
                continue
            current = rows.get(key)
            require(type(current) is int and current == op["parent_row"], "parent row map differs")
            old_line = lines[current - 1].rstrip("\r\n")
            require(old_line == op["before_text"], "exact parent row differs")
            if op["action"] == "delete":
                removed_tokens = one_line(old_line, "deletion")
                visible_sides = {side for side, start in (("left", 5), ("right", 8))
                                 if removed_tokens[start + 2] == "2"}
                if visible_sides:
                    require(op.get("deletion_evidence") == "non_target_material_in_original",
                            "removing a locatable label needs explicit non-target evidence")
                    discarded = op.get("discarded_endpoint_evidence", {})
                    require(set(discarded) == visible_sides, "every discarded locatable endpoint needs a disposition")
                    for side, start in (("left", 5), ("right", 8)):
                        if side not in visible_sides:
                            continue
                        evidence = discarded[side]
                        require(evidence["before_tokens"] == removed_tokens[start:start + 3]
                                and evidence["physical_feature"] == "non_target_material"
                                and isinstance(evidence["reason"], str) and evidence["reason"].strip(),
                                "discarded endpoint evidence does not identify the reviewed feature")
                else:
                    require(all(float(value) == 0 for value in removed_tokens[5:]),
                            "missing-point deletion is not canonical")
                require(op["after_text"] is None, "deletion has unexpected replacement")
                remove.add(current)
            else:
                replace[current] = replacement(op, record["original_size"])
        result = "".join(replace.get(i, line) for i, line in enumerate(lines, 1) if i not in remove)
        result += "".join(op["after_text"] + "\n" for op in append)
        result = result.encode()
        row_map = []
        for item in mapping:
            item = dict(item)
            current = item["candidate_row"]
            item["parent_row"] = current
            item["candidate_row"] = (None if current is None or current in remove else
                                     current - sum(r < current for r in remove))
            row_map.append(item)
        for index, op in enumerate(append, len(lines) - len(remove) + 1):
            row_map.append({"original_row": None, "addition_id": op["addition_id"],
                            "parent_row": None, "candidate_row": index})
        parsed = parse_pose_labels(result.decode(), *record["original_size"], image_id)
        check_geometry(parsed, *record["original_size"], image_id)
        effective[image_id] = parsed
        require(sorted(item["candidate_row"] for item in row_map if item["candidate_row"] is not None)
                == list(range(1, len(parsed) + 1)), "row map is not a complete bijection")
        if result != original:
            require(record["split"] == "train", "validation changed")
            payloads[relative] = result
            changes.append({"image_id": image_id, "source_label": relative, "candidate_label": relative,
                            "source_label_sha256": digest(original), "parent_label_sha256": digest(before),
                            "candidate_label_sha256": digest(result), "source_image_sha256": record["image_sha256"],
                            "source_objects": len(original.splitlines()), "parent_objects": len(lines),
                            "candidate_objects": len(parsed), "row_map": row_map})
        if not grouped[image_id]:
            require(result == before, "unreviewed parent file changed")
    counts = totals(records, effective)
    require(counts == review["expected_counts"], "effective counts differ")
    require(counts["val"] == parent["expected_counts"]["val"], "validation counts changed")
    op_counts = dict(Counter(op["action"] for op in review["operations"]))
    require(op_counts == review["expected_operation_counts"]
            and len(changes) == review["expected_changed_files"], "review scope differs")
    manifest = {"schema_version": 1, "kind": KIND, "candidate_id": review["candidate_id"],
                "status": "reviewed_not_released", "source_revision": review["source_revision"],
                "source_manifest_sha256": parent["source_manifest_sha256"], "parent": PARENT,
                "annotation_policy": parent["annotation_policy"],
                "review": {"path": review_path.relative_to(root).as_posix(), "sha256": sha(review_path)},
                "release_ready": False, "training_recipe_id": None,
                "physical_review_is_human_approval": False, "full_dataset_certified": False,
                "expected_counts": counts, "incremental_operation_counts": op_counts, "changes": changes}
    payloads["manifest.json"] = encoded(manifest)
    return payloads, manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    modes = parser.add_subparsers(dest="mode", required=True)
    build = modes.add_parser("build")
    build.add_argument("--review", type=Path, required=True)
    build.add_argument("--output", type=Path, required=True)
    verify = modes.add_parser("verify")
    verify.add_argument("--candidate", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    if args.mode == "build":
        review_path, output = args.review.resolve(), args.output.absolute()
        require(review_path.is_relative_to(root), "review must be repository-relative")
        require(not output.exists() and not output.is_symlink(), "choose a fresh output directory")
        for protected in ("data/labels", "data/images", "data/references", "weights"):
            require(not output.resolve().is_relative_to(root / protected), "output overlaps frozen inputs")
        for frozen in (root / "data/candidates").iterdir():
            if frozen.is_dir():
                require(not output.resolve().is_relative_to(frozen.resolve()), "output is inside an existing candidate")
    else:
        output = args.candidate.resolve()
        review_path = bound(root, read_json(output / "manifest.json")["review"])
    payloads, manifest = compose(root, review_path)
    if args.mode == "build":
        output.mkdir(parents=True)
        for relative, payload in payloads.items():
            target = output / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(payload)
    verify_tree(output, payloads)
    print(json.dumps({"status": "verified", "candidate_id": manifest["candidate_id"],
                      "candidate_manifest_sha256": sha(output / "manifest.json"),
                      "changed_train_files": len(manifest["changes"]),
                      "incremental_operation_counts": manifest["incremental_operation_counts"],
                      "counts": manifest["expected_counts"], "historical_parent_verified_unchanged": True,
                      "unreviewed_tokens_preserved": True, "validation_unchanged": True,
                      "release_ready": False, "training_recipe_id": None,
                      "training_or_inference_performed": False}, indent=2))


if __name__ == "__main__":
    try:
        main()
    except (ValueError, KeyError, OSError, TypeError, IndexError) as error:
        raise SystemExit(f"observed-review composition failed: {error}")
