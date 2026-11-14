#!/usr/bin/env python3
"""Build/verify an observed-label review revision of the frozen reconciled labels.

The parent verifier stays unchanged. This small follow-up accepts only explicit
box replacements, existing visible-corner refinements, and removals of
both-corners-missing rows. It cannot fill hidden corners, add inferred objects,
modify validation, stage training, or run a model.
"""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
from verify_reconciled_pose_labels import (bound, digest, encoded, expected,
                                          read_json, require, sha, verify_tree)
from verify_pose_candidate import check_geometry, read_reference, totals
from evaluate_pose import parse_pose_labels

PARENT = {
    "path": "data/candidates/reconciled-labels-2026-11-14/manifest.json",
    "sha256": "e20782e51ab010f5ab2ca045dec37768ff32b547ced8be3a039f30a76c214cc0",
}
KIND = "training_observed_review_revision"
EDGES = ("left", "top", "right", "bottom")


def box(tokens, size):
    width, height = size
    cx, cy, w, h = map(float, tokens[1:5])
    return [(cx - w / 2) * width, (cy - h / 2) * height,
            (cx + w / 2) * width, (cy + h / 2) * height]


def revision(root, review_path):
    review = read_json(review_path)
    require(review.get("schema_version") == 1 and review.get("kind") == KIND,
            "unexpected review revision schema/kind")
    require(review["parent"] == PARENT, "unexpected reconciled parent")
    require(review["release_ready"] is False and review["training_recipe_id"] is None,
            "this review cannot claim training release readiness")
    for descriptor in review["frozen_code"]:
        bound(root, descriptor)
    parent_path = bound(root, PARENT)
    parent = read_json(parent_path)
    parent_payloads, rebuilt = expected(root, bound(root, parent["specification"]))
    verify_tree(parent_path.parent, parent_payloads)
    require(parent == rebuilt, "parent reconstruction differs")
    _, records, by_id, source_bytes, _, _ = read_reference(root, parent["source_manifest_sha256"])
    for descriptor in review["evidence"]:
        bound(root, descriptor)
    operations = review["operations"]
    require(isinstance(operations, list) and operations, "no accepted operations")
    grouped, seen = defaultdict(list), set()
    for op in operations:
        key = (op["image_id"], op["original_row"])
        require(key not in seen, "multiple operations on one original row")
        seen.add(key)
        require(op["status"] == "accepted" and
                op["field"] in ("bbox", "delete_object", "left_endpoint", "right_endpoint"),
                "unsupported or unaccepted observed-label operation")
        require(op["coordinate_evidence"] == "original_image_review", "inferred geometry is ineligible")
        require(isinstance(op["reason"], str) and op["reason"].strip(), "operation needs physical evidence")
        record = by_id[op["image_id"]]
        require(record["split"] == "train", "validation edit forbidden")
        require(op["source_image_sha256"] == record["image_sha256"] and
                op["source_label_sha256"] == record["label_sha256"], "original identity differs")
        grouped[op["image_id"]].append(op)
    parent_changes = {c["image_id"]: c for c in parent["changes"]}
    payloads, changes, effective = {}, [], {}
    for image_id, record in sorted(by_id.items()):
        relative = record["label"]
        original = source_bytes[image_id]
        before = parent_payloads.get(relative, original)
        lines = before.decode().splitlines(keepends=True)
        prior = parent_changes.get(image_id)
        mapping = (prior["row_map"] if prior else
                   [{"original_row": i, "candidate_row": i} for i in range(1, len(lines) + 1)])
        original_to_parent = {r["original_row"]: r["candidate_row"] for r in mapping
                              if r["original_row"] is not None}
        removed, replaced = set(), {}
        for op in grouped[image_id]:
            row = op["original_row"]
            require(type(row) is int and original_to_parent.get(row) is not None,
                    "operation targets an absent original row")
            current = original_to_parent[row]
            require(type(op["parent_row"]) is int and op["parent_row"] == current,
                    "parent row map differs")
            old_line = lines[current - 1].rstrip("\r\n")
            require(old_line == op["before_text"], "before row differs from exact parent")
            tokens = old_line.split()
            if op["field"] == "delete_object":
                require(tokens[5:] == ["0.00000000", "0.00000000", "0",
                                       "0.00000000", "0.00000000", "0"],
                        "deletion requires canonical both-missing endpoints")
                require(op["after_text"] is None, "deletion has unexpected replacement")
                retained = op.get("retained_original_row")
                if retained is not None:
                    require(retained != row and original_to_parent.get(retained) is not None,
                            "duplicate has no retained owner")
                    require(not any(x["original_row"] == retained and x["field"] == "delete_object"
                                    for x in grouped[image_id]), "retained duplicate owner removed")
                removed.add(current)
                continue
            after_text = op["after_text"]
            require(isinstance(after_text, str) and after_text and after_text.splitlines() == [after_text],
                    "replacement must be one nonempty physical line")
            after = after_text.split()
            if op["field"] in ("left_endpoint", "right_endpoint"):
                start = 5 if op["field"] == "left_endpoint" else 8
                require(len(after) == 11 and after[:start] == tokens[:start] and
                        after[start + 3:] == tokens[start + 3:] and
                        tokens[start + 2] == after[start + 2] == "2",
                        "endpoint operation changes unrelated tokens or visibility")
                changed = op["changed_coordinates"]
                require(isinstance(changed, list) and changed and len(set(changed)) == len(changed)
                        and set(changed) <= {"x", "y"}, "invalid changed endpoint coordinates")
                target = op["after_keypoint_xyv_px"]
                require(len(target) == 3 and target[2] == 2, "visible target coordinate required")
                for offset, (axis, size) in enumerate(zip(("x", "y"), record["original_size"])):
                    require(abs(float(after[start + offset]) * size - target[offset]) <= 0.00002,
                            "encoded endpoint differs from reviewed coordinate")
                    if axis not in changed:
                        require(after[start + offset] == tokens[start + offset],
                                "undeclared endpoint coordinate changed")
                require(after[start:start + 3] != tokens[start:start + 3], "ineffective endpoint operation")
                replaced[current] = after_text + "\n"
                continue
            require(len(after) == 11 and after[0] == tokens[0] and after[5:] == tokens[5:],
                    "box operation changes class or endpoint tokens")
            require(after[1:5] != tokens[1:5], "ineffective box operation")
            old_box, new_box = box(tokens, record["original_size"]), box(after, record["original_size"])
            declared = op["changed_bounds"]
            require(isinstance(declared, list) and declared and len(set(declared)) == len(declared)
                    and set(declared) <= set(EDGES), "invalid changed bounds")
            target = op["after_bbox_xyxy_px"]
            require(len(target) == 4 and all(abs(a - b) <= 0.00002 for a, b in zip(new_box, target)),
                    "encoded replacement does not match reviewed pixel bounds")
            for edge, a, b in zip(EDGES, old_box, new_box):
                if edge not in declared:
                    require(abs(a - b) <= 0.00002, "an undeclared edge changed")
            replaced[current] = op["after_text"] + "\n"
        result = "".join(replaced.get(i, line) for i, line in enumerate(lines, 1)
                         if i not in removed).encode()
        row_map = []
        for item in mapping:
            item = dict(item)
            current = item["candidate_row"]
            item["parent_row"] = current
            item["candidate_row"] = (None if current is None or current in removed else
                                     current - sum(r < current for r in removed))
            row_map.append(item)
        parsed = parse_pose_labels(result.decode(), *record["original_size"], image_id)
        check_geometry(parsed, *record["original_size"], image_id)
        effective[image_id] = parsed
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
    require(counts == review["expected_counts"], "effective counts differ from accepted review")
    require(counts["val"] == parent["expected_counts"]["val"], "validation counts changed")
    require(counts["train"]["visible_endpoints"] == parent["expected_counts"]["train"]["visible_endpoints"],
            "visible endpoint count changed")
    op_counts = dict(Counter(op["field"] for op in operations))
    require(op_counts == review["expected_operation_counts"] and len(changes) == review["expected_changed_files"],
            "review scope differs")
    manifest = {"schema_version": 1, "kind": KIND, "candidate_id": review["candidate_id"],
                "status": "reviewed_not_released", "source_revision": review["source_revision"],
                "source_manifest_sha256": parent["source_manifest_sha256"], "parent": PARENT,
                "annotation_policy": parent["annotation_policy"],
                "review": {"path": review_path.relative_to(root).as_posix(), "sha256": sha(review_path)},
                "release_ready": False, "training_recipe_id": None,
                "physical_review_is_human_approval": False, "full_dataset_certified": False,
                "expected_counts": counts, "incremental_operation_counts": op_counts,
                "changes": changes}
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
                require(not output.resolve().is_relative_to(frozen.resolve()),
                        "output is nested inside an existing candidate")
    else:
        output = args.candidate.resolve()
        review_path = bound(root, read_json(output / "manifest.json")["review"])
    payloads, manifest = revision(root, review_path)
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
                      "counts": manifest["expected_counts"], "parent_verified_unchanged": True,
                      "unreviewed_endpoint_tokens_preserved": True, "visibility_states_preserved": True,
                      "validation_unchanged": True,
                      "release_ready": False, "training_recipe_id": None,
                      "training_or_inference_performed": False}, indent=2))


if __name__ == "__main__":
    try:
        main()
    except (ValueError, KeyError, OSError, TypeError, IndexError) as error:
        raise SystemExit(f"observed-review verification failed: {error}")
