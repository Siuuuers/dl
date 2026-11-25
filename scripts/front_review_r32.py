#!/usr/bin/env python3
"""Replay the bounded R32 review with explicit screen-edge semantics.

All R32 amendments are development-only. Eligible R31 training supervision,
original trainer validation and frozen evaluation references remain immutable.
"""
import argparse
import copy
import json
from collections import Counter
from pathlib import Path
import sys

sys.dont_write_bytecode = True
from evaluate_pose import parse_pose_labels
from front_review_experiment import apply_operations
from front_review_r27 import FROZEN_REFERENCES, STANDARDS, pixel_geometry, semantic_fields
from verify_pose_candidate import check_geometry, counts, read_reference, require, totals
from verify_reconciled_pose_labels import bound, digest, encoded, sha, verify_tree

SOURCE_REVISION = "0a8ff31cadaf0ee0691de73fd103bba5de82b038"
PARENT = {"path": "data/reviews/fronts-2026-11-24-r31/manifest.json",
          "sha256": "bbe202ae73f2af33030b2e8e48b3710257dc1449581f59cca399ce0636c9e16a"}
REVIEW_ID = "fronts-2026-11-25-r32"
REVIEW_PATH = f"data/reviews/{REVIEW_ID}/manifest.json"
LEDGER = "results/review/front-review-20261125-r32"
AUDIT_PATH = f"{LEDGER}/row-review.json"
AUDIT_SHA256 = "061f35a0f062bed01e8d67ad798beb557d163d4d763a668819e36c7523dac499"
STANDARD_PATH = "docs/annotation-boundaries-r32.md"


def descriptor(path, payload):
    return {"path": path, "sha256": digest(payload)}


def groups(before, after, key):
    a, b = before.split(), after.split()
    require(len(a) == len(b) == 11 and a[0] == b[0], "Class or token count changed")
    result = [name for name, lo, hi in (("box", 1, 5), ("left", 5, 7), ("right", 8, 10))
              if a[lo:hi] != b[lo:hi]]
    for side, start in (("left", 5), ("right", 8)):
        old_v, new_v = float(a[start+2]), float(b[start+2])
        if old_v != new_v:
            require(key == ("new_000009", 12) and side == "left" and (old_v, new_v) == (2, 0),
                    "Unapproved visibility change")
            require(all(float(v) == 0 for v in b[start:start+3]), "Removed point must be zero triplet")
            result.append(side + "_visibility")
        if old_v == 0:
            require(a[start:start+3] == b[start:start+3]
                    and all(float(v) == 0 for v in b[start:start+3]), "Missing point changed")
    return result


def expected_review(root):
    from front_review_r31 import verify_review as verify_parent
    root = Path(root).resolve()
    verify_parent(root)
    parent = json.loads(bound(root, PARENT).read_bytes())
    audit_ref = {"path": AUDIT_PATH, "sha256": AUDIT_SHA256}
    audit = json.loads(bound(root, audit_ref).read_bytes())
    require(audit["source_review"] == PARENT and audit["source_revision"] == SOURCE_REVISION,
            "Review parent/source identity differs")
    require(audit["full_dataset_certified"] is False and audit["status"] == "accepted",
            "Review scope/acceptance differs")
    for item in audit["independent_evidence"]:
        bound(root, item)
    _, records, by_id, _, _, _ = read_reference(root, parent["source_manifest_sha256"])
    records = sorted(records, key=lambda r: (r["split"], r["id"]))
    parent_files = {r["image_id"]: r for r in parent["files"]}
    before = {}
    for record in records:
        item = parent_files[record["id"]]
        require(item["split"] == record["split"] and item["label_path"] == record["label"]
                and item["source_image_sha256"] == record["image_sha256"], "Source mapping differs")
        require(sha(root / "data" / record["image"]) == record["image_sha256"], "Original pixels differ")
        before[record["id"]] = bound((root / PARENT["path"]).parent,
            {"path": item["label_path"], "sha256": item["label_sha256"]}).read_bytes()
    operations, keys, changed_groups, split_changes = [], set(), Counter(), Counter()
    for row in audit["rows"]:
        image_id, index = row["image_id"], row["row"]
        key = (image_id, index)
        require(key not in keys, "Duplicate review identity")
        keys.add(key)
        record, item = by_id[image_id], parent_files[image_id]
        require(row["before_line"] == before[image_id].decode().splitlines()[index-1], "Before row drift")
        require(row["source_image_sha256"] == record["image_sha256"]
                and row["source_label_sha256"] == item["label_sha256"]
                and row["split"] == record["split"], "Row provenance differs")
        changed = groups(row["before_line"], row["after_line"], key)
        require(changed == row["changed_groups"] and bool(changed) == (row["action"] == "replace"),
                "Action/group mismatch")
        require(row["reason"] and row["coordinate_uncertainty"]
                and row["original_pixels_inspected"] is True, "Missing review rationale")
        for name, line in (("before_pixels", "before_line"), ("after_pixels", "after_line")):
            require(row[name] == pixel_geometry(row[line], str(key)), "Decoded display differs")
        if not changed:
            require(row["before_line"] == row["after_line"], "Retained row differs")
            continue
        require(image_id not in parent["excluded_training_image_ids"] + ["new_000015"],
                "Held/excluded training photograph changed")
        require(record["split"] == "val", "R32 may not change training supervision")
        if key in STANDARDS:
            require(set(row["changed_fields"]) <= {"ymin"}, "Only reviewed top boundaries may revise former standards")
        changed_groups.update(changed)
        split_changes[record["split"]] += 1
        operations.append({"image_id": image_id, "split": record["split"], "base_row": index,
                           "action": "replace", "before_line": row["before_line"],
                           "after_line": row["after_line"], "changed_groups": changed,
                           "changed_fields": semantic_fields(row["before_pixels"], row["after_pixels"]),
                           "reason": row["reason"]})
    operations.sort(key=lambda r: (r["image_id"], r["base_row"]))
    payloads, files, parsed, maps = {}, [], {}, {}
    for record in records:
        image_id = record["id"]
        ops = [op for op in operations if op["image_id"] == image_id]
        payload, maps[image_id] = apply_operations(before[image_id], ops, image_id)
        require(len(payload.splitlines()) == len(before[image_id].splitlines()), "Row count changed")
        require(not maps[image_id]["deleted_base_rows"]
                and all(x["current_row"] == x["base_row"] for x in maps[image_id]["rows"]), "Row IDs changed")
        parsed[image_id] = parse_pose_labels(payload.decode(), *record["original_size"], image_id)
        check_geometry(parsed[image_id], *record["original_size"], image_id)
        payloads[f"data/reviews/{REVIEW_ID}/{record['label']}"] = payload
        tally = counts(parsed[image_id])
        files.append({"image_id": image_id, "split": record["split"], "label_path": record["label"],
                      "label_sha256": digest(payload), "source_image_sha256": record["image_sha256"],
                      "objects": tally["objects"], "locatable_corners": tally["visible_endpoints"],
                      "missing_corners": tally["missing_endpoints"], "changed_in_r32": bool(ops)})
    review_counts = totals(records, parsed)
    expected_counts = copy.deepcopy(parent["expected_counts"])
    expected_counts["val"]["visible_endpoints"] -= 1
    expected_counts["val"]["missing_endpoints"] += 1
    require(review_counts == expected_counts, "Unexpected object/visibility counts")
    require(any(op["image_id"] == "new_000009" and op["base_row"] == 12
                and "left_visibility" in op["changed_groups"] for op in operations), "Required clipped-point removal absent")
    changed_ids = sorted({x["image_id"] for x in operations})
    summary = {"review_id": REVIEW_ID, "parent_review": PARENT, "reviewed_rows": len(keys),
               "replacement_rows": len(operations), "retained_rows": len(keys)-len(operations),
               "changed_photos": len(changed_ids), "changed_image_ids": changed_ids,
               "changed_rows_by_split": dict(split_changes), "field_group_changes": dict(changed_groups),
               "added_rows": 0, "deleted_rows": 0, "expected_review_counts": review_counts,
               "former_standards_revised_at_top": [list(key) for key in sorted(keys & set(STANDARDS)) if any((op["image_id"],op["base_row"]) == key for op in operations)],
               "reference_058_row7_unchanged": True, "training_labels_changed": 0, "full_dataset_certified": False,
               "training_run": False, "automatic_promotion": False}
    status = copy.deepcopy(json.loads(bound(root, parent["review_status"]).read_bytes()))
    status.update(review_id=REVIEW_ID, inherited_status=parent["review_status"],
                  current_review_scope="PR110 follow-up requests, confirmed top/bottom mapping, and bounded similar upper-lip cases; all changes are development-only.",
                  current_review_rows=[{"image_id": r["image_id"], "row": r["row"], "action": r["action"]}
                                       for r in audit["rows"]], current_row_review=audit_ref,
                  full_dataset_certified=False)
    docs = {"operations": ("operations.json", operations), "row_maps": ("row-maps.json", maps),
            "review_status": ("review-status.json", status), "amendment_summary": ("amendment-summary.json", summary),
            "request_coverage": ("request-coverage.json", audit["request_coverage"])}
    manifest = {"schema_version": 1, "kind": "original_pixel_front_review", "review_id": REVIEW_ID,
                "status": "bounded_original_pixel_review_ready", "date": "2026-11-25", "parent": PARENT,
                "source_revision": SOURCE_REVISION, "source_manifest_sha256": parent["source_manifest_sha256"],
                "component": "immutable_annotation_review", "review_only": True, "full_dataset_certified": False,
                "annotation_policy": {"path": STANDARD_PATH, "sha256": sha(root/STANDARD_PATH)},
                "boundary_rubric": {"path": STANDARD_PATH, "sha256": sha(root/STANDARD_PATH)},
                "inherited_boundary_policy": parent["annotation_policy"],
                "coordinate_semantics": audit["coordinate_semantics"],
                "training_supervision": audit["training_supervision"],
                "review_method": audit["review_method"], "review_evidence": [audit_ref],
                "independent_review_evidence": audit["independent_evidence"],
                "inherited_user_references": parent["user_references"], "user_references": [],
                "changed_image_ids": changed_ids, "reviewed_image_ids": sorted({i for i, _ in keys}),
                "requested_image_ids": sorted({x["image_id"] for x in audit["request_coverage"]["identities"]}),
                "excluded_training_image_ids": parent["excluded_training_image_ids"],
                "held_training_image_ids": ["new_000015"], "operation_counts": {"replace": len(operations), "add": 0, "delete": 0},
                "field_group_changes": dict(changed_groups), "expected_counts": review_counts,
                "frozen_references": FROZEN_REFERENCES, "files": files,
                "replay_script": {"path": "scripts/front_review_r32.py", "sha256": sha(root / "scripts/front_review_r32.py")}}
    for key, (name, value) in docs.items():
        relative = f"{LEDGER}/{name}"
        payloads[relative] = encoded(value)
        manifest[key] = descriptor(relative, payloads[relative])
    payloads[REVIEW_PATH] = encoded(manifest)
    return payloads, summary


def verify_review(root):
    root = Path(root).resolve()
    payloads, summary = expected_review(root)
    for relative, payload in payloads.items():
        require((root/relative).is_file() and not (root/relative).is_symlink()
                and (root/relative).read_bytes() == payload, "R32 replay differs: " + relative)
    label_dir = root / "data/reviews" / REVIEW_ID / "labels"
    expected = {p for p in payloads if p.startswith(f"data/reviews/{REVIEW_ID}/labels/")}
    actual = {p.relative_to(root).as_posix() for p in label_dir.rglob('*') if p.is_file() or p.is_symlink()}
    require(actual == expected and len(actual) == 157, "Unexpected R32 labels")
    return {"status": "verified", "review_manifest": REVIEW_PATH,
            "review_manifest_sha256": sha(root/REVIEW_PATH), "label_files_verified": 157, **summary}



def write_new(root, payloads):
    require(all(not (root/p).exists() for p in payloads), "Preserve existing artifacts; verify instead")
    for name, payload in payloads.items():
        path = root/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--build-review", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    if args.build_review:
        write_new(root, expected_review(root)[0])
    print(json.dumps(verify_review(root), indent=2))
