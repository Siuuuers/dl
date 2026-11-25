#!/usr/bin/env python3
"""Replay the verified R36 subset and preserve every unresolved field exactly."""
import argparse
import copy
import json
from collections import Counter
from pathlib import Path
import sys

sys.dont_write_bytecode = True
from evaluate_pose import parse_pose_labels
from front_review_experiment import apply_operations
from front_review_r27 import pixel_geometry, semantic_fields
from front_review_r33 import groups
from verify_pose_candidate import check_geometry, counts, read_reference, require, totals
from verify_reconciled_pose_labels import bound, digest, encoded, sha

SOURCE_REVISION = "635e6a75d573f87571eae99b3631b95762ec4567"
PARENT = {"path": "data/reviews/fronts-2026-11-25-r35/manifest.json",
          "sha256": "1ec7e4826d3e14c61d83e5267fdb798255348858c834e64ec2d66347076f9019"}
REVIEW_ID = "fronts-2026-11-26-r36"
REVIEW_PATH = f"data/reviews/{REVIEW_ID}/manifest.json"
LEDGER = "results/review/front-review-20261126-r36"
AUDIT_PATH = f"{LEDGER}/row-review.json"
AUDIT_SHA256 = "a79b4961eac7c716863555678b4e9df960d980438625ae47aa3cbfe3b7c1aee6"
STANDARD_PATH = "docs/annotation-boundaries-r36.md"
REQUIRED = {("new_000006", 53), ("new_000034", 34), ("new_000034", 35),
            ("new_000010", 24), ("new_000057", 1)}


def expected_review(root):
    root = Path(root).resolve()
    parent = json.loads(bound(root, PARENT).read_bytes())
    audit_ref = {"path": AUDIT_PATH, "sha256": AUDIT_SHA256}
    audit = json.loads(bound(root, audit_ref).read_bytes())
    require(audit["source_review"] == PARENT and audit["source_revision"] == SOURCE_REVISION
            and audit["status"] == "verified_subset_with_pending_requests"
            and audit["full_dataset_certified"] is False, "R36 acceptance or source scope differs")
    for descriptor in audit["independent_evidence"]:
        bound(root, descriptor)
    _, records, by_id, _, _, _ = read_reference(root, parent["source_manifest_sha256"])
    records = sorted(records, key=lambda r: (r["split"], r["id"]))
    parent_files = {r["image_id"]: r for r in parent["files"]}
    require(set(parent_files) == set(by_id), "R35 image inventory differs")
    before = {}
    for record in records:
        item = parent_files[record["id"]]
        require(item["split"] == record["split"] and item["label_path"] == record["label"]
                and item["source_image_sha256"] == record["image_sha256"], "Source mapping differs")
        before[record["id"]] = bound((root/PARENT["path"]).parent,
            {"path": item["label_path"], "sha256": item["label_sha256"]}).read_bytes()
    held = {(r["image_id"], r["row"]): r for r in audit["held_request_rows"]}
    require(len(held) == len(audit["held_request_rows"]) and set(held) <= REQUIRED,
            "Unknown or repeated pending request identity")
    operations, keys, field_groups, split_changes, dispositions = [], set(), Counter(), Counter(), Counter()
    for row in audit["rows"]:
        iid, index = row["image_id"], row["row"]
        key = (iid, index)
        require(key not in keys and key in REQUIRED, "Repeated or unknown review identity")
        keys.add(key)
        record, item = by_id[iid], parent_files[iid]
        require(row["before_line"] == before[iid].decode().splitlines()[index-1], "Before row drift")
        require(sha(root/"data"/record["image"]) == row["source_image_sha256"] == record["image_sha256"]
                and row["source_label_sha256"] == item["label_sha256"]
                and row["split"] == record["split"], "Review pixel/label provenance differs")
        require(row["original_pixels_inspected"] and row["root_endorsed"] and row["reason"],
                "Missing observed evidence")
        require(row["before_pixels"] == pixel_geometry(row["before_line"], str(key)), "Before geometry differs")
        require(row["action"] in {"replace", "retain"}, "R36 permits no instance or visibility changes")
        changed = groups(row["before_line"], row["after_line"])
        fields = semantic_fields(row["before_pixels"], row["after_pixels"])
        require(bool(changed) == (row["action"] == "replace"), "Action differs")
        require(not any(name.endswith("visibility") for name in changed), "Unrequested visibility change")
        require(row["after_pixels"] == pixel_geometry(row["after_line"], str(key)), "After geometry differs")
        require(changed == row["changed_groups"] and fields == row["changed_fields"], "Field delta differs")
        old_tokens, new_tokens = row["before_line"].split(), row["after_line"].split()
        require(old_tokens[0] == new_tokens[0] and len(old_tokens) == len(new_tokens) == 11,
                "Class or schema changed")
        require(old_tokens[5:] == new_tokens[5:], "R36 preserves every point token")
        for start in (5, 8):
            require(old_tokens[start+2] == new_tokens[start+2], "Point visibility changed")
            if float(old_tokens[start+2]) == 0:
                require(old_tokens[start:start+3] == new_tokens[start:start+3], "Unavailable point invented")
        pending = held.get(key)
        require(bool(pending) == bool(row["pending_fields"]), "Pending field inventory differs")
        if pending:
            require(pending["pending_fields"] == row["pending_fields"] and pending["reason"],
                    "Pending request evidence differs")
            require(not (set(fields) & set(pending["pending_fields"])), "Unresolved field was changed")
            if pending["preserve_row"]:
                require(row["before_line"] == row["after_line"], "Fully held row changed")
            require(row["disposition"] == ("pending" if not changed else "partially_verified"),
                    "Pending decision overstated")
        else:
            require(row["disposition"] == "verified", "Verified decision scope differs")
        dispositions[row["disposition"]] += 1
        if changed:
            require(row["coordinate_uncertainty_px"] is not None, "Missing coordinate ambiguity")
            field_groups.update(changed)
            split_changes[record["split"]] += 1
            operations.append({"image_id": iid, "split": record["split"], "base_row": index,
                "action": "replace", "before_line": row["before_line"], "after_line": row["after_line"],
                "changed_groups": changed, "changed_fields": fields, "reason": row["reason"]})
        else:
            require(row["before_line"] == row["after_line"], "Retention changed tokens")
    require(keys == REQUIRED and operations, "User-request coverage or verified subset differs")
    operations.sort(key=lambda r: (r["image_id"], r["base_row"]))
    payloads, files, parsed, maps = {}, [], {}, {}
    for record in records:
        iid = record["id"]
        ops = [op for op in operations if op["image_id"] == iid]
        payload, maps[iid] = apply_operations(before[iid], ops, iid)
        require(len(payload.splitlines()) == len(before[iid].splitlines())
                and not maps[iid]["deleted_base_rows"]
                and all(m["current_row"] == m["base_row"] for m in maps[iid]["rows"]),
                "R36 must preserve instance identities")
        if iid == "new_000009":
            require(payload == before[iid] and len(payload.splitlines()) == 55,
                    "Previously excluded instances or retained rows changed")
        parsed[iid] = parse_pose_labels(payload.decode(), *record["original_size"], iid)
        check_geometry(parsed[iid], *record["original_size"], iid)
        payloads[f"data/reviews/{REVIEW_ID}/{record['label']}"] = payload
        tally = counts(parsed[iid])
        files.append({"image_id": iid, "split": record["split"], "label_path": record["label"],
            "label_sha256": digest(payload), "source_image_sha256": record["image_sha256"],
            "objects": tally["objects"], "locatable_corners": tally["visible_endpoints"],
            "missing_corners": tally["missing_endpoints"], "changed_in_r36": bool(ops)})
    review_counts = totals(records, parsed)
    require(review_counts == audit["expected_counts"] == parent["expected_counts"], "Review inventory changed")
    for descriptor in parent["frozen_references"].values():
        bound(root, descriptor)
    changed_ids = sorted({r["image_id"] for r in operations})
    summary = {"review_id": REVIEW_ID, "parent_review": PARENT,
        "reviewed_rows": len(keys), "replacement_rows": len(operations), "retained_rows": len(keys)-len(operations),
        "changed_photos": len(changed_ids), "changed_image_ids": changed_ids,
        "changed_rows_by_split": dict(split_changes), "field_group_changes": dict(field_groups),
        "request_dispositions": dict(dispositions), "held_request_rows": audit["held_request_rows"],
        "added_rows": 0, "deleted_rows": 0, "expected_review_counts": review_counts,
        "current_round_full_dataset_screen": False, "full_dataset_certified": False,
        "training_run": False, "automatic_promotion": False}
    status = copy.deepcopy(json.loads(bound(root, parent["review_status"]).read_bytes()))
    status.update(review_id=REVIEW_ID, inherited_status=parent["review_status"],
        current_review_scope="Five user-named rows; only verified fields change. Missing markup and ambiguous point names remain explicit pending requests.",
        current_review_rows=[{"image_id": r["image_id"], "row": r["row"], "action": r["action"],
                             "disposition": r["disposition"]} for r in audit["rows"]],
        held_request_rows=audit["held_request_rows"], current_row_review=audit_ref,
        all_new_photos_screened=False, full_dataset_certified=False)
    docs = {"operations": ("operations.json", operations), "row_maps": ("row-maps.json", maps),
        "review_status": ("review-status.json", status), "amendment_summary": ("amendment-summary.json", summary),
        "request_coverage": ("request-coverage.json", audit["request_coverage"])}
    manifest = {"schema_version": 1, "kind": "original_pixel_front_review", "review_id": REVIEW_ID,
        "status": "verified_subset_with_pending_requests", "date": "2026-11-26", "parent": PARENT,
        "source_revision": SOURCE_REVISION, "source_manifest_sha256": parent["source_manifest_sha256"],
        "component": "immutable_annotation_review", "review_only": True, "full_dataset_certified": False,
        "all_new_photos_screened": False, "all_coordinates_individually_certified": False,
        "inherited_screening": parent.get("coverage", parent["inherited_screening"]),
        "annotation_policy": {"path": STANDARD_PATH, "sha256": sha(root/STANDARD_PATH)},
        "boundary_rubric": {"path": STANDARD_PATH, "sha256": sha(root/STANDARD_PATH)},
        "inherited_boundary_policy": parent["annotation_policy"], "coordinate_semantics": audit["coordinate_semantics"],
        "review_method": audit["review_method"], "review_evidence": [audit_ref],
        "independent_review_evidence": audit["independent_evidence"],
        "inherited_user_references": parent["user_references"], "user_references": [],
        "changed_image_ids": changed_ids, "reviewed_image_ids": sorted({x[0] for x in keys}),
        "requested_image_ids": sorted({x[0] for x in keys}), "held_request_rows": audit["held_request_rows"],
        "excluded_training_image_ids": parent["excluded_training_image_ids"],
        "held_training_image_ids": parent["held_training_image_ids"],
        "operation_counts": {"replace": len(operations), "add": 0, "delete": 0},
        "field_group_changes": dict(field_groups), "expected_counts": review_counts,
        "frozen_references": parent["frozen_references"], "files": files,
        "replay_script": {"path": "scripts/front_review_r36.py", "sha256": sha(root/"scripts/front_review_r36.py")}}
    for key, (name, value) in docs.items():
        relative = f"{LEDGER}/{name}"
        payloads[relative] = encoded(value)
        manifest[key] = {"path": relative, "sha256": digest(payloads[relative])}
    payloads[REVIEW_PATH] = encoded(manifest)
    return payloads, summary


def verify_review(root):
    root = Path(root).resolve()
    payloads, summary = expected_review(root)
    for relative, payload in payloads.items():
        require((root/relative).is_file() and not (root/relative).is_symlink()
                and (root/relative).read_bytes() == payload, "R36 replay differs: "+relative)
    labels = root/"data/reviews"/REVIEW_ID/"labels"
    require({str(p.relative_to(root)) for p in labels.rglob('*') if p.is_file()} ==
            {p for p in payloads if p.startswith(f"data/reviews/{REVIEW_ID}/labels/")}, "R36 label inventory differs")
    return {"status": "verified", "review_manifest": REVIEW_PATH,
        "review_manifest_sha256": sha(root/REVIEW_PATH), "label_files_verified": 157, **summary}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--build-review", action="store_true")
    args = parser.parse_args()
    if args.build_review:
        payloads, _ = expected_review(args.root)
        require(all(not (args.root/p).exists() for p in payloads), "Preserve existing review; verify instead")
        for relative, payload in payloads.items():
            path = args.root/relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(payload)
    print(json.dumps(verify_review(args.root), indent=2))
