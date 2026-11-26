#!/usr/bin/env python3
"""Replay the R38 development-only review from immutable R37 row identities."""
import argparse
import copy
import json
from collections import Counter
from pathlib import Path
import re
import sys

sys.dont_write_bytecode = True
from evaluate_pose import parse_pose_labels
from front_review_experiment import apply_operations
from front_review_r27 import pixel_geometry, semantic_fields
from front_review_r33 import groups
from verify_pose_candidate import check_geometry, counts, read_reference, require, totals
from verify_reconciled_pose_labels import bound, digest, encoded, sha

SOURCE_REVISION = "6103bb872431ade34e23334a5940e4ccd1e70896"
PARENT = {"path": "data/reviews/fronts-2026-11-26-r37/manifest.json",
          "sha256": "35a2c6b3ddc1691eaae4b7e416144b8f68f0ab960cbd3ca8b65edd647b619471"}
REVIEW_ID = "fronts-2026-11-26-r38"
REVIEW_PATH = f"data/reviews/{REVIEW_ID}/manifest.json"
LEDGER = "results/review/front-review-20261126-r38"
AUDIT_PATH = f"{LEDGER}/row-review.json"
AUDIT_SHA256 = "fb72478f033858a86045cdf0c7e5c2a6a94bc68618a6b043625d0eac7c961adb"
STANDARD_PATH = "docs/annotation-boundaries-r38.md"
TRAINING = {"path": "data/candidates/front-review-2026-11-26-r37/manifest.json",
            "sha256": "ccfe09cae8e890b80896dac5d025f981573e5da3f49bb9f40175f2d0685de2f3"}
REPLACEMENTS = {
    ("new_000007", 47): {"tokens": {2, 4}, "fields": ["ymin"], "targets": {"ymin": 385.0}},
    ("new_000008", 7): {"tokens": {2, 4}, "fields": ["ymax"], "targets": {"ymax": 23.0}},
    ("new_000010", 29): {"tokens": {2, 4, 5, 6}, "fields": ["ymin", "left.x", "left.y"],
                           "targets": {"ymin": 666.0, "left.x": 325.0, "left.y": 714.0}},
}
DELETIONS = {("new_000086", 57), ("new_000088", 57), ("new_000053", 1),
             ("new_000010", 30), ("new_000009", 52)}
CURRENT = set(REPLACEMENTS) | DELETIONS
INHERITED = {("new_000010", 24)}


def stable_id(image_id, row):
    return f"fronts-2026-11-26-r37/{image_id}:{row}"


def image_binding(path, expected_sha256, require_original_pixels):
    """Bind actual bytes normally, or committed LFS identity for cached replay."""
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), "Original-image path differs")
    if not require_original_pixels and path.stat().st_size < 1024:
        pointer = path.read_text(encoding="ascii")
        match = re.fullmatch(r"version https://git-lfs.github.com/spec/v1\noid sha256:([0-9a-f]{64})\nsize ([1-9][0-9]*)\n", pointer)
        require(match is not None and match.group(1) == expected_sha256,
                "Committed original-image LFS identity differs")
    else:
        require(sha(path) == expected_sha256, "Original-image bytes differ")


def expected_review(root, *, require_original_pixels=True):
    root = Path(root).resolve()
    parent = json.loads(bound(root, PARENT).read_bytes())
    audit_ref = {"path": AUDIT_PATH, "sha256": AUDIT_SHA256}
    audit = json.loads(bound(root, audit_ref).read_bytes())
    require(audit["source_review"] == PARENT and audit["source_revision"] == SOURCE_REVISION
            and audit["status"] == "accepted_development_only_inherited_pending"
            and audit["full_dataset_certified"] is False, "R38 scope or source differs")
    for descriptor in audit["independent_evidence"]:
        bound(root, descriptor)
    caches = {}
    for descriptor in audit["prediction_evidence"]:
        records = [json.loads(line) for line in bound(root, descriptor).read_text().splitlines()]
        caches[descriptor["path"]] = {r["image_id"]: r for r in records if r.get("record_type") == "prediction"}
    require({(x["image_id"], x["row"]) for x in audit["current_requested_rows"]} == CURRENT
            and {(x["image_id"], x["row"]) for x in audit["inherited_pending_rows"]} == INHERITED,
            "Current or inherited request inventory differs")
    _, records, by_id, _, _, original_counts = read_reference(root, parent["source_manifest_sha256"])
    records = sorted(records, key=lambda r: (r["split"], r["id"]))
    parent_files = {r["image_id"]: r for r in parent["files"]}
    require(set(parent_files) == set(by_id) and len(records) == 157, "R37 image inventory differs")
    before = {}
    for record in records:
        item = parent_files[record["id"]]
        require(item["split"] == record["split"] and item["label_path"] == record["label"]
                and item["source_image_sha256"] == record["image_sha256"], "R37 source mapping differs")
        before[record["id"]] = bound((root/PARENT["path"]).parent,
            {"path": item["label_path"], "sha256": item["label_sha256"]}).read_bytes()
    held = {(r["image_id"], r["row"]): r for r in audit["held_request_rows"]}
    require(set(held) == INHERITED and len(held) == len(audit["held_request_rows"]), "Pending inventory differs")
    operations, keys, field_groups, dispositions = [], set(), Counter(), Counter()
    for row in audit["rows"]:
        iid, index = row["image_id"], row["row"]
        key = (iid, index)
        require(key not in keys and key in CURRENT | INHERITED, "Unknown or repeated R37 row identity")
        keys.add(key)
        record, item = by_id[iid], parent_files[iid]
        require(row["stable_id"] == stable_id(iid, index) and row["row_reference"] == "R37_before_deletions"
                and row["before_line"] == before[iid].decode().splitlines()[index-1]
                and row["before_line_sha256"] == digest(row["before_line"].encode()), "R37 row identity drift")
        require(row["split"] == record["split"] == "val"
                and row["source_image_sha256"] == record["image_sha256"]
                and row["source_label_sha256"] == item["label_sha256"], "Pixel, split or label provenance differs")
        image_binding(root/"data"/record["image"], record["image_sha256"], require_original_pixels)
        require(row["original_pixels_inspected"] and row["root_endorsed"] and row["reason"], "Missing endorsed evidence")
        require(row["before_pixels"] == pixel_geometry(row["before_line"], str(key)), "Before geometry differs")
        for cue in row["prediction_cues"]:
            cache = caches[cue["cache"]["path"]][iid]
            require(cue["cache"] in audit["prediction_evidence"]
                    and cache["image_sha256"] == record["image_sha256"]
                    and cache["predictions"][cue["prediction_id"]-1] == cue["prediction"],
                    "Prediction cue cache, image or original pID differs")
        if key in DELETIONS:
            require(row["action"] == "delete" and row["scope"] == "current_user_request"
                    and row["after_line"] is None and row["after_pixels"] is None
                    and row["user_directed_exclusion"] is True
                    and row["confirmed_duplicate_annotation"] is False
                    and row["disposition"] == "user_directed_exclusion", "Unapproved deletion or unsupported claim")
            changed, fields = ["instance"], ["instance.delete"]
        else:
            allowed = REPLACEMENTS.get(key, {"tokens": set(), "fields": [], "targets": {}})
            require(row["action"] == ("replace" if key in REPLACEMENTS else "retain"), "R38 action differs")
            require(row["after_pixels"] == pixel_geometry(row["after_line"], str(key)), "After geometry differs")
            changed = groups(row["before_line"], row["after_line"])
            fields = semantic_fields(row["before_pixels"], row["after_pixels"])
            old, new = row["before_line"].split(), row["after_line"].split()
            require(len(old) == len(new) == 11 and old[0] == new[0]
                    and fields == allowed["fields"]
                    and all(a == b for i, (a, b) in enumerate(zip(old, new)) if i not in allowed["tokens"]),
                    "Class, unrequested token or physical field changed")
            observed = dict(zip(("xmin", "ymin", "xmax", "ymax"), row["after_pixels"]["box_xyxy"]))
            observed.update({f"{side}.{axis}": row["after_pixels"]["keypoints"][j][i]
                for side, j in (("left", 0), ("right", 1)) for axis, i in (("x", 0), ("y", 1))})
            require(all(abs(observed[name]-value) < 1e-6 for name, value in allowed["targets"].items()),
                    "Endorsed coordinate differs")
            for start in (5, 8):
                require(old[start+2] == new[start+2], "Point visibility changed")
                if float(old[start+2]) == 0:
                    require(old[start:start+3] == new[start:start+3], "Unavailable point invented")
            if key in INHERITED:
                require(row["scope"] == "inherited_pending" and row["inherited_from"] == PARENT
                        and row["before_line"] == row["after_line"] and row["disposition"] == "pending"
                        and held[key]["preserve_row"] and row["pending_fields"] == held[key]["pending_fields"],
                        "Inherited pending row changed or overstated")
            else:
                require(row["scope"] == "current_user_request" and row["disposition"] == "verified"
                        and row["coordinate_uncertainty_px"] == 2, "Replacement evidence or uncertainty differs")
        require(changed == row["changed_groups"] and fields == row["changed_fields"], "Field ledger differs")
        dispositions[row["disposition"]] += 1
        if changed:
            field_groups.update(changed)
            op = {"image_id": iid, "split": "val", "base_row": index, "stable_id": row["stable_id"],
                  "action": row["action"], "before_line": row["before_line"],
                  "changed_groups": changed, "changed_fields": fields, "reason": row["reason"]}
            if row["action"] == "replace":
                op["after_line"] = row["after_line"]
            operations.append(op)
    require(keys == CURRENT | INHERITED and len(operations) == 8, "Requested R38 coverage differs")
    actions = Counter(op["action"] for op in operations)
    require(dict(actions) == {"replace": 3, "delete": 5}, "R38 operation counts differ")
    operations.sort(key=lambda r: (r["image_id"], r["base_row"]))
    payloads, files, parsed, maps, train_audit = {}, [], {}, {}, []
    for record in records:
        iid = record["id"]
        ops = [op for op in operations if op["image_id"] == iid]
        payload, lineage = apply_operations(before[iid], ops, iid)
        deleted = sorted(row for image, row in DELETIONS if image == iid)
        require(lineage["deleted_base_rows"] == deleted
                and len(payload.splitlines()) == len(before[iid].splitlines())-len(deleted), "Deletion lineage differs")
        if record["split"] == "train" or not ops:
            require(payload == before[iid], "An unrequested file changed")
        before_lines, after_lines = before[iid].decode().splitlines(), payload.decode().splitlines()
        for item in lineage["rows"]:
            base, current = item["base_row"], item["current_row"]
            require(current == base-sum(d < base for d in deleted), "R37-to-R38 row numbering differs")
            item.update(stable_id=stable_id(iid, base), action="replace" if item["changed"] else "retain")
            if not item["changed"]:
                require(before_lines[base-1] == after_lines[current-1], "A retained row changed tokens")
        lineage.update(base_review_id=parent["review_id"], current_review_id=REVIEW_ID,
            base_label_sha256=digest(before[iid]), current_label_sha256=digest(payload),
            base_row_count=len(before_lines), current_row_count=len(after_lines),
            deleted_rows=[{"base_row": d, "current_row": None, "stable_id": stable_id(iid, d),
                           "action": "delete", "before_line_sha256": digest(before_lines[d-1].encode())} for d in deleted])
        maps[iid] = lineage
        parsed[iid] = parse_pose_labels(payload.decode(), *record["original_size"], iid)
        check_geometry(parsed[iid], *record["original_size"], iid)
        payloads[f"data/reviews/{REVIEW_ID}/{record['label']}"] = payload
        tally = counts(parsed[iid])
        files.append({"image_id": iid, "split": record["split"], "label_path": record["label"],
            "label_sha256": digest(payload), "source_image_sha256": record["image_sha256"],
            "objects": tally["objects"], "locatable_corners": tally["visible_endpoints"],
            "missing_corners": tally["missing_endpoints"], "changed_in_r38": bool(ops)})
        if record["split"] == "train":
            train_audit.append({"image_id": iid, "r37_label_sha256": digest(before[iid]),
                                "r38_label_sha256": digest(payload), "byte_identical": True})
    review_counts = totals(records, parsed)
    expected = copy.deepcopy(parent["expected_counts"])
    expected["val"].update(objects=654, visible_endpoints=1220, missing_endpoints=88)
    require(review_counts == audit["expected_counts"] == expected and len(train_audit) == 137,
            "R38 inventory or unchanged training counts differ")
    require(sum(all(p[2] > 0 for p in o["keypoints"]) for record in records if record["split"] == "val"
                for o in parsed[record["id"]]) == 583, "Eligible two-corner denominator changed")
    training = json.loads(bound(root, TRAINING).read_bytes())
    for source in training["image_sources"]:
        bound(root, source["effective_label"])
    training_audit = {"schema_version": 1, "kind": "development_only_unchanged_training_audit",
        "review_parent": PARENT, "review_id": REVIEW_ID, "review_training_files_compared": 137,
        "all_review_training_files_byte_identical": True, "changed_training_rows": 0,
        "changed_training_images": [], "eligible_training_manifest": TRAINING,
        "eligible_training_counts": training["expected_counts"],
        "original_trainer_validation_counts": original_counts["val"],
        "new_training_manifest_created": False, "paired_training_run_needed": False, "files": train_audit}
    for descriptor in parent["frozen_references"].values():
        bound(root, descriptor)
    changed_ids = sorted({r["image_id"] for r in operations})
    summary = {"review_id": REVIEW_ID, "parent_review": PARENT, "reviewed_rows": len(keys),
        "current_requested_rows": 8, "inherited_pending_rows": 1, "replacement_rows": 3, "retained_rows": 1,
        "changed_photos": len(changed_ids), "changed_image_ids": changed_ids,
        "changed_rows_by_split": {"val": 8}, "field_group_changes": dict(field_groups),
        "request_dispositions": dict(dispositions), "held_request_rows": audit["held_request_rows"],
        "added_rows": 0, "deleted_rows": 5, "expected_review_counts": review_counts,
        "development_two_corner_objects": 583, "changed_training_rows": 0,
        "current_round_full_dataset_screen": False, "full_dataset_certified": False,
        "training_run": False, "automatic_promotion": False}
    status = copy.deepcopy(json.loads(bound(root, parent["review_status"]).read_bytes()))
    status.update(review_id=REVIEW_ID, status=audit["status"], inherited_status=parent["review_status"],
        current_review_scope="Eight R37 development-row requests: three replacements and five explicit case-specific exclusions. The inherited new010:24 landmark remains unchanged; training supervision is exact R37.",
        current_review_rows=[{"image_id": r["image_id"], "row": r["row"], "stable_id": r["stable_id"],
                             "action": r["action"], "disposition": r["disposition"]}
                            for r in audit["rows"] if r["scope"] == "current_user_request"],
        inherited_pending_rows=audit["inherited_pending_rows"], held_request_rows=audit["held_request_rows"],
        current_row_review=audit_ref, all_new_photos_screened=False, full_dataset_certified=False)
    docs = {"operations": ("operations.json", operations), "row_maps": ("row-maps.json", maps),
        "review_status": ("review-status.json", status), "amendment_summary": ("amendment-summary.json", summary),
        "request_coverage": ("request-coverage.json", audit["request_coverage"]),
        "training_unchanged_audit": ("training-unchanged-audit.json", training_audit)}
    manifest = {"schema_version": 1, "kind": "original_pixel_front_review", "review_id": REVIEW_ID,
        "status": audit["status"], "date": "2026-11-26", "parent": PARENT,
        "source_revision": SOURCE_REVISION, "source_manifest_sha256": parent["source_manifest_sha256"],
        "component": "immutable_annotation_review", "review_only": True, "development_only": True,
        "full_dataset_certified": False, "all_new_photos_screened": False,
        "all_coordinates_individually_certified": False,
        "inherited_screening": parent.get("coverage", parent["inherited_screening"]),
        "annotation_policy": {"path": STANDARD_PATH, "sha256": sha(root/STANDARD_PATH)},
        "boundary_rubric": {"path": STANDARD_PATH, "sha256": sha(root/STANDARD_PATH)},
        "inherited_boundary_policy": parent["annotation_policy"], "coordinate_semantics": audit["coordinate_semantics"],
        "row_identity_semantics": audit["row_identity_semantics"], "review_method": audit["review_method"],
        "review_evidence": [audit_ref], "independent_review_evidence": audit["independent_evidence"],
        "prediction_evidence": audit["prediction_evidence"],
        "inherited_user_references": parent["user_references"], "user_references": [],
        "current_requested_rows": audit["current_requested_rows"], "inherited_pending_rows": audit["inherited_pending_rows"],
        "changed_image_ids": changed_ids, "reviewed_image_ids": sorted({x[0] for x in keys}),
        "requested_image_ids": sorted({x[0] for x in CURRENT}), "held_request_rows": audit["held_request_rows"],
        "excluded_training_image_ids": parent["excluded_training_image_ids"],
        "held_training_image_ids": parent["held_training_image_ids"], "eligible_training_manifest": TRAINING,
        "operation_counts": {"replace": 3, "add": 0, "delete": 5}, "field_group_changes": dict(field_groups),
        "changed_training_rows": 0, "expected_counts": review_counts, "development_two_corner_objects": 583,
        "frozen_references": parent["frozen_references"], "files": files,
        "replay_script": {"path": "scripts/front_review_r38.py", "sha256": sha(root/"scripts/front_review_r38.py")}}
    for key, (name, value) in docs.items():
        relative = f"{LEDGER}/{name}"
        payloads[relative] = encoded(value)
        manifest[key] = {"path": relative, "sha256": digest(payloads[relative])}
    payloads[REVIEW_PATH] = encoded(manifest)
    return payloads, summary


def verify_review(root, *, require_original_pixels=True):
    root = Path(root).resolve()
    payloads, summary = expected_review(root, require_original_pixels=require_original_pixels)
    for relative, payload in payloads.items():
        require((root/relative).is_file() and not (root/relative).is_symlink()
                and (root/relative).read_bytes() == payload, "R38 replay differs: "+relative)
    labels = root/"data/reviews"/REVIEW_ID/"labels"
    require({str(p.relative_to(root)) for p in labels.rglob('*') if p.is_file()} ==
            {p for p in payloads if p.startswith(f"data/reviews/{REVIEW_ID}/labels/")}, "R38 label inventory differs")
    return {"status": "verified", "review_manifest": REVIEW_PATH,
        "review_manifest_sha256": sha(root/REVIEW_PATH), "label_files_verified": 157,
        "original_pixel_bytes_required": require_original_pixels,
        "image_binding_mode": "materialized_original_sha256" if require_original_pixels else "original_sha256_or_committed_lfs_oid",
        "visual_pixel_review_repeated_by_verifier": False, **summary}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--build-review", action="store_true")
    parser.add_argument("--allow-lfs-pointers", action="store_true",
                        help="Verify prior original-image identities for cached replay without requiring image downloads")
    args = parser.parse_args()
    if args.build_review:
        require(not args.allow_lfs_pointers, "Building the accepted review requires original image bytes")
        payloads, _ = expected_review(args.root)
        require(all(not (args.root/p).exists() for p in payloads), "Preserve existing review; verify instead")
        for relative, payload in payloads.items():
            path = args.root/relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(payload)
    print(json.dumps(verify_review(args.root, require_original_pixels=not args.allow_lfs_pointers), indent=2))
