#!/usr/bin/env python3
"""Replay R33 amendments and verify the complete new-photo screening inventory.

Screening coverage, detailed coordinate adjudication and training eligibility are
separate claims. Earlier references and original trainer validation are immutable.
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
from front_review_r27 import FROZEN_REFERENCES, pixel_geometry, semantic_fields
from verify_pose_candidate import check_geometry, counts, read_reference, require, totals
from verify_reconciled_pose_labels import bound, digest, encoded, sha, verify_tree

SOURCE_REVISION = "3f4c67908f7d59754fc14c520416aef8576eae5a"
PARENT = {"path": "data/reviews/fronts-2026-11-25-r32/manifest.json",
          "sha256": "bc2e14ffeb1608be55ae39c5082ac8bed5d769b7bfc098a2592684fdb2ed654f"}
CONTROL = {"manifest": "data/candidates/front-review-2026-11-24-r31/manifest.json",
           "sha256": "8fb270a31131bf1e20c9bf5aeb05471cdcc29c03860da76e93614eeb7183fdb7"}
REVIEW_ID = "fronts-2026-11-25-r33"
REVIEW_PATH = f"data/reviews/{REVIEW_ID}/manifest.json"
LEDGER = "results/review/front-review-20261125-r33"
AUDIT_PATH = f"{LEDGER}/row-review.json"
AUDIT_SHA256 = "9ffb33f3be364f137105fa78f359f9c35372e9aa1ab170d380b99f929e7459c7"
STANDARD_PATH = "docs/annotation-boundaries-r33.md"
CANDIDATE_ID = "front-review-2026-11-25-r33"
CANDIDATE_PATH = f"data/candidates/{CANDIDATE_ID}/manifest.json"
PROTOCOL_PATH = "experiments/front-review-2026-11-25-r33.json"
DELTA_PATH = f"{LEDGER}/training-input-delta.json"


def descriptor(path, payload):
    return {"path": path, "sha256": digest(payload)}


def groups(before, after):
    a, b = before.split(), after.split()
    require(len(a) == len(b) == 11 and a[0] == b[0], "Class or token count changed")
    result = [name for name, lo, hi in (("box", 1, 5), ("left", 5, 7), ("right", 8, 10))
              if a[lo:hi] != b[lo:hi]]
    for side, start in (("left", 5), ("right", 8)):
        old_v, new_v = float(a[start+2]), float(b[start+2])
        require(old_v in (0, 2) and new_v in (0, 2), "Only observed or unavailable points allowed")
        if new_v == 0:
            require(all(float(v) == 0 for v in b[start:start+3]), "Unavailable point is not canonical zero")
        if old_v != new_v:
            result.append(side + "_visibility")
    return result


def verify_coverage(root, audit, records, parent_files, before):
    coverage = json.loads(bound(root, audit["coverage"]).read_bytes())
    require(coverage["source_review"] == PARENT and coverage["all_new_photos_screened"] is True
            and coverage["all_coordinates_individually_certified"] is False,
            "Coverage scope or source differs")
    expected = {r["id"]: r for r in records if r["id"].startswith("new_")}
    photos = coverage["photos"]
    require(len(photos) == len(expected) == 126
            and {p["image_id"] for p in photos} == set(expected), "New-photo inventory differs")
    seen_rows, empty = set(), 0
    for photo in photos:
        iid = photo["image_id"]
        record, item = expected[iid], parent_files[iid]
        lines = before[iid].decode().splitlines()
        require(photo["split"] == record["split"]
                and photo["source_image_sha256"] == record["image_sha256"]
                and photo["source_label_sha256"] == item["label_sha256"]
                and photo["objects"] == len(lines)
                and photo["full_context_screened"] is True, "Photo coverage provenance differs")
        rows = photo["rows"]
        require(len(rows) == len(lines) and {x["row"] for x in rows} == set(range(1, len(lines)+1)),
                "Missing or duplicate row coverage")
        for row in rows:
            require(row["inspection_level"] in ("full_context_screen", "native_crop_review")
                    and isinstance(row["flags"], list) and row["decision"], "Invalid review granularity")
            seen_rows.add((iid, row["row"]))
        if not lines:
            empty += 1
            require(photo["empty_scene_visually_checked"] is True, "Empty scene not visually checked")
    require(len(seen_rows) == 4178 and empty == 16, "Coverage totals differ")
    require({(r["image_id"], r["row"]) for r in audit["rows"]} <= seen_rows,
            "Detailed review falls outside screened scope")
    return coverage


def expected_review(root):
    from front_review_r32 import verify_review as verify_parent
    root = Path(root).resolve()
    verify_parent(root)
    parent = json.loads(bound(root, PARENT).read_bytes())
    audit_ref = {"path": AUDIT_PATH, "sha256": AUDIT_SHA256}
    audit = json.loads(bound(root, audit_ref).read_bytes())
    require(audit["source_review"] == PARENT and audit["source_revision"] == SOURCE_REVISION
            and audit["status"] == "accepted" and audit["full_dataset_certified"] is False,
            "Review parent, acceptance or scope differs")
    for item in audit["independent_evidence"]:
        bound(root, item)
    from triage_front_geometry_r33 import audit as numerical_audit
    require(bound(root, audit["geometry_triage"]).read_bytes() == encoded(numerical_audit(root)),
            "Complete numerical triage differs from original labels")
    _, records, by_id, _, _, _ = read_reference(root, parent["source_manifest_sha256"])
    records = sorted(records, key=lambda r: (r["split"], r["id"]))
    parent_files = {r["image_id"]: r for r in parent["files"]}
    before = {}
    for record in records:
        item = parent_files[record["id"]]
        require(item["split"] == record["split"] and item["label_path"] == record["label"]
                and item["source_image_sha256"] == record["image_sha256"], "Source mapping differs")
        require(sha(root/"data"/record["image"]) == record["image_sha256"], "Original pixels differ")
        before[record["id"]] = bound((root/PARENT["path"]).parent,
            {"path": item["label_path"], "sha256": item["label_sha256"]}).read_bytes()
    coverage = verify_coverage(root, audit, records, parent_files, before)
    operations, keys, field_groups, split_changes = [], set(), Counter(), Counter()
    for row in audit["rows"]:
        iid, index = row["image_id"], row["row"]
        key = (iid, index)
        require(key not in keys, "Duplicate detailed review identity")
        keys.add(key)
        record, item = by_id[iid], parent_files[iid]
        require(row["before_line"] == before[iid].decode().splitlines()[index-1], "Before row drift")
        require(row["source_image_sha256"] == record["image_sha256"]
                and row["source_label_sha256"] == item["label_sha256"]
                and row["split"] == record["split"], "Detailed row provenance differs")
        changed = groups(row["before_line"], row["after_line"])
        require(changed == row["changed_groups"] and bool(changed) == (row["action"] == "replace"),
                "Action or field-group mismatch")
        require(row["reason"] and row["coordinate_uncertainty"]
                and row["original_pixels_inspected"] is True, "Missing visual rationale")
        for name, line in (("before_pixels", "before_line"), ("after_pixels", "after_line")):
            require(row[name] == pixel_geometry(row[line], str(key)), "Decoded geometry differs")
        require(row["changed_fields"] == semantic_fields(row["before_pixels"], row["after_pixels"]),
                "Semantic field delta differs")
        if not changed:
            require(row["before_line"] == row["after_line"], "Retained row differs")
            continue
        require(iid.startswith("new_") and key != ("new_000058", 7), "Reference or old-series row changed")
        require(row["independent_root_endorsement"] is True, "Changed row lacks independent pixel review")
        field_groups.update(changed)
        split_changes[record["split"]] += 1
        operations.append({"image_id": iid, "split": record["split"], "base_row": index,
                           "action": "replace", "before_line": row["before_line"],
                           "after_line": row["after_line"], "changed_groups": changed,
                           "changed_fields": row["changed_fields"], "reason": row["reason"]})
    operations.sort(key=lambda r: (r["image_id"], r["base_row"]))
    payloads, files, parsed, maps = {}, [], {}, {}
    for record in records:
        iid = record["id"]
        ops = [op for op in operations if op["image_id"] == iid]
        payload, maps[iid] = apply_operations(before[iid], ops, iid)
        require(len(payload.splitlines()) == len(before[iid].splitlines()), "Object count changed")
        require(not maps[iid]["deleted_base_rows"]
                and all(x["current_row"] == x["base_row"] for x in maps[iid]["rows"]), "Stable row IDs changed")
        parsed[iid] = parse_pose_labels(payload.decode(), *record["original_size"], iid)
        check_geometry(parsed[iid], *record["original_size"], iid)
        payloads[f"data/reviews/{REVIEW_ID}/{record['label']}"] = payload
        tally = counts(parsed[iid])
        files.append({"image_id": iid, "split": record["split"], "label_path": record["label"],
                      "label_sha256": digest(payload), "source_image_sha256": record["image_sha256"],
                      "objects": tally["objects"], "locatable_corners": tally["visible_endpoints"],
                      "missing_corners": tally["missing_endpoints"], "changed_in_r33": bool(ops)})
    review_counts = totals(records, parsed)
    require(review_counts == audit["expected_counts"], "Review counts differ")
    changed_ids = sorted({x["image_id"] for x in operations})
    summary = {"review_id": REVIEW_ID, "parent_review": PARENT, "screened_photos": 126,
               "screened_nonempty_photos": 110, "screened_empty_photos": 16, "screened_rows": 4178,
               "detailed_review_rows": len(keys), "replacement_rows": len(operations),
               "retained_detailed_rows": len(keys)-len(operations), "changed_photos": len(changed_ids),
               "changed_image_ids": changed_ids, "changed_rows_by_split": dict(split_changes),
               "field_group_changes": dict(field_groups), "added_rows": 0, "deleted_rows": 0,
               "expected_review_counts": review_counts, "all_new_photos_screened": True,
               "all_coordinates_individually_certified": False, "full_dataset_certified": False,
               "reference_058_row7_unchanged": True, "training_run": False, "automatic_promotion": False}
    status = copy.deepcopy(json.loads(bound(root, parent["review_status"]).read_bytes()))
    status.update(review_id=REVIEW_ID, inherited_status=parent["review_status"],
                  current_review_scope="All 126 new-series photos screened; all 4,178 rows numerically checked; named and visually suspicious rows adjudicated at original pixels.",
                  current_review_rows=[{"image_id": r["image_id"], "row": r["row"], "action": r["action"]}
                                       for r in audit["rows"]], current_row_review=audit_ref,
                  coverage=audit["coverage"], all_new_photos_screened=True, full_dataset_certified=False)
    docs = {"operations": ("operations.json", operations), "row_maps": ("row-maps.json", maps),
            "review_status": ("review-status.json", status), "amendment_summary": ("amendment-summary.json", summary),
            "request_coverage": ("request-coverage.json", audit["request_coverage"])}
    manifest = {"schema_version": 1, "kind": "original_pixel_front_review", "review_id": REVIEW_ID,
                "status": "all_new_photos_screened_with_focused_amendments", "date": "2026-11-25",
                "parent": PARENT, "source_revision": SOURCE_REVISION,
                "source_manifest_sha256": parent["source_manifest_sha256"],
                "component": "immutable_annotation_review", "review_only": True,
                "full_dataset_certified": False, "all_new_photos_screened": True,
                "all_coordinates_individually_certified": False,
                "annotation_policy": {"path": STANDARD_PATH, "sha256": sha(root/STANDARD_PATH)},
                "boundary_rubric": {"path": STANDARD_PATH, "sha256": sha(root/STANDARD_PATH)},
                "inherited_boundary_policy": parent["annotation_policy"],
                "coordinate_semantics": audit["coordinate_semantics"],
                "review_method": audit["review_method"], "coverage": audit["coverage"],
                "review_evidence": [audit_ref], "independent_review_evidence": audit["independent_evidence"],
                "inherited_user_references": parent["user_references"], "user_references": [],
                "changed_image_ids": changed_ids,
                "reviewed_image_ids": sorted(p["image_id"] for p in coverage["photos"]),
                "requested_image_ids": ["new_000059", "new_000088", "new_000090"],
                "excluded_training_image_ids": parent["excluded_training_image_ids"],
                "held_training_image_ids": ["new_000015"],
                "operation_counts": {"replace": len(operations), "add": 0, "delete": 0},
                "field_group_changes": dict(field_groups), "expected_counts": review_counts,
                "frozen_references": FROZEN_REFERENCES, "files": files,
                "replay_script": {"path": "scripts/front_review_r33.py", "sha256": sha(root/"scripts/front_review_r33.py")}}
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
                and (root/relative).read_bytes() == payload, "R33 replay differs: " + relative)
    label_dir = root/"data/reviews"/REVIEW_ID/"labels"
    expected = {p for p in payloads if p.startswith(f"data/reviews/{REVIEW_ID}/labels/")}
    actual = {p.relative_to(root).as_posix() for p in label_dir.rglob('*') if p.is_file() or p.is_symlink()}
    require(actual == expected and len(actual) == 157, "Unexpected R33 label inventory")
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
